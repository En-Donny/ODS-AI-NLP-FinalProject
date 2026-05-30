from __future__ import annotations

import copy
import gc
import os
from typing import Dict, Optional

import numpy as np
import torch
from numpy import asarray
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from tqdm.notebook import tqdm
from transformers import get_linear_schedule_with_warmup

from model import ModelForClassification


class Trainer:
    def __init__(self, config: Dict):
        self.config = dict(config)
        self.n_epochs = int(config["n_epochs"])
        self.optimizer = None
        self.scheduler = None
        self.model = None
        self.history = None

        self.loss_fn = config.get("crit", None) or CrossEntropyLoss()
        if config.get("crit", None) is not None:
            print("Using custom/balanced loss")

        self.device = torch.device(config["device"])
        self.verbose = bool(config.get("verbose", True))
        self.early_stopping = bool(config.get("early_stopping", True))
        self.patience = int(config.get("patience", 2))

        self.save_model_dir = config.get("save_model_dir", "./checkpoints")
        self.save_model_name = config.get("save_model_name", "bert_classifier")
        self.save_metric = config.get("save_metric", "f1")
        self.greater_is_better = bool(config.get("greater_is_better", True))
        self.keep_only_best = bool(config.get("keep_only_best", True))
        self.best_checkpoint_path: Optional[str] = None

        self.freeze_embeddings = bool(config.get("freeze_embeddings", False))
        self.freeze_first_n_layers = int(config.get("freeze_first_n_layers", 0))
        self.use_amp = bool(config.get("use_amp", True)) and self.device.type == "cuda"
        self.max_grad_norm = float(config.get("max_grad_norm", 1.0))
        scaler_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scaler = torch.amp.GradScaler(scaler_device, enabled=self.use_amp)

    def _apply_freezing(self):
        if self.model is None:
            return

        if self.freeze_embeddings:
            for name, param in self.model.bert.named_parameters():
                if "embeddings" in name:
                    param.requires_grad = False

        if self.freeze_first_n_layers > 0:
            for name, param in self.model.bert.named_parameters():
                if "encoder.layer" in name:
                    try:
                        layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
                    except Exception:
                        continue
                    if layer_num < self.freeze_first_n_layers:
                        param.requires_grad = False

    def _build_optimizer_and_scheduler(self, train_dataloader):
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        named_params = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in named_params if not any(nd in n for nd in no_decay)],
                "weight_decay": float(self.config.get("weight_decay", 0.01)),
            },
            {
                "params": [p for n, p in named_params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=float(self.config["lr"]))

        total_steps = len(train_dataloader) * self.n_epochs
        warmup_ratio = float(self.config.get("warmup_ratio", 0.1))
        warmup_steps = int(total_steps * warmup_ratio)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    @staticmethod
    def _is_better(current: float, best: Optional[float], greater_is_better: bool = True) -> bool:
        if best is None:
            return True
        return current > best if greater_is_better else current < best

    def fit(self, model, train_dataloader, val_dataloader):
        self.model = model.to(self.device)
        self._apply_freezing()
        self._build_optimizer_and_scheduler(train_dataloader)

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "val_macro_f1": [],
            "val_weighted_f1": [],
        }

        os.makedirs(self.save_model_dir, exist_ok=True)

        best_metric_value = None
        no_improve = 0

        for epoch in range(self.n_epochs):
            print(f"Epoch {epoch + 1}/{self.n_epochs}")

            train_info = self.train_epoch(train_dataloader)
            val_info = self.val_epoch(val_dataloader)

            self.history["train_loss"].append(train_info["loss"])
            self.history["val_loss"].append(val_info["loss"])
            self.history["val_acc"].append(val_info["acc"])
            self.history["val_macro_f1"].append(val_info["macro_f1"])
            self.history["val_weighted_f1"].append(val_info["weighted_f1"])

            current_metric = val_info[self.save_metric]
            if self._is_better(current_metric, best_metric_value, self.greater_is_better):
                print(
                    f"New best {self.save_metric}: "
                    f"{current_metric:.5f} (previous: {best_metric_value})"
                )
                best_metric_value = current_metric
                no_improve = 0
                self._save_best_checkpoint(epoch, current_metric, val_info)
            else:
                no_improve += 1
                if self.early_stopping and no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        if self.best_checkpoint_path and os.path.exists(self.best_checkpoint_path):
            print(f"Loading best checkpoint: {self.best_checkpoint_path}")
            checkpoint = torch.load(self.best_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])

        return self.model.eval()

    def _forward_batch(self, batch):
        ids = batch["ids"].to(self.device, dtype=torch.long)
        mask = batch["mask"].to(self.device, dtype=torch.long)
        meta = batch.get("meta_features", None)
        if meta is not None:
            meta = meta.to(self.device, dtype=torch.float)

        token_type_ids = batch.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device, dtype=torch.long)

        return self.model(
            input_ids=ids,
            attention_mask=mask,
            meta_features=meta,
            token_type_ids=token_type_ids,
        )

    def train_epoch(self, train_dataloader):
        self.model.train()
        losses = []

        iterator = tqdm(train_dataloader) if self.verbose else train_dataloader

        for batch in iterator:
            targets = batch["targets"].to(self.device, dtype=torch.long)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = self._forward_batch(batch)
                    loss = self.loss_fn(outputs, targets)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
            else:
                outputs = self._forward_batch(batch)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()

            loss_val = float(loss.item())
            losses.append(loss_val)

            if self.verbose:
                iterator.set_description(f"Loss={loss_val:.4f}")

        mean_loss = float(np.mean(losses)) if losses else 0.0
        print(f"Train: Loss={mean_loss:.4f}")
        return {"loss": mean_loss}

    def val_epoch(self, val_dataloader):
        self.model.eval()
        all_logits = []
        all_labels = []
        losses = []

        iterator = tqdm(val_dataloader) if self.verbose else val_dataloader

        with torch.no_grad():
            for batch in iterator:
                targets = batch["targets"].to(self.device, dtype=torch.long)

                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        outputs = self._forward_batch(batch)
                        loss = self.loss_fn(outputs, targets)
                else:
                    outputs = self._forward_batch(batch)
                    loss = self.loss_fn(outputs, targets)

                all_logits.append(outputs.detach().float().cpu())
                all_labels.append(targets.detach().cpu())
                losses.append(float(loss.item()))

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        preds = all_logits.argmax(1).numpy()
        labels = all_labels.numpy()

        acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average="macro")
        weighted_f1 = f1_score(labels, preds, average="weighted")
        loss = float(np.mean(losses)) if losses else 0.0

        print(f"Validation: Loss={loss:.4f}; Acc={acc:.4f}; Macro-F1={macro_f1:.4f}; Weighted-F1={weighted_f1:.4f}")

        return {
            "acc": float(acc),
            "loss": loss,
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
        }

    def predict(self, test_dataloader):
        if self.model is None:
            raise RuntimeError("You should train the model first")

        self.model.eval()
        predictions = []
        logits_all = []

        iterator = tqdm(test_dataloader) if self.verbose else test_dataloader

        with torch.no_grad():
            for batch in iterator:
                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        outputs = self._forward_batch(batch)
                else:
                    outputs = self._forward_batch(batch)

                outputs = outputs.detach().float().cpu()
                predictions.extend(outputs.argmax(1).tolist())
                logits_all.append(outputs)

        return asarray(predictions)

    def _save_best_checkpoint(self, epoch: int, metric_value: float, val_info: Dict):
        safe_metric = f"{metric_value:.5f}".replace(".", "_")
        filename = f"{self.save_model_name}_epoch_{epoch + 1:02d}_{self.save_metric}_{safe_metric}.ckpt"
        path = os.path.join(self.save_model_dir, filename)

        if self.keep_only_best and self.best_checkpoint_path and os.path.exists(self.best_checkpoint_path):
            try:
                os.remove(self.best_checkpoint_path)
                print(f"Deleted previous checkpoint: {self.best_checkpoint_path}")
            except OSError as e:
                print(f"Could not delete previous checkpoint: {e}")

        self.save(path, epoch=epoch, metric_value=metric_value, val_info=val_info)
        self.best_checkpoint_path = path
        print(f"Saved best checkpoint: {path}")

    def save(self, path: str, epoch: Optional[int] = None, metric_value: Optional[float] = None, val_info: Optional[Dict] = None):
        if self.model is None:
            raise RuntimeError("You should train the model first")

        trainer_config = copy.deepcopy(self.config)
        trainer_config.pop("crit", None)

        checkpoint = {
            "config": self.model.config,
            "trainer_config": trainer_config,
            "model_name": self.model.model_name,
            "model_state_dict": self.model.state_dict(),
            "epoch": epoch,
            "best_metric": self.save_metric,
            "best_metric_value": metric_value,
            "val_info": val_info,
            "history": self.history,
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str, map_location: Optional[str] = None):
        ckpt = torch.load(path, map_location=map_location)
        keys = ["config", "trainer_config", "model_state_dict", "model_name"]
        for key in keys:
            if key not in ckpt:
                raise RuntimeError(f"Missing key {key} in checkpoint")

        new_model = ModelForClassification(ckpt["model_name"], ckpt["config"])
        new_model.load_state_dict(ckpt["model_state_dict"])

        new_trainer = cls(ckpt["trainer_config"])
        new_trainer.model = new_model.to(new_trainer.device)
        new_trainer.best_checkpoint_path = path
        new_trainer.history = ckpt.get("history")
        return new_trainer
