from __future__ import annotations

import inspect
from typing import Dict, Optional

import torch
from transformers import AutoModel


class ModelForClassification(torch.nn.Module):
    """
    Transformer classifier for text + optional numeric metadata.

    Supported pooling strategies:
    - "cls": first token embedding, good default for normal BERT-style classification
    - "mean": mask-aware mean pooling, recommended for SBERT-like sentence embedding models
    - "pooler": pooler_output if the model has it, otherwise falls back to CLS
    """

    def __init__(self, model_path: str, config: Dict):
        super().__init__()

        self.model_name = model_path
        self.config = dict(config)
        self.n_classes = int(config["num_classes"])
        self.dropout_rate = float(config.get("dropout_rate", 0.2))
        self.pooling = config.get("pooling", "cls")
        self.meta_dim = int(config.get("meta_dim", 0))

        self.bert = AutoModel.from_pretrained(self.model_name)

        if config.get("gradient_checkpointing", False) and hasattr(self.bert, "gradient_checkpointing_enable"):
            self.bert.gradient_checkpointing_enable()
            # HF models usually require this when gradient checkpointing is enabled.
            if hasattr(self.bert.config, "use_cache"):
                self.bert.config.use_cache = False

        self._bert_forward_params = set(inspect.signature(self.bert.forward).parameters.keys())

        hidden = int(self.bert.config.hidden_size)
        classifier_input = hidden + self.meta_dim
        inner = max(hidden // 2, self.n_classes * 8)

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(self.dropout_rate),
            torch.nn.Linear(classifier_input, inner),
            torch.nn.GELU(),
            torch.nn.Dropout(self.dropout_rate),
            torch.nn.Linear(inner, self.n_classes),
        )

    @staticmethod
    def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def _pool(self, outputs, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return self.mean_pooling(outputs.last_hidden_state, attention_mask)

        if self.pooling == "pooler" and getattr(outputs, "pooler_output", None) is not None:
            return outputs.pooler_output

        # Default: CLS representation.
        return outputs.last_hidden_state[:, 0]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        meta_features: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # RoBERTa-like models often do not accept token_type_ids.
        if token_type_ids is not None and "token_type_ids" in self._bert_forward_params:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.bert(**kwargs)
        features = self._pool(outputs, attention_mask)

        if self.meta_dim > 0:
            if meta_features is None:
                raise ValueError("meta_features are required because config['meta_dim'] > 0")
            meta_features = meta_features.to(features.device).float()
            if meta_features.ndim == 1:
                meta_features = meta_features.unsqueeze(1)
            features = torch.cat([features, meta_features], dim=1)

        return self.classifier(features)
