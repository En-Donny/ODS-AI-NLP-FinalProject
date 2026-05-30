from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class ReviewDataset(Dataset):
    """
    Dataset for Transformer-based review classification.

    It can return:
    - ids: input_ids for BERT/RoBERTa/etc.
    - mask: attention_mask
    - token_type_ids: only if the tokenizer/model returns it
    - meta_features: numeric features such as rating and thumbs_up_count
    - targets: class ids, if target_col exists in dataframe
    """

    def __init__(
        self,
        dataframe,
        tokenizer,
        max_seq_len: int,
        text_col: str = "text",
        target_col: Optional[str] = "label_num",
        meta_features: Optional[np.ndarray] = None,
        use_meta: bool = True,
    ):
        self.data = dataframe.reset_index(drop=True).copy()
        self.text_col = text_col
        self.target_col = target_col
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.use_meta = use_meta

        if self.text_col not in self.data.columns:
            raise ValueError(f"Column '{self.text_col}' was not found in dataframe")

        self.texts = self.data[self.text_col].fillna("").astype(str).tolist()

        if self.target_col is not None and self.target_col in self.data.columns:
            self.targets = self.data[self.target_col].astype(int).tolist()
        elif "rate" in self.data.columns:
            # Backward-compatible fallback for the old 5-class notebook.
            self.targets = self.data["rate"].astype(int).tolist()
        else:
            self.targets = None

        if self.use_meta:
            if meta_features is None:
                raise ValueError(
                    "meta_features must be provided when use_meta=True. "
                    "Scale rating/thumbs outside the Dataset to avoid validation/test leakage."
                )
            meta_features = np.asarray(meta_features, dtype=np.float32)
            if len(meta_features) != len(self.data):
                raise ValueError(
                    f"meta_features length ({len(meta_features)}) != dataframe length ({len(self.data)})"
                )
            self.meta_features = meta_features
        else:
            self.meta_features = None

    def __getitem__(self, index: int):
        text = " ".join(str(self.texts[index]).split())

        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )

        item = {
            "ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
            "mask": torch.tensor(inputs["attention_mask"], dtype=torch.long),
        }

        if "token_type_ids" in inputs:
            item["token_type_ids"] = torch.tensor(inputs["token_type_ids"], dtype=torch.long)

        if self.use_meta:
            item["meta_features"] = torch.tensor(self.meta_features[index], dtype=torch.float32)

        if self.targets is not None:
            item["targets"] = torch.tensor(self.targets[index], dtype=torch.long)

        return item

    def __len__(self) -> int:
        return len(self.texts)
