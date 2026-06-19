# =============================================================
# utils.py — Shared utilities for lora-imdb-classifier
# =============================================================
# Functions used by multiple scripts are defined here once.
# Import what you need:
#   from utils import IMDBDataset, compute_metrics, load_deberta_model
# =============================================================

import os
import json
import numpy as np
import torch
import safetensors.torch as st
from torch.utils.data import Dataset
from transformers import (
    DebertaV2ForSequenceClassification,
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)

from config import MAX_LEN, HEAD_LEN, RESULTS_DIR


# ------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------

def head_tail_truncate(text: str, tokenizer, max_len: int = MAX_LEN,
                       head_len: int = HEAD_LEN) -> dict:
    """
    Head+Tail truncation strategy.

    Retains the first `head_len` and last `max_len - head_len` tokens
    so that both the opening statement and closing verdict of a review
    are preserved. For reviews within the token limit, no truncation
    is applied.

    Args:
        text      : Raw (cleaned) review text.
        tokenizer : HuggingFace tokenizer.
        max_len   : Maximum sequence length including special tokens (512).
        head_len  : Number of head tokens to keep (256).

    Returns:
        Tokenizer output dict with input_ids and attention_mask.
    """
    tail_len = max_len - head_len

    tokens = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_tensors=None,
    )
    input_ids      = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    if len(input_ids) > max_len - 2:
        input_ids      = input_ids[:head_len] + input_ids[-tail_len:]
        attention_mask = attention_mask[:head_len] + attention_mask[-tail_len:]

    return tokenizer(
        tokenizer.decode(input_ids),
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors=None,
    )


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class IMDBDataset(Dataset):
    """
    PyTorch Dataset for IMDB sentiment classification.

    Applies head+tail truncation to each review at __getitem__ time.

    Args:
        df        : DataFrame with 'text_clean' and 'label' columns.
        tokenizer : HuggingFace tokenizer.
        max_len   : Maximum sequence length (default 512).
        head_len  : Head tokens for truncation (default 256).
        text_col  : Column to use; 'text_clean' for transformer models,
                    'text_clean_lower' for TF-IDF.
    """

    def __init__(self, df, tokenizer, max_len: int = MAX_LEN,
                 head_len: int = HEAD_LEN, text_col: str = "text_clean"):
        self.df        = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.head_len  = head_len
        self.text_col  = text_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text  = self.df.loc[idx, self.text_col]
        label = self.df.loc[idx, "label"]

        encoding = head_tail_truncate(
            text, self.tokenizer,
            max_len=self.max_len,
            head_len=self.head_len,
        )

        return {
            "input_ids"     : torch.tensor(encoding["input_ids"],      dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "labels"        : torch.tensor(label,                      dtype=torch.long),
        }


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def compute_metrics(eval_pred) -> dict:
    """
    HuggingFace Trainer-compatible metric function.

    Returns accuracy, F1, precision, recall, and ROC-AUC.
    """
    logits, labels = eval_pred
    preds  = np.argmax(logits, axis=-1)
    probs  = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()

    return {
        "accuracy" : accuracy_score(labels, preds),
        "f1"       : f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall"   : recall_score(labels, preds),
        "roc_auc"  : roc_auc_score(labels, probs),
    }


# ------------------------------------------------------------------
# DeBERTa checkpoint loading
# ------------------------------------------------------------------

def load_deberta_model(checkpoint_dir: str, lora_r: int = 32,
                       lora_alpha: int = 64, device: str = "cpu"):
    """
    Load a DeBERTa-v3 + LoRA model from a checkpoint saved with
    the custom SaveCallbackFixed (separate extra_weights.pt).

    Background: When using PEFT with task_type=SEQ_CLS, the pooler
    and classifier weights are not saved with the LoRA adapter.
    Loading only the adapter causes near-random predictions.
    This function re-maps the saved keys and loads everything correctly.

    Args:
        checkpoint_dir : Path to the epoch/final checkpoint folder
                         (must contain adapter_model.safetensors
                         and extra_weights.pt).
        lora_r         : LoRA rank used during training (default 32).
        lora_alpha     : LoRA alpha used during training (default 64).
        device         : 'cuda' or 'cpu'.

    Returns:
        Loaded PeftModel in eval mode.
    """
    from transformers import DebertaV2ForSequenceClassification

    base_model = DebertaV2ForSequenceClassification.from_pretrained(
        "microsoft/deberta-v3-base",
        num_labels=2,
        torch_dtype=torch.float32,
        ignore_mismatched_sizes=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=["query_proj", "value_proj"],
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)

    # Load adapter weights — remap key names saved by older PEFT versions
    weights = st.load_file(
        os.path.join(checkpoint_dir, "adapter_model.safetensors")
    )
    extra = torch.load(
        os.path.join(checkpoint_dir, "extra_weights.pt"),
        map_location="cpu",
    )

    new_weights = {}
    for k, v in weights.items():
        if "lora_A.weight" in k:
            new_weights[k.replace("lora_A.weight", "lora_A.default.weight")] = v
        elif "lora_B.weight" in k:
            new_weights[k.replace("lora_B.weight", "lora_B.default.weight")] = v

    # Classifier and pooler must be assigned to both namespaces
    new_weights["base_model.model.classifier.modules_to_save.default.weight"] = extra["classifier.weight"]
    new_weights["base_model.model.classifier.modules_to_save.default.bias"]   = extra["classifier.bias"]
    new_weights["base_model.model.classifier.original_module.weight"]          = extra["classifier.weight"]
    new_weights["base_model.model.classifier.original_module.bias"]            = extra["classifier.bias"]
    new_weights["base_model.model.pooler.dense.weight"]                        = extra["pooler.weight"]
    new_weights["base_model.model.pooler.dense.bias"]                          = extra["pooler.bias"]

    model.load_state_dict(new_weights, strict=False)
    model = model.to(torch.float32).to(device)
    model.eval()
    return model


# ------------------------------------------------------------------
# Results
# ------------------------------------------------------------------

def save_results(metrics: dict, filename: str = "results.json"):
    """Save evaluation metrics to RESULTS_DIR as JSON."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Results saved: {path}")
