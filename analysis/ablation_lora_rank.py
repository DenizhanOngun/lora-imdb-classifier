# =============================================================
# analysis/ablation_lora_rank.py
# =============================================================
# LoRA rank ablation study for RoBERTa.
# Trains RoBERTa+LoRA with r ∈ {8, 16, 32} and compares
# accuracy, F1, and parameter count.
#
# Key finding: r=16 is optimal — r=32 adds no meaningful gain
# (+0.0001 accuracy) while using 50% more parameters.
#
# Output:
#   data/results/ablation_lora_rank.json
#
# Run:
#   python analysis/ablation_lora_rank.py
# =============================================================

import os
import time
import numpy as np
import pandas as pd
import torch
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType

from config import DATA_DIR, RESULTS_DIR, SEED, ROBERTA_CONFIG
from utils import IMDBDataset, compute_metrics, save_results

np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
print("Loading preprocessed data...")
train_df = pd.read_parquet(os.path.join(DATA_DIR, "train_df_v2.parquet"))
test_df  = pd.read_parquet(os.path.join(DATA_DIR, "test_df_v2.parquet"))

cfg = ROBERTA_CONFIG
tokenizer     = RobertaTokenizer.from_pretrained(cfg["base_model"])
train_dataset = IMDBDataset(train_df, tokenizer)
test_dataset  = IMDBDataset(test_df,  tokenizer)
print(f"Train: {len(train_dataset)} | Test: {len(test_dataset)}")


# ------------------------------------------------------------------
# 2. Training function
# ------------------------------------------------------------------
def train_with_rank(r: int) -> dict:
    """Train RoBERTa+LoRA with a specific rank and return metrics."""
    lora_alpha = r * 2
    print(f"\n{'='*50}")
    print(f"  r={r}, alpha={lora_alpha}")
    print(f"{'='*50}")

    base_model = RobertaForSequenceClassification.from_pretrained(
        cfg["base_model"], num_labels=2
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model = model.to(device)
    model.print_trainable_parameters()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())

    output_dir = os.path.join(RESULTS_DIR, f"roberta_r{r}")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"] * 2,
        learning_rate=cfg["learning_rate"],
        warmup_steps=cfg["warmup_steps"],
        eval_strategy="epoch",
        save_strategy="no",
        fp16=cfg["fp16"],
        seed=SEED,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    start = time.time()
    trainer.train()
    train_time = time.time() - start

    metrics = trainer.evaluate()
    results = {
        "r"                  : r,
        "lora_alpha"         : lora_alpha,
        "trainable_params"   : trainable,
        "total_params"       : total,
        "pct_trainable"      : round(trainable / total * 100, 2),
        "accuracy"           : round(metrics["eval_accuracy"], 4),
        "f1"                 : round(metrics["eval_f1"], 4),
        "precision"          : round(metrics["eval_precision"], 4),
        "recall"             : round(metrics["eval_recall"], 4),
        "roc_auc"            : round(metrics["eval_roc_auc"], 4),
        "train_time_minutes" : round(train_time / 60, 1),
    }

    print(f"\n  r={r} result: Accuracy={results['accuracy']} | F1={results['f1']}")
    del model, trainer, base_model
    torch.cuda.empty_cache()
    return results


# ------------------------------------------------------------------
# 3. Run ablation
# ------------------------------------------------------------------
ablation_results = []
for r in [8, 16, 32]:
    ablation_results.append(train_with_rank(r))
    save_results({"ablation_lora_rank": ablation_results},
                 "ablation_lora_rank.json")


# ------------------------------------------------------------------
# 4. Summary table
# ------------------------------------------------------------------
df = pd.DataFrame(ablation_results)
print(f"\n{'='*65}")
print("LORA RANK ABLATION — ROBERTA")
print(f"{'='*65}")
print(df[["r", "trainable_params", "pct_trainable", "accuracy", "f1",
          "train_time_minutes"]].to_string(index=False))

print("\nLoRA rank ablation complete.")
