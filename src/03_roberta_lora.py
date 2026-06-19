# =============================================================
# 03_roberta_lora.py
# =============================================================
# Fine-tunes RoBERTa-base with LoRA (r=16) on the full IMDB
# training set and evaluates on the test set.
#
# Best config: r=16, lr=2e-4, fp16, 3 epochs → 95.60% accuracy
#
# Output:
#   data/checkpoints/roberta_lora/final/   (adapter weights)
#   data/results/roberta_lora.json
#
# Run:
#   python src/03_roberta_lora.py
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

from config import DATA_DIR, CHECKPOINT_DIR, SEED, ROBERTA_CONFIG
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
print(f"Train: {len(train_df)} | Test: {len(test_df)}")


# ------------------------------------------------------------------
# 2. Tokenizer and datasets
# ------------------------------------------------------------------
print("Loading RoBERTa tokenizer...")
tokenizer     = RobertaTokenizer.from_pretrained(ROBERTA_CONFIG["base_model"])
train_dataset = IMDBDataset(train_df, tokenizer)
test_dataset  = IMDBDataset(test_df,  tokenizer)
print(f"Train dataset: {len(train_dataset)} | Test dataset: {len(test_dataset)}")


# ------------------------------------------------------------------
# 3. Model + LoRA
# ------------------------------------------------------------------
cfg = ROBERTA_CONFIG
print(f"\nLoading RoBERTa with LoRA (r={cfg['lora_r']})...")

base_model = RobertaForSequenceClassification.from_pretrained(
    cfg["base_model"], num_labels=2
)
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=cfg["lora_r"],
    lora_alpha=cfg["lora_alpha"],
    lora_dropout=cfg["lora_dropout"],
    target_modules=cfg["target_modules"],
    bias="none",
)
model = get_peft_model(base_model, lora_config)
model = model.to(device)
model.print_trainable_parameters()


# ------------------------------------------------------------------
# 4. Training
# ------------------------------------------------------------------
output_dir = os.path.join(CHECKPOINT_DIR, "roberta_lora")
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=cfg["epochs"],
    per_device_train_batch_size=cfg["batch_size"],
    per_device_eval_batch_size=cfg["batch_size"] * 2,
    learning_rate=cfg["learning_rate"],
    warmup_steps=cfg["warmup_steps"],
    weight_decay=cfg["weight_decay"],
    gradient_accumulation_steps=cfg["grad_accum_steps"],
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
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

print("\nTraining started...")
start = time.time()
trainer.train()
train_time = time.time() - start
print(f"\nTraining complete. Duration: {train_time/60:.1f} minutes")


# ------------------------------------------------------------------
# 5. Evaluate and save
# ------------------------------------------------------------------
metrics = trainer.evaluate()
print(f"\nTest Accuracy : {metrics['eval_accuracy']:.4f}")
print(f"Test F1       : {metrics['eval_f1']:.4f}")
print(f"Test ROC-AUC  : {metrics['eval_roc_auc']:.4f}")

final_dir = os.path.join(output_dir, "final")
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
print(f"\nModel saved to {final_dir}")

save_results({
    "model"              : "RoBERTa + LoRA",
    "lora_r"             : cfg["lora_r"],
    "lora_alpha"         : cfg["lora_alpha"],
    "learning_rate"      : cfg["learning_rate"],
    "epochs"             : cfg["epochs"],
    "fp16"               : cfg["fp16"],
    "head_tail"          : "256/256",
    "accuracy"           : round(metrics["eval_accuracy"], 4),
    "f1"                 : round(metrics["eval_f1"], 4),
    "precision"          : round(metrics["eval_precision"], 4),
    "recall"             : round(metrics["eval_recall"], 4),
    "roc_auc"            : round(metrics["eval_roc_auc"], 4),
    "train_time_minutes" : round(train_time / 60, 1),
}, "roberta_lora.json")

print("\nRoBERTa + LoRA training complete.")
