# =============================================================
# 04_deberta_lora.py
# =============================================================
# Fine-tunes DeBERTa-v3-base with LoRA (r=32) on the full IMDB
# training set and evaluates on the test set.
#
# Best config: r=32, lr=5e-5, fp32, 3 epochs → 95.92% accuracy
#
# Note: DeBERTa-v3 requires fp32 precision due to gradient
# instability under fp16 (ELECTRA-style pretraining). Pooler
# and classifier weights are saved separately from the LoRA
# adapter to ensure correct checkpoint restoration.
#
# Output:
#   data/checkpoints/deberta_lora/epoch_1/
#   data/checkpoints/deberta_lora/epoch_2/
#   data/checkpoints/deberta_lora/epoch_3/   ← used by downstream scripts
#   data/checkpoints/deberta_lora/final/
#   data/results/deberta_lora.json
#
# Run:
#   python src/04_deberta_lora.py
# =============================================================

import os
import time
import numpy as np
import pandas as pd
import torch
from transformers import (
    DebertaV2Tokenizer,
    DebertaV2ForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType

from config import DATA_DIR, CHECKPOINT_DIR, SEED, DEBERTA_CONFIG
from utils import IMDBDataset, compute_metrics, save_results

np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ------------------------------------------------------------------
# Callback: save LoRA adapter + classifier + pooler after each epoch
# ------------------------------------------------------------------
class SaveCallbackFixed(TrainerCallback):
    """
    Saves LoRA adapter and extra weights (classifier + pooler)
    after each epoch. Standard PeftModel.save_pretrained() with
    task_type=SEQ_CLS does not persist the classifier and pooler,
    causing near-random predictions on reload. This callback saves
    them separately in extra_weights.pt.
    """

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        save_path = os.path.join(args.output_dir, f"epoch_{epoch}")
        os.makedirs(save_path, exist_ok=True)

        kwargs["model"].save_pretrained(save_path)

        extra_state = {
            "classifier.weight": kwargs["model"].base_model.model.classifier.weight.data.cpu(),
            "classifier.bias"  : kwargs["model"].base_model.model.classifier.bias.data.cpu(),
            "pooler.weight"    : kwargs["model"].base_model.model.pooler.dense.weight.data.cpu(),
            "pooler.bias"      : kwargs["model"].base_model.model.pooler.dense.bias.data.cpu(),
        }
        torch.save(extra_state, os.path.join(save_path, "extra_weights.pt"))
        print(f"Epoch {epoch} saved — LoRA + classifier + pooler.")


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
print("Loading DeBERTa tokenizer...")
tokenizer     = DebertaV2Tokenizer.from_pretrained(DEBERTA_CONFIG["base_model"])
train_dataset = IMDBDataset(train_df, tokenizer)
test_dataset  = IMDBDataset(test_df,  tokenizer)
print(f"Train dataset: {len(train_dataset)} | Test dataset: {len(test_dataset)}")


# ------------------------------------------------------------------
# 3. Model + LoRA
# ------------------------------------------------------------------
cfg = DEBERTA_CONFIG
print(f"\nLoading DeBERTa-v3 with LoRA (r={cfg['lora_r']})...")

base_model = DebertaV2ForSequenceClassification.from_pretrained(
    cfg["base_model"],
    num_labels=2,
    torch_dtype=torch.float32,
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
model = model.to(torch.float32).to(device)
model.print_trainable_parameters()


# ------------------------------------------------------------------
# 4. Training
# ------------------------------------------------------------------
output_dir = os.path.join(CHECKPOINT_DIR, "deberta_lora")
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
    save_strategy="no",
    fp16=cfg["fp16"],
    bf16=False,
    seed=SEED,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[SaveCallbackFixed()],
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

# Save final checkpoint
final_dir = os.path.join(output_dir, "final")
os.makedirs(final_dir, exist_ok=True)
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
torch.save({
    "classifier.weight": model.base_model.model.classifier.weight.data.cpu(),
    "classifier.bias"  : model.base_model.model.classifier.bias.data.cpu(),
    "pooler.weight"    : model.base_model.model.pooler.dense.weight.data.cpu(),
    "pooler.bias"      : model.base_model.model.pooler.dense.bias.data.cpu(),
}, os.path.join(final_dir, "extra_weights.pt"))
print(f"\nFinal model saved to {final_dir}")

save_results({
    "model"              : "DeBERTa-v3 + LoRA",
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
}, "deberta_lora.json")

print("\nDeBERTa + LoRA training complete.")
