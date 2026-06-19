# =============================================================
# 05_oof_training.py
# =============================================================
# Generates 5-fold out-of-fold (OOF) probability predictions
# for RoBERTa and DeBERTa using stratified cross-validation.
# These predictions are used as meta-features for the stacking
# ensemble in script 06.
#
# Each fold trains for 2 epochs (reduced from 3 to limit cost
# while preserving signal quality).
#
# Output:
#   data/oof_predictions/roberta_oof.npy      (25000,)
#   data/oof_predictions/deberta_oof.npy      (25000,)
#   data/oof_predictions/roberta_test_probs.npy
#   data/oof_predictions/deberta_test_probs.npy
#   data/results/oof_training.json
#
# Run:
#   python src/05_oof_training.py
# =============================================================

import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import (
    RobertaTokenizer, RobertaForSequenceClassification,
    DebertaV2Tokenizer, DebertaV2ForSequenceClassification,
    TrainingArguments, Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from config import (
    DATA_DIR, OOF_DIR, CHECKPOINT_DIR, SEED, N_FOLDS,
    ROBERTA_CONFIG, DEBERTA_CONFIG,
)
from utils import IMDBDataset, compute_metrics, load_deberta_model, save_results

np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
os.makedirs(OOF_DIR, exist_ok=True)


# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
print("Loading preprocessed data...")
train_df = pd.read_parquet(os.path.join(DATA_DIR, "train_df_v2.parquet"))
test_df  = pd.read_parquet(os.path.join(DATA_DIR, "test_df_v2.parquet"))
y_train  = train_df["label"].values
y_test   = test_df["label"].values
print(f"Train: {len(train_df)} | Test: {len(test_df)}")


# ------------------------------------------------------------------
# Helper: run OOF for one model family
# ------------------------------------------------------------------

def run_oof(model_name: str, tokenizer, build_model_fn,
            train_df, y_train, test_df, cfg: dict) -> tuple:
    """
    5-fold OOF training for a single model family.

    Returns (oof_preds, test_probs).
    oof_preds  : (25000,) probability array for training set
    test_probs : (25000,) probability array for test set, averaged
                 across folds
    """
    skf       = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(train_df))
    test_probs_list = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y_train)):
        print(f"\n{'='*50}")
        print(f"  {model_name} — Fold {fold+1}/{N_FOLDS}")
        print(f"{'='*50}")

        fold_train = train_df.iloc[train_idx].reset_index(drop=True)
        fold_val   = train_df.iloc[val_idx].reset_index(drop=True)

        train_dataset = IMDBDataset(fold_train, tokenizer)
        val_dataset   = IMDBDataset(fold_val,   tokenizer)
        test_dataset  = IMDBDataset(test_df,    tokenizer)

        model = build_model_fn().to(device)

        fold_dir = os.path.join(OOF_DIR, f"{model_name}_fold{fold}")
        training_args = TrainingArguments(
            output_dir=fold_dir,
            num_train_epochs=cfg["oof_epochs"],
            per_device_train_batch_size=cfg["batch_size"],
            per_device_eval_batch_size=cfg["batch_size"] * 4,
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
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )
        trainer.train()

        # OOF predictions on validation fold
        val_output = trainer.predict(val_dataset)
        val_probs  = torch.softmax(torch.tensor(val_output.predictions), dim=-1)[:, 1].numpy()
        oof_preds[val_idx] = val_probs

        fold_acc = accuracy_score(y_train[val_idx], (val_probs > 0.5).astype(int))
        fold_f1  = f1_score(y_train[val_idx], (val_probs > 0.5).astype(int))
        print(f"  Fold {fold+1} — Accuracy: {fold_acc:.4f} | F1: {fold_f1:.4f}")

        # Test predictions for this fold
        test_output = trainer.predict(test_dataset)
        test_probs  = torch.softmax(torch.tensor(test_output.predictions), dim=-1)[:, 1].numpy()
        test_probs_list.append(test_probs)

        del model, trainer
        torch.cuda.empty_cache()

    oof_acc = accuracy_score(y_train, (oof_preds > 0.5).astype(int))
    oof_f1  = f1_score(y_train, (oof_preds > 0.5).astype(int))
    oof_auc = roc_auc_score(y_train, oof_preds)
    print(f"\n{model_name} OOF — Accuracy: {oof_acc:.4f} | F1: {oof_f1:.4f} | AUC: {oof_auc:.4f}")

    avg_test_probs = np.mean(test_probs_list, axis=0)
    return oof_preds, avg_test_probs


# ------------------------------------------------------------------
# 2. RoBERTa OOF
# ------------------------------------------------------------------
cfg_r = ROBERTA_CONFIG
roberta_tokenizer = RobertaTokenizer.from_pretrained(cfg_r["base_model"])

def build_roberta():
    base = RobertaForSequenceClassification.from_pretrained(cfg_r["base_model"], num_labels=2)
    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=cfg_r["lora_r"], lora_alpha=cfg_r["lora_alpha"],
        lora_dropout=cfg_r["lora_dropout"],
        target_modules=cfg_r["target_modules"], bias="none",
    )
    return get_peft_model(base, lora)

print("\n" + "="*60)
print("ROBERTA OOF TRAINING")
print("="*60)
roberta_oof, roberta_test_probs = run_oof(
    "roberta", roberta_tokenizer, build_roberta,
    train_df, y_train, test_df, cfg_r,
)
np.save(os.path.join(OOF_DIR, "roberta_oof.npy"),        roberta_oof)
np.save(os.path.join(OOF_DIR, "roberta_test_probs.npy"), roberta_test_probs)
print("RoBERTa OOF predictions saved.")


# ------------------------------------------------------------------
# 3. DeBERTa OOF
# ------------------------------------------------------------------
cfg_d = DEBERTA_CONFIG
deberta_tokenizer = DebertaV2Tokenizer.from_pretrained(cfg_d["base_model"])

def build_deberta():
    base = DebertaV2ForSequenceClassification.from_pretrained(
        cfg_d["base_model"], num_labels=2, torch_dtype=torch.float32,
    )
    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=cfg_d["lora_r"], lora_alpha=cfg_d["lora_alpha"],
        lora_dropout=cfg_d["lora_dropout"],
        target_modules=cfg_d["target_modules"], bias="none",
    )
    return get_peft_model(base, lora).to(torch.float32)

print("\n" + "="*60)
print("DEBERTA OOF TRAINING")
print("="*60)
deberta_oof, deberta_test_probs = run_oof(
    "deberta", deberta_tokenizer, build_deberta,
    train_df, y_train, test_df, cfg_d,
)
np.save(os.path.join(OOF_DIR, "deberta_oof.npy"),        deberta_oof)
np.save(os.path.join(OOF_DIR, "deberta_test_probs.npy"), deberta_test_probs)
print("DeBERTa OOF predictions saved.")


# ------------------------------------------------------------------
# 4. Save summary
# ------------------------------------------------------------------
save_results({
    "roberta_oof": {
        "accuracy": round(float(accuracy_score(y_train, (roberta_oof > 0.5).astype(int))), 4),
        "f1"      : round(float(f1_score(y_train, (roberta_oof > 0.5).astype(int))), 4),
        "roc_auc" : round(float(roc_auc_score(y_train, roberta_oof)), 4),
    },
    "deberta_oof": {
        "accuracy": round(float(accuracy_score(y_train, (deberta_oof > 0.5).astype(int))), 4),
        "f1"      : round(float(f1_score(y_train, (deberta_oof > 0.5).astype(int))), 4),
        "roc_auc" : round(float(roc_auc_score(y_train, deberta_oof)), 4),
    },
}, "oof_training.json")

print("\nOOF training complete.")
