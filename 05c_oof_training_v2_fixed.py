from google.colab import drive
drive.mount('/content/drive')
exec(open("/content/drive/MyDrive/imdb_peft_project/code/lora-imdb-classifier/00_colab_setup.py").read())

!pip install transformers peft accelerate torchao --upgrade -q

import pandas as pd
import numpy as np
import torch
import time
import os
from transformers import (DebertaV2Tokenizer, DebertaV2ForSequenceClassification,
                          TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# V2 fixed OOF folder
DIRS["oof_v2_fixed"] = f"{DIRS['root']}/oof_predictions_v2_fixed"
os.makedirs(DIRS["oof_v2_fixed"], exist_ok=True)
print(f"✓ {DIRS['oof_v2_fixed']}")

# Load data
train_df = pd.read_parquet(f"{DIRS['root']}/train_df_v2.parquet")
print(f"Train: {len(train_df)} samples")

def head_tail_truncate_v2(text, tokenizer, max_len=512, head_len=256):
    """V2: Equal split — first 256 + last 256 tokens."""
    tail_len = max_len - head_len
    tokens = tokenizer(text, add_special_tokens=False,
                       truncation=False, return_tensors=None)
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
        return_tensors=None
    )

class IMDBDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512, head_len=256):
        self.df        = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.head_len  = head_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text  = self.df.loc[idx, "text_clean"]
        label = self.df.loc[idx, "label"]
        encoding = head_tail_truncate_v2(
            text, self.tokenizer, self.max_len, self.head_len
        )
        return {
            "input_ids":      torch.tensor(encoding["input_ids"],      dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "labels":         torch.tensor(label,                      dtype=torch.long),
        }

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    return {
        "accuracy" : accuracy_score(labels, preds),
        "f1"       : f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall"   : recall_score(labels, preds),
        "roc_auc"  : roc_auc_score(labels, probs)
    }

print("✓ Helper functions ready.")

print("Loading DeBERTa tokenizer...")
tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
print("✓ Tokenizer ready.")

def train_oof_deberta_fixed(df, n_folds=5, learning_rate=3e-5,
                            batch_size=8, epochs=2,
                            warmup_steps=500, weight_decay=0.01,
                            grad_accum=2):
    """
    DeBERTa V2 Fixed OOF training.
    Saves LoRA adapter and classifier separately to avoid
    the modules_to_save namespace issue.
    """
    skf       = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    labels    = df["label"].values
    oof_preds = np.zeros(len(df))

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, labels)):
        print(f"\n{'='*50}")
        print(f"FOLD {fold+1}/{n_folds} — deberta_v2_fixed")
        print(f"{'='*50}")

        fold_train = df.iloc[train_idx].reset_index(drop=True)
        fold_val   = df.iloc[val_idx].reset_index(drop=True)

        train_dataset = IMDBDataset(fold_train, tokenizer)
        val_dataset   = IMDBDataset(fold_val,   tokenizer)

        # Load fresh base model
        base_model = DebertaV2ForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-base",
            num_labels=2,
            torch_dtype=torch.float32
        )

        # Add LoRA without modules_to_save
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=16, lora_alpha=32, lora_dropout=0.1,
            target_modules=["query_proj", "value_proj"],
            bias="none"
            # No modules_to_save — this was causing the checkpoint issue
        )
        model = get_peft_model(base_model, lora_config)
        model = model.to(torch.float32).to(device)

        training_args = TrainingArguments(
            output_dir=f"{DIRS['oof_v2_fixed']}/deberta_v2_fixed_fold{fold}",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=32,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            gradient_accumulation_steps=grad_accum,
            eval_strategy="epoch",
            save_strategy="no",
            fp16=False,
            seed=SEED,
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        # Generate OOF predictions
        preds_output = trainer.predict(val_dataset)
        logits       = preds_output.predictions
        probs        = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
        oof_preds[val_idx] = probs

        fold_acc = accuracy_score(labels[val_idx], (probs > 0.5).astype(int))
        fold_f1  = f1_score(labels[val_idx], (probs > 0.5).astype(int))
        print(f"Fold {fold+1} — Accuracy: {fold_acc:.4f} | F1: {fold_f1:.4f}")

        # Save OOF predictions
        np.save(f"{DIRS['oof_v2_fixed']}/deberta_v2_fixed_fold{fold}.npy", oof_preds)
        print(f"✓ OOF saved: fold {fold}")

        del model, trainer, base_model
        torch.cuda.empty_cache()

    print(f"\n{'='*50}")
    print(f"OOF COMPLETE — deberta_v2_fixed")
    print(f"OOF Accuracy : {accuracy_score(labels, (oof_preds > 0.5).astype(int)):.4f}")
    print(f"OOF F1       : {f1_score(labels, (oof_preds > 0.5).astype(int)):.4f}")
    print(f"OOF ROC-AUC  : {roc_auc_score(labels, oof_preds):.4f}")
    print(f"{'='*50}")

    return oof_preds

print("✓ OOF function ready.")

print("DeBERTa V2 Fixed OOF training starting...")
print("5 folds x 2 epochs — approximately 1.5 hours on A100")

deberta_oof_fixed = train_oof_deberta_fixed(
    df           = train_df,
    n_folds      = 5,
    learning_rate= 3e-5,
    batch_size   = 8,
    epochs       = 2,
    warmup_steps = 500,
    weight_decay = 0.01,
    grad_accum   = 2
)

print("✓ DeBERTa V2 Fixed OOF complete.")
