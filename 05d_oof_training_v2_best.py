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

DIRS["oof_v2_best"] = f"{DIRS['root']}/oof_predictions_v2_best"
os.makedirs(DIRS["oof_v2_best"], exist_ok=True)
print(f"✓ {DIRS['oof_v2_best']}")

train_df = pd.read_parquet(f"{DIRS['root']}/train_df_v2.parquet")
test_df  = pd.read_parquet(f"{DIRS['root']}/test_df_v2.parquet")
print(f"Train: {len(train_df)} | Test: {len(test_df)}")

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

print("Loading DeBERTa tokenizer...")
tokenizer     = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
train_dataset = IMDBDataset(train_df, tokenizer)
test_dataset  = IMDBDataset(test_df,  tokenizer)
print(f"✓ Train: {len(train_dataset)} — Test: {len(test_dataset)}")

def train_oof_deberta_best(df, n_folds=5):
    """
    DeBERTa V2 Best OOF training.
    Best config: r=32, lr=5e-5, lora_alpha=64
    Saves LoRA adapter + classifier + pooler separately.
    """
    skf       = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    labels    = df["label"].values
    oof_preds = np.zeros(len(df))

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, labels)):
        print(f"\n{'='*50}")
        print(f"FOLD {fold+1}/{n_folds} — deberta_v2_best")
        print(f"{'='*50}")

        fold_train = df.iloc[train_idx].reset_index(drop=True)
        fold_val   = df.iloc[val_idx].reset_index(drop=True)

        train_dataset = IMDBDataset(fold_train, tokenizer)
        val_dataset   = IMDBDataset(fold_val,   tokenizer)

        base_model = DebertaV2ForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-base",
            num_labels=2,
            torch_dtype=torch.float32
        )

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=32,
            lora_alpha=64,
            lora_dropout=0.1,
            target_modules=["query_proj", "value_proj"],
            bias="none"
        )
        model = get_peft_model(base_model, lora_config)
        model = model.to(torch.float32).to(device)

        training_args = TrainingArguments(
            output_dir=f"{DIRS['oof_v2_best']}/fold{fold}",
            num_train_epochs=2,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=32,
            learning_rate=5e-5,
            warmup_steps=500,
            weight_decay=0.01,
            gradient_accumulation_steps=2,
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

        preds_output = trainer.predict(val_dataset)
        logits       = preds_output.predictions
        probs        = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
        oof_preds[val_idx] = probs

        fold_acc = accuracy_score(labels[val_idx], (probs > 0.5).astype(int))
        fold_f1  = f1_score(labels[val_idx], (probs > 0.5).astype(int))
        print(f"Fold {fold+1} — Accuracy: {fold_acc:.4f} | F1: {fold_f1:.4f}")

        np.save(f"{DIRS['oof_v2_best']}/deberta_v2_best_fold{fold}.npy", oof_preds)
        print(f"✓ OOF saved: fold {fold}")

        del model, trainer, base_model
        torch.cuda.empty_cache()

    print(f"\n{'='*50}")
    print(f"OOF COMPLETE — deberta_v2_best")
    print(f"OOF Accuracy : {accuracy_score(labels, (oof_preds > 0.5).astype(int)):.4f}")
    print(f"OOF F1       : {f1_score(labels, (oof_preds > 0.5).astype(int)):.4f}")
    print(f"OOF ROC-AUC  : {roc_auc_score(labels, oof_preds):.4f}")
    print(f"{'='*50}")

    return oof_preds

print("✓ OOF function ready.")

print("DeBERTa V2 Best OOF training starting...")
print("5 folds x 2 epochs — approximately 2 hours on A100")

deberta_oof_best = train_oof_deberta_best(train_df, n_folds=5)

print("✓ DeBERTa V2 Best OOF complete.")

# Save OOF results
save_results({
    "deberta_v2_best_oof": {
        "accuracy": float(accuracy_score(train_df["label"].values,
                         (deberta_oof_best > 0.5).astype(int))),
        "f1"      : float(f1_score(train_df["label"].values,
                         (deberta_oof_best > 0.5).astype(int))),
        "roc_auc" : float(roc_auc_score(train_df["label"].values, deberta_oof_best)),
        "config"  : "r=32, lr=5e-5, lora_alpha=64",
        "folds"   : 5,
        "epochs"  : 2,
    }
}, "oof_deberta_v2_best_results.json")
print("✓ Results saved.")

NOTEBOOK_NAME = "05d_oof_training_v2_best"
!jupyter nbconvert --to script \
  "/content/drive/MyDrive/Colab Notebooks/{NOTEBOOK_NAME}.ipynb" \
  --output-dir "/content/"

import os
os.rename(f"/content/{NOTEBOOK_NAME}.txt",
          f"/content/{NOTEBOOK_NAME}.py")

save_code_to_repo(f"/content/{NOTEBOOK_NAME}.py")
push_to_github("deberta v2 best oof complete r32 lr5e5")
