from google.colab import drive
drive.mount('/content/drive')
exec(open("/content/drive/MyDrive/imdb_peft_project/code/lora-imdb-classifier/00_colab_setup.py").read())

!pip install transformers datasets peft accelerate torchao --upgrade -q

import pandas as pd
import numpy as np
import torch
import re
import time
from transformers import (DebertaV2Tokenizer, DebertaV2ForSequenceClassification,
                          TrainingArguments, Trainer, TrainerCallback)
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

import os

DIRS["checkpoints_v2"]  = f"{DIRS['root']}/checkpoints/roberta_lora_v2"
DIRS["checkpoints2_v2"] = f"{DIRS['root']}/checkpoints/deberta_lora_v2"
DIRS["oof_v2"]          = f"{DIRS['root']}/oof_predictions_v2"

for path in [DIRS["checkpoints_v2"], DIRS["checkpoints2_v2"], DIRS["oof_v2"]]:
    os.makedirs(path, exist_ok=True)
    print(f"✓ {path}")

train_df = pd.read_parquet(f"{DIRS['root']}/train_df_v2.parquet")
test_df  = pd.read_parquet(f"{DIRS['root']}/test_df_v2.parquet")

print(f"Train : {len(train_df)} samples")
print(f"Test  : {len(test_df)} samples")

def head_tail_truncate_v2(text, tokenizer, max_len=512, head_len=256):
    """V2: Equal split — first 256 + last 256 tokens."""
    tail_len = max_len - head_len

    tokens = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_tensors=None
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
        return_tensors=None
    )

class IMDBDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512, head_len=256,
                 text_col="text_clean"):
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
        encoding = head_tail_truncate_v2(
            text, self.tokenizer,
            max_len=self.max_len,
            head_len=self.head_len
        )
        return {
            "input_ids":      torch.tensor(encoding["input_ids"],      dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "labels":         torch.tensor(label,                      dtype=torch.long),
        }

print("Loading DeBERTa tokenizer...")
tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
train_dataset = IMDBDataset(train_df, tokenizer)
test_dataset  = IMDBDataset(test_df,  tokenizer)
print(f"✓ Train: {len(train_dataset)} — Test: {len(test_dataset)}")

def load_deberta_model():
    """Load DeBERTa-v3 with LoRA — call fresh each time."""
    base_model = DebertaV2ForSequenceClassification.from_pretrained(
        "microsoft/deberta-v3-base",
        num_labels=2,
        torch_dtype=torch.float32
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query_proj", "value_proj"],
        bias="none"
    )

    model = get_peft_model(base_model, lora_config)
    model = model.to(torch.float32)
    return model

print("✓ Model loader function ready.")
print("  Call load_deberta_model() when GPU is available.")

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

class SaveCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        path  = f"{DIRS['checkpoints2_v2']}/epoch_{epoch}"
        trainer.model.save_pretrained(path)
        print(f"✓ Epoch {epoch} saved to Drive.")

training_args = TrainingArguments(
    output_dir=f"{DIRS['checkpoints2_v2']}",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=3e-5,              # V1: 2e-5 → V2: 3e-5
    warmup_steps=1000,               # V1: 500  → V2: 1000
    weight_decay=0.01,               # V1: yok  → V2: eklendi
    gradient_accumulation_steps=2,   # V1: yok  → V2: efektif batch 16
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=False,
    bf16=False,
    seed=SEED,
    report_to="none"
)

print("✓ Training arguments ready.")
print("  V2 changes vs V1:")
print("  lr           : 2e-5 → 3e-5")
print("  warmup_steps : 500  → 1000")
print("  weight_decay : 0    → 0.01")
print("  grad_accum   : 1    → 2 (effective batch: 16)")

# !! BU HÜCREYİ ROBERTA EĞİTİMİ BİTİNCE ÇALIŞTIR !!
# Önce çalışma zamanını T4 GPU'ya değiştir

print("Loading DeBERTa model...")
model = load_deberta_model()
model = model.to(device)
model.print_trainable_parameters()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[SaveCallback()]
)

print("\nTraining started...")
start = time.time()
trainer.train()
train_time = time.time() - start
print(f"\n✓ Training complete. Duration: {train_time/60:.1f} minutes")
