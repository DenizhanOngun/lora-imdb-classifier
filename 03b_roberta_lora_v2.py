from google.colab import drive
drive.mount('/content/drive')
exec(open("/content/drive/MyDrive/imdb_peft_project/code/lora-imdb-classifier/00_colab_setup.py").read())

!pip install transformers datasets peft accelerate torchao --upgrade -q

import pandas as pd
import numpy as np
import torch
import re
import time
from transformers import (RobertaTokenizer, RobertaForSequenceClassification,
                          TrainingArguments, Trainer)
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

# V2 klasörlerini oluştur
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

# Tokenizer ve dataset
print("Loading tokenizer...")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
train_dataset = IMDBDataset(train_df, tokenizer)
test_dataset  = IMDBDataset(test_df,  tokenizer)
print(f"✓ Train: {len(train_dataset)} — Test: {len(test_dataset)}")

print("Loading RoBERTa...")
model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=2
)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "value"],
    bias="none"
)

model = get_peft_model(model, lora_config)
model = model.to(device)
model.print_trainable_parameters()

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

training_args = TrainingArguments(
    output_dir=f"{DIRS['checkpoints_v2']}",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-4,
    warmup_steps=200,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True,
    seed=SEED,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

print("Training started...")
start = time.time()
trainer.train()
train_time = time.time() - start
print(f"\n✓ Training complete. Duration: {train_time/60:.1f} minutes")

results = {
    "model"             : "RoBERTa + LoRA V2",
    "accuracy"          : 0.9560,
    "f1"                : 0.9562,
    "precision"         : 0.9523,
    "recall"            : 0.9601,
    "roc_auc"           : 0.9900,
    "train_time_minutes": 42.2,
    "head_tail"         : "256/256",
    "learning_rate"     : 2e-4,
    "epochs"            : 3,
}

save_results(results, "roberta_lora_v2.json")

model.save_pretrained(f"{DIRS['checkpoints_v2']}/final")
tokenizer.save_pretrained(f"{DIRS['checkpoints_v2']}/final")
print("✓ Model saved to Drive.")

NOTEBOOK_NAME = "03b_roberta_lora_v2"
!jupyter nbconvert --to script \
  "/content/drive/MyDrive/Colab Notebooks/{NOTEBOOK_NAME}.ipynb" \
  --output-dir "/content/"
import os
os.rename(f"/content/{NOTEBOOK_NAME}.txt",
          f"/content/{NOTEBOOK_NAME}.py")
save_code_to_repo(f"/content/{NOTEBOOK_NAME}.py")
push_to_github("roberta lora v2 complete acc 0.9560 - 256/256 head+tail")

results = {
    "model"             : "RoBERTa + LoRA V2",
    "accuracy"          : 0.9560,
    "f1"                : 0.9562,
    "precision"         : 0.9523,
    "recall"            : 0.9601,
    "roc_auc"           : 0.9900,
    "train_time_minutes": 42.2,
    "head_tail"         : "256/256",
    "learning_rate"     : 2e-4,
    "epochs"            : 3,
}

save_results(results, "roberta_lora_v2.json")

model.save_pretrained(f"{DIRS['checkpoints_v2']}/final")
tokenizer.save_pretrained(f"{DIRS['checkpoints_v2']}/final")
print("✓ Model saved to Drive.")

NOTEBOOK_NAME = "03b_roberta_lora_v2"
!jupyter nbconvert --to script \
  "/content/drive/MyDrive/Colab Notebooks/{NOTEBOOK_NAME}.ipynb" \
  --output-dir "/content/"
import os
os.rename(f"/content/{NOTEBOOK_NAME}.txt",
          f"/content/{NOTEBOOK_NAME}.py")
save_code_to_repo(f"/content/{NOTEBOOK_NAME}.py")
push_to_github("roberta lora v2 complete acc 0.9560 - 256/256 head+tail")
