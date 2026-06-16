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

DIRS["checkpoints2_best"] = f"{DIRS['root']}/checkpoints/deberta_lora_v2_best"
os.makedirs(DIRS["checkpoints2_best"], exist_ok=True)
print(f"✓ {DIRS['checkpoints2_best']}")

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

class SaveCallbackFixed(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        save_path = f"{DIRS['checkpoints2_best']}/epoch_{epoch}"
        os.makedirs(save_path, exist_ok=True)
        kwargs["model"].save_pretrained(save_path)
        extra_state = {
            "classifier.weight": kwargs["model"].base_model.model.classifier.weight.data.cpu(),
            "classifier.bias"  : kwargs["model"].base_model.model.classifier.bias.data.cpu(),
            "pooler.weight"    : kwargs["model"].base_model.model.pooler.dense.weight.data.cpu(),
            "pooler.bias"      : kwargs["model"].base_model.model.pooler.dense.bias.data.cpu(),
        }
        torch.save(extra_state, f"{save_path}/extra_weights.pt")
        print(f"✓ Epoch {epoch} saved — LoRA + classifier + pooler.")

print("Loading DeBERTa...")
model = DebertaV2ForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base",
    num_labels=2,
    torch_dtype=torch.float32
)

# Best config: r=32, lr=5e-5
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=32,
    lora_alpha=64,
    lora_dropout=0.1,
    target_modules=["query_proj", "value_proj"],
    bias="none"
)

model = get_peft_model(model, lora_config)
model = model.to(torch.float32).to(device)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir=f"{DIRS['checkpoints2_best']}",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
    warmup_steps=500,
    weight_decay=0.01,
    gradient_accumulation_steps=2,
    eval_strategy="epoch",
    save_strategy="no",
    fp16=False,
    bf16=False,
    seed=SEED,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[SaveCallbackFixed()]
)

print("Training started...")
start = time.time()
trainer.train()
train_time = time.time() - start
print(f"\n✓ Training complete. Duration: {train_time/60:.1f} minutes")
