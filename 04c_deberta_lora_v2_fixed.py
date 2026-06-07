from google.colab import drive
drive.mount('/content/drive')
exec(open("/content/drive/MyDrive/imdb_peft_project/code/lora-imdb-classifier/00_colab_setup.py").read())

!pip install transformers peft accelerate torchao --upgrade -q

import pandas as pd
import numpy as np
import torch
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

DIRS["checkpoints2_v2_fixed"] = f"{DIRS['root']}/checkpoints/deberta_lora_v2_fixed"
os.makedirs(DIRS["checkpoints2_v2_fixed"], exist_ok=True)
print(f"✓ {DIRS['checkpoints2_v2_fixed']}")

train_df = pd.read_parquet(f"{DIRS['root']}/train_df_v2.parquet")
test_df  = pd.read_parquet(f"{DIRS['root']}/test_df_v2.parquet")

print(f"Train : {len(train_df)} samples")
print(f"Test  : {len(test_df)} samples")

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

print("Loading DeBERTa tokenizer...")
tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
train_dataset = IMDBDataset(train_df, tokenizer)
test_dataset  = IMDBDataset(test_df,  tokenizer)
print(f"✓ Train: {len(train_dataset)} — Test: {len(test_dataset)}")

print("Loading DeBERTa...")
model = DebertaV2ForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base",
    num_labels=2,
    torch_dtype=torch.float32
)

# Key fix: modules_to_save is NOT available — this resolves the checkpoint issue.
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query_proj", "value_proj"],
    bias="none"
    # modules_to_save removed!
)

model = get_peft_model(model, lora_config)
model = model.to(torch.float32)
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

class SaveCallbackFixed(TrainerCallback):
    """
    Saves both LoRA adapter and classifier weights separately.
    This avoids the modules_to_save namespace issue.
    """
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        save_path = f"{DIRS['checkpoints2_v2_fixed']}/epoch_{epoch}"
        os.makedirs(save_path, exist_ok=True)

        # Save LoRA adapter
        kwargs["model"].save_pretrained(save_path)

        # Save classifier separately
        classifier_state = {
            "classifier.weight": kwargs["model"].base_model.model.classifier.weight.data.cpu(),
            "classifier.bias"  : kwargs["model"].base_model.model.classifier.bias.data.cpu(),
        }
        torch.save(classifier_state, f"{save_path}/classifier.pt")
        print(f"✓ Epoch {epoch} saved — LoRA + classifier.")

training_args = TrainingArguments(
    output_dir=f"{DIRS['checkpoints2_v2_fixed']}",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=3e-5,
    warmup_steps=1000,
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

import safetensors.torch as st

# Epoch 3 checkpoint'ini yükle
epoch_path = f"{DIRS['checkpoints2_v2_fixed']}/epoch_3"

# LoRA adapter
weights = st.load_file(f"{epoch_path}/adapter_model.safetensors")
print("LoRA keys:")
for k in list(weights.keys())[:3]:
    print(f"  {k}: {weights[k].shape}")

# Classifier
classifier = torch.load(f"{epoch_path}/classifier.pt")
print("\nClassifier keys:")
for k, v in classifier.items():
    print(f"  {k}: {v.shape}")

# Base modeli yükle
base_model = DebertaV2ForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base",
    num_labels=2,
    torch_dtype=torch.float32,
    ignore_mismatched_sizes=True
)

# LoRA ekle (modules_to_save olmadan)
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16, lora_alpha=32, lora_dropout=0.1,
    target_modules=["query_proj", "value_proj"],
    bias="none"
)
test_model = get_peft_model(base_model, lora_config)

# LoRA ağırlıklarını yükle
new_weights = {}
for k, v in weights.items():
    if "lora_A.weight" in k:
        new_weights[k.replace("lora_A.weight", "lora_A.default.weight")] = v
    elif "lora_B.weight" in k:
        new_weights[k.replace("lora_B.weight", "lora_B.default.weight")] = v

# Classifier ağırlıklarını yükle
new_weights["base_model.model.classifier.weight"] = classifier["classifier.weight"]
new_weights["base_model.model.classifier.bias"]   = classifier["classifier.bias"]

missing, unexpected = test_model.load_state_dict(new_weights, strict=False)
print(f"Missing: {len(missing)} | Unexpected: {len(unexpected)}")

test_model = test_model.to(torch.float32).to(device)
test_model.eval()

# 5 örnek test et
for i in [0, 1, 2, 100, 200]:
    sample = test_dataset[i]
    input_ids      = sample["input_ids"].unsqueeze(0).to(device)
    attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
    with torch.no_grad():
        output = test_model(input_ids=input_ids, attention_mask=attention_mask)
        probs  = torch.softmax(output.logits, dim=-1)
        pred   = probs.argmax().item()
    print(f"Sample {i:3d} — Label: {sample['labels'].item()} | "
          f"Pred: {pred} | Probs: [{probs[0][0]:.4f}, {probs[0][1]:.4f}]")

# Pozitif labelları olan örnekleri test et
for i in range(len(test_dataset)):
    if test_dataset[i]["labels"].item() == 1:
        sample = test_dataset[i]
        input_ids      = sample["input_ids"].unsqueeze(0).to(device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
        with torch.no_grad():
            output = test_model(input_ids=input_ids, attention_mask=attention_mask)
            probs  = torch.softmax(output.logits, dim=-1)
            pred   = probs.argmax().item()
        print(f"Label: 1 | Pred: {pred} | Probs: [{probs[0][0]:.4f}, {probs[0][1]:.4f}]")
        break  # sadece ilk pozitif örneği göster

from transformers import Trainer, TrainingArguments
import numpy as np

training_args = TrainingArguments(
    output_dir="/tmp/eval_fixed",
    per_device_eval_batch_size=16,
    report_to="none"
)
trainer = Trainer(model=test_model, args=training_args)
preds_output = trainer.predict(test_dataset)
logits = preds_output.predictions
preds  = np.argmax(logits, axis=-1)
labels = preds_output.label_ids
probs  = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

print(f"\n{'='*40}")
print("DeBERTa V2 Fixed Results")
print(f"{'='*40}")
print(f"Accuracy : {accuracy_score(labels, preds):.4f}")
print(f"F1       : {f1_score(labels, preds):.4f}")
print(f"Precision: {precision_score(labels, preds):.4f}")
print(f"Recall   : {recall_score(labels, preds):.4f}")
print(f"ROC-AUC  : {roc_auc_score(labels, probs):.4f}")
