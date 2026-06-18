from google.colab import drive
drive.mount('/content/drive')
exec(open("/content/drive/MyDrive/imdb_peft_project/code/lora-imdb-classifier/00_colab_setup.py").read())

!pip install transformers peft accelerate torchao scikit-learn xgboost safetensors --upgrade -q

import pandas as pd
import numpy as np
import torch
import os
import safetensors.torch as st
from transformers import (RobertaTokenizer, RobertaForSequenceClassification,
                          DebertaV2Tokenizer, DebertaV2ForSequenceClassification,
                          Trainer, TrainingArguments)
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score,
                             roc_auc_score, confusion_matrix,
                             roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Define paths
DIRS["oof_v2"]            = f"{DIRS['root']}/oof_predictions_v2"
DIRS["oof_v2_best"]       = f"{DIRS['root']}/oof_predictions_v2_best"
DIRS["checkpoints_v2"]    = f"{DIRS['root']}/checkpoints/roberta_lora_v2"
DIRS["checkpoints2_best"] = f"{DIRS['root']}/checkpoints/deberta_lora_v2_best"

# Load data
train_df = pd.read_parquet(f"{DIRS['root']}/train_df_v2.parquet")
test_df  = pd.read_parquet(f"{DIRS['root']}/test_df_v2.parquet")
y_test   = test_df["label"].values

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

print("✓ Dataset class ready.")

print("Loading RoBERTa V2...")
roberta_tokenizer    = RobertaTokenizer.from_pretrained("roberta-base")
roberta_test_dataset = IMDBDataset(test_df, roberta_tokenizer)

base_model    = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
roberta_model = PeftModel.from_pretrained(base_model, f"{DIRS['checkpoints_v2']}/final")
roberta_model.eval()
print("✓ RoBERTa V2 loaded.")

training_args = TrainingArguments(
    output_dir="/tmp/eval_r", per_device_eval_batch_size=64, report_to="none"
)
trainer = Trainer(model=roberta_model, args=training_args)
preds_output       = trainer.predict(roberta_test_dataset)
roberta_test_probs = torch.softmax(
    torch.tensor(preds_output.predictions), dim=-1
)[:, 1].numpy()

print(f"✓ RoBERTa test predictions ready. Shape: {roberta_test_probs.shape}")
print(f"  Sample probs: {roberta_test_probs[:5]}")

del roberta_model, trainer
torch.cuda.empty_cache()

print("Loading DeBERTa V2 Best...")
deberta_tokenizer    = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
deberta_test_dataset = IMDBDataset(test_df, deberta_tokenizer)

# Load base model
base_model = DebertaV2ForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base",
    num_labels=2,
    torch_dtype=torch.float32,
    ignore_mismatched_sizes=True
)

# Add LoRA r=32
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=32, lora_alpha=64, lora_dropout=0.1,
    target_modules=["query_proj", "value_proj"],
    bias="none"
)
deberta_model = get_peft_model(base_model, lora_config)

# Load weights
epoch3_path = f"{DIRS['checkpoints2_best']}/epoch_3"
weights     = st.load_file(f"{epoch3_path}/adapter_model.safetensors")
extra       = torch.load(f"{epoch3_path}/extra_weights.pt", map_location="cpu")

new_weights = {}
for k, v in weights.items():
    if "lora_A.weight" in k:
        new_weights[k.replace("lora_A.weight", "lora_A.default.weight")] = v
    elif "lora_B.weight" in k:
        new_weights[k.replace("lora_B.weight", "lora_B.default.weight")] = v

new_weights["base_model.model.classifier.modules_to_save.default.weight"] = extra["classifier.weight"]
new_weights["base_model.model.classifier.modules_to_save.default.bias"]   = extra["classifier.bias"]
new_weights["base_model.model.classifier.original_module.weight"]          = extra["classifier.weight"]
new_weights["base_model.model.classifier.original_module.bias"]            = extra["classifier.bias"]
new_weights["base_model.model.pooler.dense.weight"]                        = extra["pooler.weight"]
new_weights["base_model.model.pooler.dense.bias"]                          = extra["pooler.bias"]

missing, unexpected = deberta_model.load_state_dict(new_weights, strict=False)
deberta_model = deberta_model.to(torch.float32).to(device)
deberta_model.eval()
print(f"Missing: {len(missing)} | Unexpected: {len(unexpected)}")

# Quick test
sample = deberta_test_dataset[0]
with torch.no_grad():
    out   = deberta_model(
        input_ids      = sample["input_ids"].unsqueeze(0).to(device),
        attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
    )
    probs = torch.softmax(out.logits, dim=-1)
print(f"Quick test — Label: {sample['labels'].item()} | Probs: {probs[0].tolist()}")

training_args = TrainingArguments(
    output_dir="/tmp/eval_d",
    per_device_eval_batch_size=32,
    report_to="none"
)
trainer = Trainer(model=deberta_model, args=training_args)
preds_output        = trainer.predict(deberta_test_dataset)
deberta_test_probs  = torch.softmax(
    torch.tensor(preds_output.predictions), dim=-1
)[:, 1].numpy()

print(f"✓ DeBERTa Best test predictions ready. Shape: {deberta_test_probs.shape}")
print(f"  Sample probs: {deberta_test_probs[:5]}")

del deberta_model, trainer
torch.cuda.empty_cache()

# Load OOF predictions
roberta_oof = np.load(f"{DIRS['oof_v2']}/roberta_v2_fold4.npy")
deberta_oof = np.load(f"{DIRS['oof_v2_best']}/deberta_v2_best_fold4.npy")

print(f"RoBERTa OOF shape: {roberta_oof.shape}")
print(f"DeBERTa OOF shape: {deberta_oof.shape}")

y_train = train_df["label"].values

# Meta feature matrices
X_meta_train = np.column_stack([roberta_oof, deberta_oof])
X_meta_test  = np.column_stack([roberta_test_probs, deberta_test_probs])

print(f"Meta train shape: {X_meta_train.shape}")
print(f"Meta test shape : {X_meta_test.shape}")
