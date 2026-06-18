from google.colab import drive
drive.mount('/content/drive')
exec(open("/content/drive/MyDrive/imdb_peft_project/code/lora-imdb-classifier/00_colab_setup.py").read())

!pip install transformers peft accelerate torchao --upgrade -q
!pip install scikit-learn numpy pandas xgboost safetensors -q

import pandas as pd
import numpy as np
import torch
import os
import safetensors.torch as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from transformers import (RobertaTokenizer, RobertaForSequenceClassification,
                          DebertaV2Tokenizer, DebertaV2ForSequenceClassification,
                          Trainer, TrainingArguments)
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Define paths
DIRS["oof_v2"]      = f"{DIRS['root']}/oof_predictions_v2"
DIRS["oof_v2_best"] = f"{DIRS['root']}/oof_predictions_v2_best"
DIRS["checkpoints_v2"]    = f"{DIRS['root']}/checkpoints/roberta_lora_v2"
DIRS["checkpoints2_best"] = f"{DIRS['root']}/checkpoints/deberta_lora_v2_best"

# Load data
train_df = pd.read_parquet(f"{DIRS['root']}/train_df_v2.parquet")
test_df  = pd.read_parquet(f"{DIRS['root']}/test_df_v2.parquet")
y_train  = train_df["label"].values
y_test   = test_df["label"].values

# Load RoBERTa and DeBERTa OOF
roberta_oof = np.load(f"{DIRS['oof_v2']}/roberta_v2_fold4.npy")
deberta_oof = np.load(f"{DIRS['oof_v2_best']}/deberta_v2_best_fold4.npy")

# Load TF-IDF OOF
svm_oof = np.load(f"{DIRS['oof_v2']}/tfidf_svm_oof.npy")
lr_oof  = np.load(f"{DIRS['oof_v2']}/tfidf_lr_oof.npy")

print(f"RoBERTa OOF : {roberta_oof.shape}")
print(f"DeBERTa OOF : {deberta_oof.shape}")
print(f"SVM OOF     : {svm_oof.shape}")
print(f"LR OOF      : {lr_oof.shape}")
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

# Save to Drive
np.save(f"{DIRS['oof_v2']}/roberta_test_probs.npy", roberta_test_probs)
print("✓ Saved to Drive.")

del roberta_model, trainer
torch.cuda.empty_cache()

print("Loading DeBERTa V2 Best...")
deberta_tokenizer    = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
deberta_test_dataset = IMDBDataset(test_df, deberta_tokenizer)

base_model = DebertaV2ForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base",
    num_labels=2,
    torch_dtype=torch.float32,
    ignore_mismatched_sizes=True
)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=32, lora_alpha=64, lora_dropout=0.1,
    target_modules=["query_proj", "value_proj"],
    bias="none"
)
deberta_model = get_peft_model(base_model, lora_config)

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

deberta_model.load_state_dict(new_weights, strict=False)
deberta_model = deberta_model.to(torch.float32).to(device)
deberta_model.eval()
print("✓ DeBERTa V2 Best loaded.")

training_args = TrainingArguments(
    output_dir="/tmp/eval_d", per_device_eval_batch_size=32, report_to="none"
)
trainer = Trainer(model=deberta_model, args=training_args)
preds_output        = trainer.predict(deberta_test_dataset)
deberta_test_probs  = torch.softmax(
    torch.tensor(preds_output.predictions), dim=-1
)[:, 1].numpy()

print(f"✓ DeBERTa test predictions ready. Shape: {deberta_test_probs.shape}")

# Save to Drive
np.save(f"{DIRS['oof_v2']}/deberta_test_probs.npy", deberta_test_probs)
print("✓ Saved to Drive.")

del deberta_model, trainer
torch.cuda.empty_cache()

# Retrain TF-IDF models on full training set
print("Training TF-IDF models on full training set...")

tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2)
X_train_tfidf = tfidf.fit_transform(train_df["text_clean_lower"])
X_test_tfidf  = tfidf.transform(test_df["text_clean_lower"])

# Calibrated SVM
svm_final = CalibratedClassifierCV(LinearSVC(max_iter=2000), cv=3)
svm_final.fit(X_train_tfidf, y_train)
svm_test_probs = svm_final.predict_proba(X_test_tfidf)[:, 1]
print(f"✓ SVM test accuracy: {accuracy_score(y_test, (svm_test_probs > 0.5).astype(int)):.4f}")

# Logistic Regression
lr_tfidf = LogisticRegression(max_iter=1000, random_state=SEED)
lr_tfidf.fit(X_train_tfidf, y_train)
lr_test_probs = lr_tfidf.predict_proba(X_test_tfidf)[:, 1]
print(f"✓ LR  test accuracy: {accuracy_score(y_test, (lr_test_probs > 0.5).astype(int)):.4f}")

# Save
np.save(f"{DIRS['oof_v2']}/tfidf_svm_test_probs.npy", svm_test_probs)
np.save(f"{DIRS['oof_v2']}/tfidf_lr_test_probs.npy",  lr_test_probs)
print("✓ TF-IDF test predictions saved.")

def evaluate_meta(name, y_true, y_pred, y_prob):
    return {
        "model"    : name,
        "accuracy" : round(accuracy_score(y_true, y_pred), 4),
        "f1"       : round(f1_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall"   : round(recall_score(y_true, y_pred), 4),
        "roc_auc"  : round(roc_auc_score(y_true, y_prob), 4),
    }

# Meta feature matrices
X_meta_train_4 = np.column_stack([roberta_oof, deberta_oof, svm_oof, lr_oof])
X_meta_test_4  = np.column_stack([roberta_test_probs, deberta_test_probs,
                                   svm_test_probs, lr_test_probs])

print(f"Meta train shape: {X_meta_train_4.shape}")
print(f"Meta test shape : {X_meta_test_4.shape}")

results_4 = []

# 1. Logistic Regression
lr = LogisticRegression(random_state=SEED, max_iter=1000)
lr.fit(X_meta_train_4, y_train)
lr_probs = lr.predict_proba(X_meta_test_4)[:, 1]
lr_preds = lr.predict(X_meta_test_4)
res = evaluate_meta("Logistic Regression", y_test, lr_preds, lr_probs)
res["weights"] = lr.coef_[0].tolist()
results_4.append(res)
print(f"✓ LR — weights: {[round(w,3) for w in lr.coef_[0]]}")

# 2. MLP
mlp = MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu",
                    max_iter=500, random_state=SEED)
mlp.fit(X_meta_train_4, y_train)
mlp_probs = mlp.predict_proba(X_meta_test_4)[:, 1]
mlp_preds = mlp.predict(X_meta_test_4)
results_4.append(evaluate_meta("MLP", y_test, mlp_preds, mlp_probs))
print("✓ MLP complete.")

# 3. GradientBoosting
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                max_depth=3, random_state=SEED)
gb.fit(X_meta_train_4, y_train)
gb_probs = gb.predict_proba(X_meta_test_4)[:, 1]
gb_preds = gb.predict(X_meta_test_4)
results_4.append(evaluate_meta("GradientBoosting", y_test, gb_preds, gb_probs))
print("✓ GradientBoosting complete.")

# 4. XGBoost
xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                    random_state=SEED, eval_metric="logloss", verbosity=0)
xgb.fit(X_meta_train_4, y_train)
xgb_probs = xgb.predict_proba(X_meta_test_4)[:, 1]
xgb_preds = xgb.predict(X_meta_test_4)
results_4.append(evaluate_meta("XGBoost", y_test, xgb_preds, xgb_probs))
print("✓ XGBoost complete.")

# Results
results_df = pd.DataFrame(results_4)
print(f"\n{'='*70}")
print("META-LEARNER RESULTS (4 base models: RoBERTa + DeBERTa + SVM + LR)")
print(f"{'='*70}")
print(results_df[["model", "accuracy", "f1", "precision",
                   "recall", "roc_auc"]].to_string(index=False))

# Load data
train_df = pd.read_parquet(f"{DIRS['root']}/train_df_v2.parquet")
test_df  = pd.read_parquet(f"{DIRS['root']}/test_df_v2.parquet")
y_train  = train_df["label"].values
y_test   = test_df["label"].values

# Load RoBERTa and DeBERTa OOF
DIRS["oof_v2"]      = f"{DIRS['root']}/oof_predictions_v2"
DIRS["oof_v2_best"] = f"{DIRS['root']}/oof_predictions_v2_best"

roberta_oof = np.load(f"{DIRS['oof_v2']}/roberta_v2_fold4.npy")
deberta_oof = np.load(f"{DIRS['oof_v2_best']}/deberta_v2_best_fold4.npy")

print(f"RoBERTa OOF: {roberta_oof.shape}")
print(f"DeBERTa OOF: {deberta_oof.shape}")
print(f"Train: {len(train_df)} | Test: {len(test_df)}")

from sklearn.model_selection import StratifiedKFold

print("Training TF-IDF models with 5-fold OOF...")

# TF-IDF vectorizer
tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2)
X_train_tfidf = tfidf.fit_transform(train_df["text_clean_lower"])
X_test_tfidf  = tfidf.transform(test_df["text_clean_lower"])

# OOF predictions for both TF-IDF models
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

svm_oof = np.zeros(len(train_df))
lr_oof  = np.zeros(len(train_df))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_tfidf, y_train)):
    X_tr = X_train_tfidf[train_idx]
    X_val = X_train_tfidf[val_idx]
    y_tr = y_train[train_idx]
    y_val = y_train[val_idx]

    # Calibrated SVM
    svm = CalibratedClassifierCV(LinearSVC(max_iter=2000), cv=3)
    svm.fit(X_tr, y_tr)
    svm_oof[val_idx] = svm.predict_proba(X_val)[:, 1]

    # Logistic Regression
    lr_base = LogisticRegression(max_iter=1000, random_state=SEED)
    lr_base.fit(X_tr, y_tr)
    lr_oof[val_idx] = lr_base.predict_proba(X_val)[:, 1]

    fold_acc_svm = accuracy_score(y_val, (svm_oof[val_idx] > 0.5).astype(int))
    fold_acc_lr  = accuracy_score(y_val, (lr_oof[val_idx] > 0.5).astype(int))
    print(f"Fold {fold+1} — SVM: {fold_acc_svm:.4f} | LR: {fold_acc_lr:.4f}")

print(f"\nOOF SVM Accuracy: {accuracy_score(y_train, (svm_oof > 0.5).astype(int)):.4f}")
print(f"OOF LR  Accuracy: {accuracy_score(y_train, (lr_oof > 0.5).astype(int)):.4f}")

# Save OOF
np.save(f"{DIRS['root']}/oof_predictions_v2/tfidf_svm_oof.npy", svm_oof)
np.save(f"{DIRS['root']}/oof_predictions_v2/tfidf_lr_oof.npy",  lr_oof)
print("✓ TF-IDF OOF predictions saved.")

# Train final models on full training set
print("Training final TF-IDF models on full training set...")

# Calibrated SVM
svm_final = CalibratedClassifierCV(LinearSVC(max_iter=2000), cv=3)
svm_final.fit(X_train_tfidf, y_train)
svm_test_probs = svm_final.predict_proba(X_test_tfidf)[:, 1]
svm_acc = accuracy_score(y_test, (svm_test_probs > 0.5).astype(int))
print(f"✓ Calibrated SVM test accuracy: {svm_acc:.4f}")

# Logistic Regression
lr_final = LogisticRegression(max_iter=1000, random_state=SEED)
lr_final.fit(X_train_tfidf, y_train)
lr_test_probs = lr_final.predict_proba(X_test_tfidf)[:, 1]
lr_acc = accuracy_score(y_test, (lr_test_probs > 0.5).astype(int))
print(f"✓ TF-IDF LR test accuracy: {lr_acc:.4f}")

# Load RoBERTa and DeBERTa test probs
# These were saved in 06d_meta_learner_best
# We need to regenerate them or load from saved files
print("\nNote: RoBERTa and DeBERTa test probs need to be loaded from 06d notebook.")
