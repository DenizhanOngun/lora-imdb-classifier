from google.colab import drive
drive.mount('/content/drive')
exec(open("/content/drive/MyDrive/imdb_peft_project/code/lora-imdb-classifier/00_colab_setup.py").read())

!pip install transformers peft accelerate torchao safetensors scikit-learn -q

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
from sklearn.metrics import accuracy_score, f1_score

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Define paths
DIRS["checkpoints_v2"]    = f"{DIRS['root']}/checkpoints/roberta_lora_v2"
DIRS["checkpoints2_best"] = f"{DIRS['root']}/checkpoints/deberta_lora_v2_best"
DIRS["oof_v2"]            = f"{DIRS['root']}/oof_predictions_v2"

# Load test data
test_df = pd.read_parquet(f"{DIRS['root']}/test_df_v2.parquet")
y_test  = test_df["label"].values
print(f"Test: {len(test_df)} samples")

# Load saved test probabilities
roberta_test_probs = np.load(f"{DIRS['oof_v2']}/roberta_test_probs.npy")
deberta_test_probs = np.load(f"{DIRS['oof_v2']}/deberta_test_probs.npy")
svm_test_probs     = np.load(f"{DIRS['oof_v2']}/tfidf_svm_test_probs.npy")
lr_test_probs      = np.load(f"{DIRS['oof_v2']}/tfidf_lr_test_probs.npy")

print(f"✓ All test probabilities loaded.")

# Ensemble predictions (LR weights from 06e)
# RoBERTa: 2.542, DeBERTa: 2.883, SVM: 2.37, LR: 1.119
from sklearn.linear_model import LogisticRegression
train_df    = pd.read_parquet(f"{DIRS['root']}/train_df_v2.parquet")
y_train     = train_df["label"].values

roberta_oof = np.load(f"{DIRS['oof_v2']}/roberta_v2_fold4.npy")
deberta_oof = np.load(f"{DIRS['root']}/oof_predictions_v2_best/deberta_v2_best_fold4.npy")
svm_oof     = np.load(f"{DIRS['oof_v2']}/tfidf_svm_oof.npy")
lr_oof      = np.load(f"{DIRS['oof_v2']}/tfidf_lr_oof.npy")

X_meta_train = np.column_stack([roberta_oof, deberta_oof, svm_oof, lr_oof])
X_meta_test  = np.column_stack([roberta_test_probs, deberta_test_probs,
                                 svm_test_probs, lr_test_probs])

lr_meta = LogisticRegression(random_state=SEED, max_iter=1000)
lr_meta.fit(X_meta_train, y_train)
ensemble_probs = lr_meta.predict_proba(X_meta_test)[:, 1]
ensemble_preds = lr_meta.predict(X_meta_test)

print(f"Ensemble accuracy: {accuracy_score(y_test, ensemble_preds):.4f}")

# Find misclassified examples
misclassified_idx = np.where(ensemble_preds != y_test)[0]
print(f"Total misclassified: {len(misclassified_idx)} ({len(misclassified_idx)/len(y_test)*100:.2f}%)")

# False Positives: negative predicted as positive
fp_idx = np.where((ensemble_preds == 1) & (y_test == 0))[0]
# False Negatives: positive predicted as negative
fn_idx = np.where((ensemble_preds == 0) & (y_test == 1))[0]
print(f"False Positives: {len(fp_idx)} ({len(fp_idx)/len(y_test)*100:.2f}%)")
print(f"False Negatives: {len(fn_idx)} ({len(fn_idx)/len(y_test)*100:.2f}%)")

# Confidence of ensemble on misclassified examples
fp_confidence = ensemble_probs[fp_idx]
fn_confidence = 1 - ensemble_probs[fn_idx]

print(f"\nFalse Positive avg confidence : {fp_confidence.mean():.4f}")
print(f"False Negative avg confidence : {fn_confidence.mean():.4f}")

# Find most confident wrong predictions
fp_sorted = fp_idx[np.argsort(fp_confidence)[::-1]]
fn_sorted = fn_idx[np.argsort(fn_confidence)[::-1]]

print("\nTop 5 most confident False Positives:")
for i, idx in enumerate(fp_sorted[:5]):
    conf = ensemble_probs[idx]
    text = test_df.iloc[idx]["text_clean"][:200]
    print(f"\n[{i+1}] Confidence: {conf:.4f} | True: Negative")
    print(f"Text: {text}...")

print("\nTop 5 most confident False Negatives:")
for i, idx in enumerate(fn_sorted[:5]):
    conf = 1 - ensemble_probs[idx]
    text = test_df.iloc[idx]["text_clean"][:200]
    print(f"\n[{i+1}] Confidence: {conf:.4f} | True: Positive")
    print(f"Text: {text}...")

# Analyze error patterns
print("="*60)
print("ERROR PATTERN ANALYSIS")
print("="*60)

# 1. Negation analysis
negation_words = ["not", "never", "no", "n't", "neither", "nor", "hardly", "barely"]
negation_in_fp = sum(1 for idx in fp_idx
                     if any(w in test_df.iloc[idx]["text_clean"].lower()
                            for w in negation_words))
negation_in_fn = sum(1 for idx in fn_idx
                     if any(w in test_df.iloc[idx]["text_clean"].lower()
                            for w in negation_words))

print(f"\nNegation words in False Positives: {negation_in_fp}/{len(fp_idx)} ({negation_in_fp/len(fp_idx)*100:.1f}%)")
print(f"Negation words in False Negatives: {negation_in_fn}/{len(fn_idx)} ({negation_in_fn/len(fn_idx)*100:.1f}%)")

# 2. Review length analysis
fp_lengths = [len(test_df.iloc[idx]["text_clean"].split()) for idx in fp_idx]
fn_lengths = [len(test_df.iloc[idx]["text_clean"].split()) for idx in fn_idx]
correct_lengths = [len(test_df.iloc[idx]["text_clean"].split())
                   for idx in range(len(test_df)) if ensemble_preds[idx] == y_test[idx]]

print(f"\nAvg review length:")
print(f"  Correctly classified : {np.mean(correct_lengths):.1f} words")
print(f"  False Positives      : {np.mean(fp_lengths):.1f} words")
print(f"  False Negatives      : {np.mean(fn_lengths):.1f} words")

# 3. Long review error rate
long_idx  = [i for i in range(len(test_df))
             if len(test_df.iloc[i]["text_clean"].split()) > 400]
short_idx = [i for i in range(len(test_df))
             if len(test_df.iloc[i]["text_clean"].split()) <= 400]

long_errors  = sum(1 for i in long_idx  if ensemble_preds[i] != y_test[i])
short_errors = sum(1 for i in short_idx if ensemble_preds[i] != y_test[i])

print(f"\nError rate for long reviews  (>400 words): {long_errors/len(long_idx)*100:.2f}%")
print(f"Error rate for short reviews (≤400 words): {short_errors/len(short_idx)*100:.2f}%")

# 4. Model disagreement analysis
roberta_preds = (roberta_test_probs > 0.5).astype(int)
deberta_preds = (deberta_test_probs > 0.5).astype(int)
disagree_idx  = np.where(roberta_preds != deberta_preds)[0]

print(f"\nModel disagreement:")
print(f"  Total disagreements      : {len(disagree_idx)} ({len(disagree_idx)/len(y_test)*100:.1f}%)")

disagree_errors = sum(1 for i in disagree_idx if ensemble_preds[i] != y_test[i])
print(f"  Errors in disagreements  : {disagree_errors} ({disagree_errors/len(disagree_idx)*100:.1f}%)")

agree_idx   = np.where(roberta_preds == deberta_preds)[0]
agree_errors = sum(1 for i in agree_idx if ensemble_preds[i] != y_test[i])
print(f"  Errors when both agree   : {agree_errors} ({agree_errors/len(agree_idx)*100:.1f}%)")

resolved = sum(1 for i in disagree_idx
               if ensemble_preds[i] == y_test[i] and
               (roberta_preds[i] != y_test[i] or deberta_preds[i] != y_test[i]))
print(f"  Disagreements resolved by ensemble: {resolved} ({resolved/len(disagree_idx)*100:.1f}%)")

# Select interesting examples for the report

print("EXAMPLE 1 — Highly confident False Positive (sarcasm/irony):")
idx = fp_sorted[2]  # "pure genius" example
print(f"True label: Negative | Ensemble confidence: {ensemble_probs[idx]:.4f}")
print(f"RoBERTa: {roberta_test_probs[idx]:.4f} | DeBERTa: {deberta_test_probs[idx]:.4f}")
print(f"\nText (first 300 chars):")
print(test_df.iloc[idx]["text_clean"][:300])

print("\n" + "="*60)

print("\nEXAMPLE 2 — Highly confident False Negative (negative words in positive review):")
idx = fn_sorted[0]  # SPOILERS example
print(f"True label: Positive | Ensemble confidence: {1-ensemble_probs[idx]:.4f}")
print(f"RoBERTa: {roberta_test_probs[idx]:.4f} | DeBERTa: {deberta_test_probs[idx]:.4f}")
print(f"\nText (first 300 chars):")
print(test_df.iloc[idx]["text_clean"][:300])

print("\n" + "="*60)

print("\nEXAMPLE 3 — Model disagreement example:")
# Find a disagreement where RoBERTa is right but DeBERTa is wrong
for idx in disagree_idx:
    if roberta_preds[idx] == y_test[idx] and deberta_preds[idx] != y_test[idx]:
        print(f"True label: {y_test[idx]} | RoBERTa: {roberta_test_probs[idx]:.4f} | DeBERTa: {deberta_test_probs[idx]:.4f}")
        print(f"Ensemble: {ensemble_probs[idx]:.4f}")
        print(f"\nText (first 300 chars):")
        print(test_df.iloc[idx]["text_clean"][:300])
        break
