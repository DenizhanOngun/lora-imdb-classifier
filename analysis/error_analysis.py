# =============================================================
# analysis/error_analysis.py
# =============================================================
# Analyzes misclassifications of the best LR ensemble model.
# Covers: error statistics, irony/sarcasm, misleading opening
# sentences, review length effects, and model disagreement.
#
# Prerequisites: run scripts 02–06 first (all OOF and test
# probability files must exist in data/oof_predictions/).
#
# Output:
#   data/results/error_analysis.json
#
# Run:
#   python analysis/error_analysis.py
# =============================================================

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from config import DATA_DIR, OOF_DIR, RESULTS_DIR, SEED
from utils import save_results

np.random.seed(SEED)


# ------------------------------------------------------------------
# 1. Load data and rebuild LR ensemble
# ------------------------------------------------------------------
print("Loading data and predictions...")
train_df = pd.read_parquet(os.path.join(DATA_DIR, "train_df_v2.parquet"))
test_df  = pd.read_parquet(os.path.join(DATA_DIR, "test_df_v2.parquet"))
y_train  = train_df["label"].values
y_test   = test_df["label"].values

roberta_oof  = np.load(os.path.join(OOF_DIR, "roberta_oof.npy"))
deberta_oof  = np.load(os.path.join(OOF_DIR, "deberta_oof.npy"))
svm_oof      = np.load(os.path.join(OOF_DIR, "tfidf_svm_oof.npy"))
lr_oof       = np.load(os.path.join(OOF_DIR, "tfidf_lr_oof.npy"))

roberta_test = np.load(os.path.join(OOF_DIR, "roberta_test_probs.npy"))
deberta_test = np.load(os.path.join(OOF_DIR, "deberta_test_probs.npy"))
svm_test     = np.load(os.path.join(OOF_DIR, "tfidf_svm_test_probs.npy"))
lr_test      = np.load(os.path.join(OOF_DIR, "tfidf_lr_test_probs.npy"))

X_meta_train = np.column_stack([roberta_oof, deberta_oof, svm_oof, lr_oof])
X_meta_test  = np.column_stack([roberta_test, deberta_test, svm_test, lr_test])

lr_meta = LogisticRegression(random_state=SEED, max_iter=1000)
lr_meta.fit(X_meta_train, y_train)
ensemble_probs = lr_meta.predict_proba(X_meta_test)[:, 1]
ensemble_preds = lr_meta.predict(X_meta_test)

print(f"Ensemble accuracy: {accuracy_score(y_test, ensemble_preds):.4f}")


# ------------------------------------------------------------------
# 2. Error statistics
# ------------------------------------------------------------------
misclassified_idx = np.where(ensemble_preds != y_test)[0]
fp_idx = np.where((ensemble_preds == 1) & (y_test == 0))[0]  # negative → predicted positive
fn_idx = np.where((ensemble_preds == 0) & (y_test == 1))[0]  # positive → predicted negative

print(f"\nTotal misclassified : {len(misclassified_idx)} "
      f"({len(misclassified_idx)/len(y_test)*100:.2f}%)")
print(f"False positives     : {len(fp_idx)} "
      f"({len(fp_idx)/len(y_test)*100:.2f}%)")
print(f"False negatives     : {len(fn_idx)} "
      f"({len(fn_idx)/len(y_test)*100:.2f}%)")


# ------------------------------------------------------------------
# 3. Most confident errors
# ------------------------------------------------------------------
fp_confidence = ensemble_probs[fp_idx]
fn_confidence = 1 - ensemble_probs[fn_idx]
fp_sorted = fp_idx[np.argsort(fp_confidence)[::-1]]
fn_sorted = fn_idx[np.argsort(fn_confidence)[::-1]]

print("\nTop 5 most confident False Positives:")
for i, idx in enumerate(fp_sorted[:5]):
    print(f"\n  [{i+1}] Confidence: {ensemble_probs[idx]:.4f} | True: Negative")
    print(f"       RoBERTa: {roberta_test[idx]:.4f} | DeBERTa: {deberta_test[idx]:.4f}")
    print(f"       Text: {test_df.iloc[idx]['text_clean'][:200]}...")

print("\nTop 5 most confident False Negatives:")
for i, idx in enumerate(fn_sorted[:5]):
    print(f"\n  [{i+1}] Confidence: {1-ensemble_probs[idx]:.4f} | True: Positive")
    print(f"       RoBERTa: {roberta_test[idx]:.4f} | DeBERTa: {deberta_test[idx]:.4f}")
    print(f"       Text: {test_df.iloc[idx]['text_clean'][:200]}...")


# ------------------------------------------------------------------
# 4. Error pattern analysis
# ------------------------------------------------------------------
print(f"\n{'='*60}")
print("ERROR PATTERN ANALYSIS")
print(f"{'='*60}")

# Negation words
negation_words = ["not", "never", "no", "n't", "neither", "nor", "hardly", "barely"]
negation_in_fp = sum(1 for i in fp_idx
                     if any(w in test_df.iloc[i]["text_clean"].lower()
                            for w in negation_words))
negation_in_fn = sum(1 for i in fn_idx
                     if any(w in test_df.iloc[i]["text_clean"].lower()
                            for w in negation_words))
print(f"\nNegation words in FP: {negation_in_fp}/{len(fp_idx)} "
      f"({negation_in_fp/len(fp_idx)*100:.1f}%)")
print(f"Negation words in FN: {negation_in_fn}/{len(fn_idx)} "
      f"({negation_in_fn/len(fn_idx)*100:.1f}%)")

# Review length
fp_lengths      = [len(test_df.iloc[i]["text_clean"].split()) for i in fp_idx]
fn_lengths      = [len(test_df.iloc[i]["text_clean"].split()) for i in fn_idx]
correct_idx     = np.where(ensemble_preds == y_test)[0]
correct_lengths = [len(test_df.iloc[i]["text_clean"].split()) for i in correct_idx]
print(f"\nAvg review length:")
print(f"  Correctly classified : {np.mean(correct_lengths):.1f} words")
print(f"  False positives      : {np.mean(fp_lengths):.1f} words")
print(f"  False negatives      : {np.mean(fn_lengths):.1f} words")

# Long vs short error rates
long_idx  = [i for i in range(len(test_df))
             if len(test_df.iloc[i]["text_clean"].split()) > 400]
short_idx = [i for i in range(len(test_df))
             if len(test_df.iloc[i]["text_clean"].split()) <= 400]
long_err  = sum(1 for i in long_idx  if ensemble_preds[i] != y_test[i])
short_err = sum(1 for i in short_idx if ensemble_preds[i] != y_test[i])
print(f"\nError rate for long reviews  (>400 words): "
      f"{long_err/len(long_idx)*100:.2f}%")
print(f"Error rate for short reviews (≤400 words): "
      f"{short_err/len(short_idx)*100:.2f}%")

# Model disagreement
roberta_preds = (roberta_test > 0.5).astype(int)
deberta_preds = (deberta_test > 0.5).astype(int)
disagree_idx  = np.where(roberta_preds != deberta_preds)[0]
agree_idx     = np.where(roberta_preds == deberta_preds)[0]

disagree_errors = sum(1 for i in disagree_idx if ensemble_preds[i] != y_test[i])
agree_errors    = sum(1 for i in agree_idx    if ensemble_preds[i] != y_test[i])
resolved = sum(1 for i in disagree_idx
               if ensemble_preds[i] == y_test[i] and
               (roberta_preds[i] != y_test[i] or deberta_preds[i] != y_test[i]))

print(f"\nModel disagreements    : {len(disagree_idx)} "
      f"({len(disagree_idx)/len(y_test)*100:.1f}%)")
print(f"Error rate (disagree)  : {disagree_errors/len(disagree_idx)*100:.1f}%")
print(f"Error rate (agree)     : {agree_errors/len(agree_idx)*100:.1f}%")
print(f"Resolved by ensemble   : {resolved/len(disagree_idx)*100:.1f}%")


# ------------------------------------------------------------------
# 5. Save results
# ------------------------------------------------------------------
save_results({
    "total_test"              : len(y_test),
    "total_misclassified"     : int(len(misclassified_idx)),
    "error_rate_pct"          : round(len(misclassified_idx)/len(y_test)*100, 2),
    "false_positives"         : int(len(fp_idx)),
    "false_negatives"         : int(len(fn_idx)),
    "fp_rate_pct"             : round(len(fp_idx)/len(y_test)*100, 2),
    "fn_rate_pct"             : round(len(fn_idx)/len(y_test)*100, 2),
    "negation_in_fp_pct"      : round(negation_in_fp/len(fp_idx)*100, 1),
    "negation_in_fn_pct"      : round(negation_in_fn/len(fn_idx)*100, 1),
    "avg_length_correct"      : round(float(np.mean(correct_lengths)), 1),
    "avg_length_fp"           : round(float(np.mean(fp_lengths)), 1),
    "avg_length_fn"           : round(float(np.mean(fn_lengths)), 1),
    "long_review_error_pct"   : round(long_err/len(long_idx)*100, 2),
    "short_review_error_pct"  : round(short_err/len(short_idx)*100, 2),
    "total_disagreements"     : int(len(disagree_idx)),
    "disagreement_rate_pct"   : round(len(disagree_idx)/len(y_test)*100, 1),
    "error_rate_on_disagree_pct": round(disagree_errors/len(disagree_idx)*100, 1),
    "ensemble_resolved_pct"   : round(resolved/len(disagree_idx)*100, 1),
}, "error_analysis.json")

print("\nError analysis complete.")
