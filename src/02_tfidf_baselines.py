# =============================================================
# 02_tfidf_baselines.py
# =============================================================
# Trains TF-IDF + SVM (calibrated) and TF-IDF + LR baselines.
# Generates 5-fold OOF predictions for the stacking ensemble
# and evaluates both models on the test set.
#
# Output:
#   data/oof_predictions/tfidf_svm_oof.npy
#   data/oof_predictions/tfidf_lr_oof.npy
#   data/oof_predictions/tfidf_svm_test_probs.npy
#   data/oof_predictions/tfidf_lr_test_probs.npy
#   data/results/tfidf_baselines.json
#
# Run:
#   python src/02_tfidf_baselines.py
# =============================================================

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)

from config import DATA_DIR, OOF_DIR, SEED, N_FOLDS, TFIDF_CONFIG
from utils import save_results

np.random.seed(SEED)


# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
print("Loading preprocessed data...")
train_df = pd.read_parquet(os.path.join(DATA_DIR, "train_df_v2.parquet"))
test_df  = pd.read_parquet(os.path.join(DATA_DIR, "test_df_v2.parquet"))
y_train  = train_df["label"].values
y_test   = test_df["label"].values
print(f"Train: {len(train_df)} | Test: {len(test_df)}")


# ------------------------------------------------------------------
# 2. TF-IDF vectorization
# ------------------------------------------------------------------
print("\nFitting TF-IDF vectorizer...")
tfidf = TfidfVectorizer(**TFIDF_CONFIG)
X_train = tfidf.fit_transform(train_df["text_clean_lower"])
X_test  = tfidf.transform(test_df["text_clean_lower"])
print(f"Vocabulary size: {len(tfidf.vocabulary_):,}")


# ------------------------------------------------------------------
# 3. 5-fold OOF predictions
# ------------------------------------------------------------------
print(f"\nRunning {N_FOLDS}-fold OOF training...")
skf     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
svm_oof = np.zeros(len(train_df))
lr_oof  = np.zeros(len(train_df))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr  = X_train[train_idx]
    X_val = X_train[val_idx]
    y_tr  = y_train[train_idx]
    y_val = y_train[val_idx]

    # Calibrated SVM
    svm = CalibratedClassifierCV(LinearSVC(max_iter=2000), cv=3)
    svm.fit(X_tr, y_tr)
    svm_oof[val_idx] = svm.predict_proba(X_val)[:, 1]

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=SEED)
    lr.fit(X_tr, y_tr)
    lr_oof[val_idx] = lr.predict_proba(X_val)[:, 1]

    fold_svm = accuracy_score(y_val, (svm_oof[val_idx] > 0.5).astype(int))
    fold_lr  = accuracy_score(y_val, (lr_oof[val_idx]  > 0.5).astype(int))
    print(f"  Fold {fold+1} — SVM: {fold_svm:.4f} | LR: {fold_lr:.4f}")

oof_svm_acc = accuracy_score(y_train, (svm_oof > 0.5).astype(int))
oof_lr_acc  = accuracy_score(y_train, (lr_oof  > 0.5).astype(int))
print(f"\nOOF SVM Accuracy: {oof_svm_acc:.4f}")
print(f"OOF LR  Accuracy: {oof_lr_acc:.4f}")


# ------------------------------------------------------------------
# 4. Final models trained on full training set
# ------------------------------------------------------------------
print("\nTraining final models on full training set...")

svm_final = CalibratedClassifierCV(LinearSVC(max_iter=2000), cv=3)
svm_final.fit(X_train, y_train)
svm_test_probs = svm_final.predict_proba(X_test)[:, 1]
svm_test_preds = (svm_test_probs > 0.5).astype(int)

lr_final = LogisticRegression(max_iter=1000, random_state=SEED)
lr_final.fit(X_train, y_train)
lr_test_probs = lr_final.predict_proba(X_test)[:, 1]
lr_test_preds = (lr_test_probs > 0.5).astype(int)

svm_acc = accuracy_score(y_test, svm_test_preds)
lr_acc  = accuracy_score(y_test, lr_test_preds)
print(f"TF-IDF + SVM Test Accuracy : {svm_acc:.4f}")
print(f"TF-IDF + LR  Test Accuracy : {lr_acc:.4f}")


# ------------------------------------------------------------------
# 5. Save OOF and test predictions
# ------------------------------------------------------------------
os.makedirs(OOF_DIR, exist_ok=True)
np.save(os.path.join(OOF_DIR, "tfidf_svm_oof.npy"),        svm_oof)
np.save(os.path.join(OOF_DIR, "tfidf_lr_oof.npy"),         lr_oof)
np.save(os.path.join(OOF_DIR, "tfidf_svm_test_probs.npy"), svm_test_probs)
np.save(os.path.join(OOF_DIR, "tfidf_lr_test_probs.npy"),  lr_test_probs)
print(f"\nOOF and test predictions saved to {OOF_DIR}/")

save_results({
    "tfidf_svm": {
        "test_accuracy" : round(float(svm_acc), 4),
        "test_f1"       : round(float(f1_score(y_test, svm_test_preds)), 4),
        "test_precision": round(float(precision_score(y_test, svm_test_preds)), 4),
        "test_recall"   : round(float(recall_score(y_test, svm_test_preds)), 4),
        "oof_accuracy"  : round(float(oof_svm_acc), 4),
    },
    "tfidf_lr": {
        "test_accuracy" : round(float(lr_acc), 4),
        "test_f1"       : round(float(f1_score(y_test, lr_test_preds)), 4),
        "test_precision": round(float(precision_score(y_test, lr_test_preds)), 4),
        "test_recall"   : round(float(recall_score(y_test, lr_test_preds)), 4),
        "test_roc_auc"  : round(float(roc_auc_score(y_test, lr_test_probs)), 4),
        "oof_accuracy"  : round(float(oof_lr_acc), 4),
    },
}, "tfidf_baselines.json")

print("\nTF-IDF baselines complete.")
