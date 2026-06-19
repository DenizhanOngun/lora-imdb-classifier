# =============================================================
# 06_stacking_ensemble.py
# =============================================================
# Trains and evaluates stacking ensemble meta-learners using
# OOF predictions from all four base models as meta-features.
#
# Meta-feature matrix (25000 x 4):
#   [roberta_oof, deberta_oof, tfidf_svm_oof, tfidf_lr_oof]
#
# Meta-learners compared: Logistic Regression, MLP, XGBoost
# Best result: LR → 96.18% accuracy, 96.19% F1
#
# Prerequisites: run scripts 02 through 05 first.
#
# Output:
#   data/results/stacking_ensemble.json
#   data/results/roc_curves.png
#   data/results/confusion_matrices.png
#
# Run:
#   python src/06_stacking_ensemble.py
# =============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
    confusion_matrix, roc_curve, auc,
)

from config import DATA_DIR, OOF_DIR, RESULTS_DIR, SEED
from utils import save_results

np.random.seed(SEED)
torch.manual_seed(SEED)


# ------------------------------------------------------------------
# 1. Load OOF predictions and labels
# ------------------------------------------------------------------
print("Loading OOF predictions...")
train_df = pd.read_parquet(os.path.join(DATA_DIR, "train_df_v2.parquet"))
test_df  = pd.read_parquet(os.path.join(DATA_DIR, "test_df_v2.parquet"))
y_train  = train_df["label"].values
y_test   = test_df["label"].values

roberta_oof = np.load(os.path.join(OOF_DIR, "roberta_oof.npy"))
deberta_oof = np.load(os.path.join(OOF_DIR, "deberta_oof.npy"))
svm_oof     = np.load(os.path.join(OOF_DIR, "tfidf_svm_oof.npy"))
lr_oof      = np.load(os.path.join(OOF_DIR, "tfidf_lr_oof.npy"))

roberta_test = np.load(os.path.join(OOF_DIR, "roberta_test_probs.npy"))
deberta_test = np.load(os.path.join(OOF_DIR, "deberta_test_probs.npy"))
svm_test     = np.load(os.path.join(OOF_DIR, "tfidf_svm_test_probs.npy"))
lr_test      = np.load(os.path.join(OOF_DIR, "tfidf_lr_test_probs.npy"))

print(f"RoBERTa OOF : {roberta_oof.shape}")
print(f"DeBERTa OOF : {deberta_oof.shape}")
print(f"SVM OOF     : {svm_oof.shape}")
print(f"LR OOF      : {lr_oof.shape}")


# ------------------------------------------------------------------
# 2. Build meta-feature matrices
# ------------------------------------------------------------------
X_meta_train = np.column_stack([roberta_oof, deberta_oof, svm_oof, lr_oof])
X_meta_test  = np.column_stack([roberta_test, deberta_test, svm_test, lr_test])

print(f"\nMeta train shape: {X_meta_train.shape}")
print(f"Meta test shape : {X_meta_test.shape}")


# ------------------------------------------------------------------
# 3. Train and evaluate meta-learners
# ------------------------------------------------------------------
def evaluate_meta(name, y_true, y_pred, y_prob) -> dict:
    return {
        "model"    : name,
        "accuracy" : round(float(accuracy_score(y_true, y_pred)), 4),
        "f1"       : round(float(f1_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred)), 4),
        "recall"   : round(float(recall_score(y_true, y_pred)), 4),
        "roc_auc"  : round(float(roc_auc_score(y_true, y_prob)), 4),
    }

results = []

# Logistic Regression
print("\nTraining Logistic Regression meta-learner...")
lr_meta = LogisticRegression(random_state=SEED, max_iter=1000)
lr_meta.fit(X_meta_train, y_train)
lr_probs = lr_meta.predict_proba(X_meta_test)[:, 1]
lr_preds = lr_meta.predict(X_meta_test)
res = evaluate_meta("Logistic Regression", y_test, lr_preds, lr_probs)
res["weights"] = {
    "roberta": round(float(lr_meta.coef_[0][0]), 4),
    "deberta": round(float(lr_meta.coef_[0][1]), 4),
    "svm"    : round(float(lr_meta.coef_[0][2]), 4),
    "lr_base": round(float(lr_meta.coef_[0][3]), 4),
}
results.append(res)
print(f"  Accuracy: {res['accuracy']:.4f} | F1: {res['f1']:.4f}")
print(f"  Weights: {res['weights']}")

# MLP
print("\nTraining MLP meta-learner...")
mlp = MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu",
                    max_iter=500, random_state=SEED)
mlp.fit(X_meta_train, y_train)
mlp_probs = mlp.predict_proba(X_meta_test)[:, 1]
mlp_preds = mlp.predict(X_meta_test)
res = evaluate_meta("MLP", y_test, mlp_preds, mlp_probs)
results.append(res)
print(f"  Accuracy: {res['accuracy']:.4f} | F1: {res['f1']:.4f}")

# GradientBoosting
print("\nTraining GradientBoosting meta-learner...")
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                max_depth=3, random_state=SEED)
gb.fit(X_meta_train, y_train)
gb_probs = gb.predict_proba(X_meta_test)[:, 1]
gb_preds = gb.predict(X_meta_test)
res = evaluate_meta("GradientBoosting", y_test, gb_preds, gb_probs)
results.append(res)
print(f"  Accuracy: {res['accuracy']:.4f} | F1: {res['f1']:.4f}")

# XGBoost
print("\nTraining XGBoost meta-learner...")
xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                    random_state=SEED, eval_metric="logloss", verbosity=0)
xgb.fit(X_meta_train, y_train)
xgb_probs = xgb.predict_proba(X_meta_test)[:, 1]
xgb_preds = xgb.predict(X_meta_test)
res = evaluate_meta("XGBoost", y_test, xgb_preds, xgb_probs)
results.append(res)
print(f"  Accuracy: {res['accuracy']:.4f} | F1: {res['f1']:.4f}")


# ------------------------------------------------------------------
# 4. Results table
# ------------------------------------------------------------------
results_df = pd.DataFrame(results)
print(f"\n{'='*70}")
print("STACKING ENSEMBLE RESULTS (4 base models)")
print(f"{'='*70}")
print(results_df[["model", "accuracy", "f1", "precision", "recall", "roc_auc"]].to_string(index=False))


# ------------------------------------------------------------------
# 5. ROC curves
# ------------------------------------------------------------------
os.makedirs(RESULTS_DIR, exist_ok=True)
models_list = [
    ("Logistic Regression", lr_preds,  lr_probs),
    ("MLP",                 mlp_preds, mlp_probs),
    ("GradientBoosting",    gb_preds,  gb_probs),
    ("XGBoost",             xgb_preds, xgb_probs),
]

plt.figure(figsize=(8, 6))
for name, _, probs in models_list:
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc_score   = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc_score:.4f})")
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — Meta-Learner Comparison (4 Base Models)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "roc_curves.png"), dpi=150)
plt.close()
print(f"\nROC curves saved to {RESULTS_DIR}/roc_curves.png")


# ------------------------------------------------------------------
# 6. Confusion matrices
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax, (name, preds, _) in zip(axes, models_list):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Neg", "Pos"],
                yticklabels=["Neg", "Pos"])
    ax.set_title(f"{name}\nAcc: {accuracy_score(y_test, preds):.4f}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrices.png"), dpi=150)
plt.close()
print(f"Confusion matrices saved to {RESULTS_DIR}/confusion_matrices.png")


# ------------------------------------------------------------------
# 7. Save results
# ------------------------------------------------------------------
save_results({"meta_learners": results}, "stacking_ensemble.json")
print("\nStacking ensemble complete.")
