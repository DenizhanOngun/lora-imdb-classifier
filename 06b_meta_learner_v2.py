from google.colab import drive
drive.mount('/content/drive')
exec(open("/content/drive/MyDrive/imdb_peft_project/code/lora-imdb-classifier/00_colab_setup.py").read())

!pip install scikit-learn numpy pandas matplotlib seaborn -q

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score,
                             roc_auc_score, confusion_matrix,
                             roc_curve, auc)
import json, time

SEED = 42
np.random.seed(SEED)

print("✓ All imports ready.")

import os

# V2 OOF klasörü
DIRS["oof_v2"] = f"{DIRS['root']}/oof_predictions_v2"

# Fold 4 — tüm 25k OOF tahminlerini içeriyor
roberta_oof_v2 = np.load(f"{DIRS['oof_v2']}/roberta_v2_fold4.npy")
deberta_oof_v2 = np.load(f"{DIRS['oof_v2']}/deberta_v2_fold4.npy")

print(f"RoBERTa V2 OOF shape: {roberta_oof_v2.shape}")
print(f"DeBERTa V2 OOF shape: {deberta_oof_v2.shape}")

# Train labels
train_df = pd.read_parquet(f"{DIRS['root']}/train_df_v2.parquet")
y_train  = train_df["label"].values

# Feature matrix
X_meta = np.column_stack([roberta_oof_v2, deberta_oof_v2])
print(f"Meta feature matrix : {X_meta.shape}")
print(f"Sample: {X_meta[0]} → label: {y_train[0]}")

X_train_meta, X_val_meta, y_train_meta, y_val_meta = train_test_split(
    X_meta, y_train,
    test_size=0.2,
    random_state=SEED,
    stratify=y_train
)

print(f"Meta train : {X_train_meta.shape}")
print(f"Meta val   : {X_val_meta.shape}")
print(f"\nLabel distribution (train): {np.bincount(y_train_meta)}")
print(f"Label distribution (val)  : {np.bincount(y_val_meta)}")

def evaluate_meta(name, y_true, y_pred, y_prob):
    return {
        "model"    : name,
        "accuracy" : round(accuracy_score(y_true, y_pred), 4),
        "f1"       : round(f1_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall"   : round(recall_score(y_true, y_pred), 4),
        "roc_auc"  : round(roc_auc_score(y_true, y_prob), 4),
    }

results = []

# ── 1. Weighted Average ───────────────────────────────────────
_, X_val_r, _, y_val_check = train_test_split(
    roberta_oof_v2, y_train, test_size=0.2, random_state=SEED, stratify=y_train
)
_, X_val_d = train_test_split(
    deberta_oof_v2, test_size=0.2, random_state=SEED
)
wa_probs = X_val_r * 0.5 + X_val_d * 0.5
wa_preds = (wa_probs > 0.5).astype(int)
results.append(evaluate_meta("Weighted Average", y_val_meta, wa_preds, wa_probs))
print("✓ Weighted Average complete.")

# ── 2. Logistic Regression ────────────────────────────────────
start = time.time()
lr = LogisticRegression(random_state=SEED, max_iter=1000)
lr.fit(X_train_meta, y_train_meta)
lr_time = time.time() - start
lr_probs = lr.predict_proba(X_val_meta)[:, 1]
lr_preds = lr.predict(X_val_meta)
res = evaluate_meta("Logistic Regression", y_val_meta, lr_preds, lr_probs)
res["train_time"] = round(lr_time, 3)
res["roberta_weight"] = round(lr.coef_[0][0], 4)
res["deberta_weight"] = round(lr.coef_[0][1], 4)
results.append(res)
print(f"✓ Logistic Regression complete. ({lr_time:.2f}s)")
print(f"  RoBERTa weight: {lr.coef_[0][0]:.4f}")
print(f"  DeBERTa weight: {lr.coef_[0][1]:.4f}")

# ── 3. MLP ────────────────────────────────────────────────────
start = time.time()
mlp = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    max_iter=500,
    random_state=SEED
)
mlp.fit(X_train_meta, y_train_meta)
mlp_time = time.time() - start
mlp_probs = mlp.predict_proba(X_val_meta)[:, 1]
mlp_preds = mlp.predict(X_val_meta)
res = evaluate_meta("MLP", y_val_meta, mlp_preds, mlp_probs)
res["train_time"] = round(mlp_time, 3)
results.append(res)
print(f"✓ MLP complete. ({mlp_time:.2f}s)")

# ── 4. XGBoost (GradientBoosting) ─────────────────────────────
start = time.time()
xgb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=SEED
)
xgb.fit(X_train_meta, y_train_meta)
xgb_time = time.time() - start
xgb_probs = xgb.predict_proba(X_val_meta)[:, 1]
xgb_preds = xgb.predict(X_val_meta)
res = evaluate_meta("GradientBoosting", y_val_meta, xgb_preds, xgb_probs)
res["train_time"] = round(xgb_time, 3)
results.append(res)
print(f"✓ GradientBoosting complete. ({xgb_time:.2f}s)")

# ── Sonuçlar tablosu ──────────────────────────────────────────
results_df = pd.DataFrame(results)
print(f"\n{'='*70}")
print("META-LEARNER COMPARISON V2")
print(f"{'='*70}")
print(results_df[["model", "accuracy", "f1", "precision",
                   "recall", "roc_auc"]].to_string(index=False))

models_list = [
    ("Weighted Average",    wa_preds,  wa_probs),
    ("Logistic Regression", lr_preds,  lr_probs),
    ("MLP",                 mlp_preds, mlp_probs),
    ("GradientBoosting",    xgb_preds, xgb_probs),
]

fig, axes = plt.subplots(1, 4, figsize=(20, 4))

for ax, (name, preds, probs) in zip(axes, models_list):
    cm = confusion_matrix(y_val_meta, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    ax.set_title(f"{name}\nAcc: {accuracy_score(y_val_meta, preds):.4f}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig(f"{DIRS['results']}/confusion_matrices_v2.png", dpi=150)
plt.show()
print("✓ Confusion matrices saved to Drive.")

plt.figure(figsize=(8, 6))

for name, preds, probs in models_list:
    fpr, tpr, _ = roc_curve(y_val_meta, probs)
    auc_score   = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc_score:.4f})")

plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — Meta-Learner Comparison V2")
plt.legend()
plt.tight_layout()
plt.savefig(f"{DIRS['results']}/roc_curves_v2.png", dpi=150)
plt.show()
print("✓ ROC curves saved to Drive.")
