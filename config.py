# =============================================================
# config.py — Central configuration for lora-imdb-classifier
# =============================================================
# All paths, seeds, and hyperparameters are defined here.
# Every script imports from this file instead of hardcoding values.
#
# Usage:
#   from config import DATA_DIR, MODEL_DIR, ROBERTA_CONFIG, ...
# =============================================================

import os

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
# Root directory for all project data (checkpoints, predictions, results).
# Override with the DATA_ROOT environment variable if needed.
DATA_ROOT = os.environ.get("DATA_ROOT", "./data")

DATA_DIR        = DATA_ROOT                          # preprocessed parquet files
CHECKPOINT_DIR  = os.path.join(DATA_ROOT, "checkpoints")
OOF_DIR         = os.path.join(DATA_ROOT, "oof_predictions")
RESULTS_DIR     = os.path.join(DATA_ROOT, "results")

ROBERTA_CKPT    = os.path.join(CHECKPOINT_DIR, "roberta_lora", "final")
DEBERTA_CKPT    = os.path.join(CHECKPOINT_DIR, "deberta_lora", "final")
DEBERTA_EPOCH3  = os.path.join(CHECKPOINT_DIR, "deberta_lora", "epoch_3")

TRAIN_PARQUET   = os.path.join(DATA_DIR, "train_df_v2.parquet")
TEST_PARQUET    = os.path.join(DATA_DIR, "test_df_v2.parquet")

# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------
SEED = 42

# ------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------
MAX_LEN  = 512
HEAD_LEN = 256   # head+tail truncation: first 256 + last 256 tokens

# ------------------------------------------------------------------
# RoBERTa + LoRA
# ------------------------------------------------------------------
ROBERTA_CONFIG = {
    "base_model"       : "roberta-base",
    "lora_r"           : 16,
    "lora_alpha"       : 32,
    "lora_dropout"     : 0.1,
    "target_modules"   : ["query", "value"],
    "learning_rate"    : 2e-4,
    "batch_size"       : 16,
    "grad_accum_steps" : 1,
    "warmup_steps"     : 200,
    "weight_decay"     : 0.0,
    "epochs"           : 3,
    "fp16"             : True,
    "oof_epochs"       : 2,
}

# ------------------------------------------------------------------
# DeBERTa-v3 + LoRA  (best config: r=32, lr=5e-5)
# ------------------------------------------------------------------
DEBERTA_CONFIG = {
    "base_model"       : "microsoft/deberta-v3-base",
    "lora_r"           : 32,
    "lora_alpha"       : 64,
    "lora_dropout"     : 0.1,
    "target_modules"   : ["query_proj", "value_proj"],
    "learning_rate"    : 5e-5,
    "batch_size"       : 8,
    "grad_accum_steps" : 2,   # effective batch size = 16
    "warmup_steps"     : 500,
    "weight_decay"     : 0.01,
    "epochs"           : 3,
    "fp16"             : False,  # fp32 required — ELECTRA-style pretraining
    "oof_epochs"       : 2,
}

# ------------------------------------------------------------------
# TF-IDF
# ------------------------------------------------------------------
TFIDF_CONFIG = {
    "max_features" : 50000,
    "ngram_range"  : (1, 2),
    "min_df"       : 2,
}

# ------------------------------------------------------------------
# Stacking ensemble
# ------------------------------------------------------------------
N_FOLDS = 5
