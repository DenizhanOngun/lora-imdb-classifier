# =============================================================
# 01_data_preprocessing.py
# =============================================================
# Downloads the IMDB dataset, applies HTML cleaning and
# head+tail truncation analysis, and saves train/test DataFrames
# as parquet files for use by all subsequent scripts.
#
# Output:
#   data/train_df_v2.parquet
#   data/test_df_v2.parquet
#
# Run:
#   python src/01_data_preprocessing.py
# =============================================================

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from transformers import RobertaTokenizer, DebertaV2Tokenizer

from config import DATA_DIR, RESULTS_DIR, SEED, MAX_LEN, HEAD_LEN
from utils import head_tail_truncate, IMDBDataset

np.random.seed(SEED)
torch.manual_seed(SEED)


# ------------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------------
print("Loading IMDB dataset...")
dataset  = load_dataset("stanfordnlp/imdb")
train_df = pd.DataFrame(dataset["train"])
test_df  = pd.DataFrame(dataset["test"])

print(f"Train : {len(train_df)} samples")
print(f"Test  : {len(test_df)} samples")
print(f"\nLabel distribution (train):\n{train_df['label'].value_counts()}")


# ------------------------------------------------------------------
# 2. Text cleaning
# ------------------------------------------------------------------
def clean_text(text: str, lowercase: bool = False) -> str:
    """
    IMDB-specific text cleaning.

    lowercase=False  for transformer models — capitalization carries
                     sentiment signal (TERRIBLE != terrible).
    lowercase=True   for TF-IDF + classical models.
    """
    text = re.sub(r"<[^>]+>", " ", text)   # remove HTML tags
    text = re.sub(r"\s+", " ", text)        # normalize whitespace
    text = text.strip()
    if lowercase:
        text = text.lower()
    return text


train_df["text_clean"]       = train_df["text"].apply(clean_text)
test_df["text_clean"]        = test_df["text"].apply(clean_text)
train_df["text_clean_lower"] = train_df["text"].apply(lambda x: clean_text(x, lowercase=True))
test_df["text_clean_lower"]  = test_df["text"].apply(lambda x: clean_text(x, lowercase=True))

print("\nText cleaning complete.")


# ------------------------------------------------------------------
# 3. Review length analysis
# ------------------------------------------------------------------
train_df["word_count"] = train_df["text"].apply(lambda x: len(x.split()))

print(f"\nAverage words : {train_df['word_count'].mean():.0f}")
print(f"Median words  : {train_df['word_count'].median():.0f}")
print(f"Max words     : {train_df['word_count'].max()}")
print(f"Over 512      : {(train_df['word_count'] > 512).sum()} "
      f"({(train_df['word_count'] > 512).mean()*100:.1f}%)")

os.makedirs(RESULTS_DIR, exist_ok=True)
plt.figure(figsize=(10, 4))
plt.hist(train_df["word_count"], bins=50, color="steelblue", edgecolor="white")
plt.axvline(512, color="red", linestyle="--", label="512 token limit")
plt.xlabel("Word count")
plt.ylabel("Frequency")
plt.title("IMDB Review Length Distribution")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "length_distribution.png"), dpi=150)
plt.close()
print(f"Length distribution plot saved to {RESULTS_DIR}/length_distribution.png")


# ------------------------------------------------------------------
# 4. Verify tokenization (quick sanity check, no full tokenization)
# ------------------------------------------------------------------
print("\nLoading tokenizers for truncation verification...")
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
deberta_tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")

sample_df = train_df.head(500).copy()
sample_df["token_count"] = sample_df["text_clean"].apply(
    lambda x: len(
        roberta_tokenizer(x, add_special_tokens=False, truncation=False)["input_ids"]
    )
)
truncated = (sample_df["token_count"] > MAX_LEN).sum()
print(f"Samples requiring truncation (first 500): {truncated} / 500 ({truncated/5:.1f}%)")
print(f"Head+Tail strategy: {HEAD_LEN} head + {MAX_LEN - HEAD_LEN} tail tokens")


# ------------------------------------------------------------------
# 5. Save preprocessed DataFrames
# ------------------------------------------------------------------
os.makedirs(DATA_DIR, exist_ok=True)
train_df.to_parquet(os.path.join(DATA_DIR, "train_df_v2.parquet"))
test_df.to_parquet(os.path.join(DATA_DIR, "test_df_v2.parquet"))

print(f"\nSaved:")
print(f"  {DATA_DIR}/train_df_v2.parquet")
print(f"  {DATA_DIR}/test_df_v2.parquet")
print("\nPreprocessing complete.")
