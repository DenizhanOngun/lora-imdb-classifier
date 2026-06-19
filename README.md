# LoRA IMDB Classifier

Parameter-efficient stacking ensemble for binary sentiment classification on the IMDB dataset.

**96.18% accuracy** — combining LoRA-adapted RoBERTa and DeBERTa-v3 with TF-IDF baselines via 5-fold stacking, while updating less than 1% of model parameters.

[![Hugging Face Demo](https://img.shields.io/badge/🤗-Live%20Demo-yellow)](https://huggingface.co/spaces/Denizhaan/imdb-sentiment-demo)

---

## Results

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| TF-IDF + SVM | 89.57% | 89.53% | — |
| TF-IDF + LR | 89.43% | 89.40% | 96.12% |
| RoBERTa + LoRA | 95.60% | 95.62% | 99.00% |
| DeBERTa-v3 + LoRA | 95.92% | 95.95% | 99.14% |
| **LR Ensemble (best)** | **96.18%** | **96.19%** | 98.86% |

---

## Setup

```bash
git clone https://github.com/DenizhanOngun/lora-imdb-classifier.git
cd lora-imdb-classifier
pip install -r requirements.txt
```

By default, all data, checkpoints, and results are written to `./data/`.
Override with the `DATA_ROOT` environment variable:

```bash
export DATA_ROOT=/path/to/your/data
```

---

## Running the pipeline

Run scripts in order. Each script reads from and writes to `data/`.

```bash
# 1. Download and preprocess the IMDB dataset
python src/01_data_preprocessing.py

# 2. Train TF-IDF baselines and generate OOF predictions
python src/02_tfidf_baselines.py

# 3. Fine-tune RoBERTa + LoRA (r=16, ~42 min on L4 GPU)
python src/03_roberta_lora.py

# 4. Fine-tune DeBERTa-v3 + LoRA (r=32, ~49 min on A100 GPU)
python src/04_deberta_lora.py

# 5. Generate OOF predictions for transformer models (5-fold)
python src/05_oof_training.py

# 6. Train stacking ensemble meta-learners and produce final results
python src/06_stacking_ensemble.py
```

### Analysis scripts (optional)

```bash
# LoRA rank ablation for RoBERTa (r = 8 / 16 / 32)
python analysis/ablation_lora_rank.py

# DeBERTa hyperparameter grid search (rank × learning rate)
python analysis/ablation_deberta_hyperparams.py

# Error analysis of the best ensemble model
python analysis/error_analysis.py
```

---

## Project structure

```
lora-imdb-classifier/
├── config.py                          # paths, seeds, hyperparameters
├── utils.py                           # shared functions
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── 01_data_preprocessing.py       # HTML cleaning, head+tail truncation
│   ├── 02_tfidf_baselines.py          # TF-IDF + SVM/LR, OOF predictions
│   ├── 03_roberta_lora.py             # RoBERTa + LoRA training
│   ├── 04_deberta_lora.py             # DeBERTa-v3 + LoRA training
│   ├── 05_oof_training.py             # 5-fold OOF for transformer models
│   └── 06_stacking_ensemble.py        # meta-learner training and evaluation
│
└── analysis/
    ├── ablation_lora_rank.py          # RoBERTa rank ablation (Table 3)
    ├── ablation_deberta_hyperparams.py # DeBERTa grid search (Table 4)
    └── error_analysis.py              # misclassification analysis (Section 6)
```

**Model weights** are saved to `data/checkpoints/` during training and are not included in this repository.

---

## Key technical notes

**Head+Tail truncation:** Reviews exceeding 512 tokens retain the first 256 and last 256 tokens. This preserves both the opening statement and closing verdict, which carry the strongest sentiment signal.

**DeBERTa checkpoint saving:** When using PEFT with `task_type=SEQ_CLS`, the pooler and classifier weights are not saved with the LoRA adapter. Loading only the adapter causes near-random predictions. `04_deberta_lora.py` saves them separately as `extra_weights.pt`; `utils.load_deberta_model()` handles the correct reload.

**DeBERTa fp32:** DeBERTa-v3 requires fp32 precision due to gradient instability under fp16 (ELECTRA-style pretraining).

---

## Citation

```
Ongun, D., Buck, F. (2025). Parameter-Efficient Fine-Tuning Ensembles
for Sentiment Analysis. CENG463 Final Report, İzmir Institute of Technology.
```
