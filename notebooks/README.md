# Development Notebooks

These are the original Google Colab notebooks used during development.
The cleaned, production-ready Python scripts are in `src/` and `analysis/`.

| Notebook | Corresponds to | Notes |
|---|---|---|
| `00_colab_setup.ipynb` | — | Google Drive mount and GitHub push helpers. Colab-specific, not part of the main pipeline. |
| `01b_data_preprocessing_v2.ipynb` | `src/01_data_preprocessing.py` | V2: 256/256 head+tail truncation (final version). |
| `03b_roberta_lora_v2.ipynb` | `src/03_roberta_lora.py` | RoBERTa + LoRA, r=16. Achieves 95.60% accuracy. |
| `04d_deberta_lora_v2_best.ipynb` | `src/04_deberta_lora.py` | DeBERTa-v3 + LoRA, r=32, lr=5e-5. Achieves 95.92% accuracy. |
| `05d_oof_training_v2_best.ipynb` | `src/05_oof_training.py` | 5-fold OOF training for DeBERTa best config. |
| `06d_meta_learner_best.ipynb` | — | Intermediate experiment: meta-learner with 2 base models (RoBERTa + DeBERTa only). Superseded by `06e`. |
| `06e_meta_learner_three_models.ipynb` | `src/02_tfidf_baselines.py` + `src/06_stacking_ensemble.py` | Final pipeline: TF-IDF OOF training + 4-model stacking ensemble. Achieves 96.18% accuracy. |
| `ablation_lora_rank.ipynb` | `analysis/ablation_lora_rank.py` | RoBERTa LoRA rank ablation: r ∈ {8, 16, 32}. |
| `ablation_deberta_hyperparams.ipynb` | `analysis/ablation_deberta_hyperparams.py` | DeBERTa grid search: rank × learning rate. |
| `error_analysis.ipynb` | `analysis/error_analysis.py` | Misclassification analysis of the final LR ensemble. |

## How to run

Open any notebook in Google Colab. Each notebook starts with:

```python
from google.colab import drive
drive.mount('/content/drive')
exec(open("/content/drive/MyDrive/imdb_peft_project/code/lora-imdb-classifier/00_colab_setup.py").read())
```

This mounts Google Drive and sets up the `DIRS` path dictionary used throughout.
Model checkpoints and prediction files are stored in Google Drive, not in this repository.
