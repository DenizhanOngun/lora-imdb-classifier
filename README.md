# Parameter-Efficient Fine-Tuning Ensembles for IMDB Sentiment Analysis

**Team:** Denizhan, Mohammed, Friedrich | **Supervisor:** Prof. Dr. Onan

**Demo:** https://huggingface.co/spaces/Denizhaan/imdb-sentiment-demo

---

## Final Results

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| TF-IDF + SVM | 0.8957 | 0.8953 | — |
| RoBERTa + LoRA V2 | 0.9560 | 0.9562 | 0.9900 |
| DeBERTa + LoRA V2 | 0.9536 | 0.9540 | 0.9890 |
| **Meta-Learner MLP** | **0.9584** | **0.9584** | **0.9908** |

## LoRA Efficiency

| Model | Trainable | Total | % |
|---|---|---|---|
| RoBERTa + LoRA | 1,181,954 | 125,829,124 | 0.94% |
| DeBERTa + LoRA | 591,362 | 185,015,044 | 0.32% |

## Research Questions

- **RQ1:** PEFT stacking > single models? Yes (0.9584 > 0.9560)
- **RQ2:** Best meta-learner? MLP (32-16)
- **RQ3:** Efficiency trade-off? +6.3% accuracy with <1% parameters

## References

- Hu et al. (2022). LoRA. ICLR 2022.
- He et al. (2021). DeBERTaV3. arXiv:2111.09543.
- Wang et al. (2023). LoRA ensembles. arXiv:2310.00035.
