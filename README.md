# Classification of merger and non-interacting-galaxies

**Results**

| Model | Classification Accuracy | Area under ROC curve |
| ----- | ----------------------- | -------------------- |
| CNN | 94.72% | 0.94 |
| XGBoost + PCA | 86.75% | 0.85 |
| Transfer Learning (`ResNet18_2`) | 96.25% | 0.96 |
| Transfer Learning (`ResNet34`) | 94.75% | 0.95 |
| Transfer Learning (`ResNet18`) | 92.89% | 0.93 |
| Transfer Learning (`Xception`) | 79.19% | 0.77 |
| CNN Ensemble | 93.77% | 0.93 |
| XGBoost | 89.12% | 0.95 |

Some statistics of the best model (`Resnet18_2`) are <sup>1</sup>:

| Recall | Precision | F1 score |
| ------ | --------- | -------- |
| 0.966 | 0.9695 | 0.9677 |

[1] These values were calculated after a single experiment but could change slightly on a different pass.

**References**

Ackermann, S., Schawinski, K., Zhang, C., Weigel, A., & Turp, M. (2018). Using transfer learning to detect galaxy mergers. Monthly Notices Of The Royal Astronomical Society, 479(1), 415-425. doi: 10.1093/mnras/sty1398
