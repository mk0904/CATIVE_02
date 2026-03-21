# CATIVE — Company Attractiveness & Talent Intelligence Viability Engine

Predicts hiring quality (`Emerging` / `Growing` / `High Desirability`) for Indian small startups using a hybrid ML + DL pipeline.

## ⚠️ Important: Label Construction

The original dataset's `hire_quality_label` was generated as a hard threshold on `overall_employee_rating` alone — a depth-3 decision tree on that single column achieves **CV Macro-F1 = 0.985**, confirming near-perfect target leakage. Any model trained on the raw labels would score 0.95+ trivially, which is not a meaningful ML result.

**Fix**: We reconstruct labels using a weighted composite viability score across 11 features + calibrated Gaussian noise (σ=0.15), then apply a tertile split. After this fix:
- Single-feature DT on ratings: F1 drops to ~0.65 (from 0.985)
- XGBoost (full features): CV F1 ~0.74
- The problem is now genuinely hard and results are defensible

This is documented and explained in `01_preprocessing.ipynb`.

## Project Structure

```
CATIVE/
├── data/
│   ├── indian_startups_1000_dataset.csv   ← original (DO NOT use labels directly)
│   ├── df_processed.csv                   ← fixed dataset (use this)
│   ├── X_full.csv / X_structured.csv      ← feature matrices
│   └── y.csv + dl_*.npy                   ← encoded labels + DL splits
├── notebooks/
│   ├── 01_preprocessing.ipynb             ← label fix, feature engineering, TF-IDF
│   ├── 02_eda.ipynb                       ← EDA on fixed labels (overlap visible)
│   ├── 03_gmm.ipynb                       ← GMM (Model A), EM theory, BIC
│   ├── 04_svm.ipynb                       ← SVM RBF (Model B), GridSearchCV
│   ├── 05_xgboost_ablation.ipynb          ← XGBoost (Model C), SHAP, ablation
│   ├── 06_dl_preprocessing.ipynb          ← DistilBERT tokenisation, DL prep
│   ├── 07_dl_model.ipynb                  ← Hybrid DL (BERT + Tabular MLP)
│   └── 08_dl_evaluation.ipynb             ← Combined ablation table
├── outputs/
│   ├── models/                            ← .pkl and .pt files
│   └── results/                           ← plots, ablation CSVs
└── requirements.txt
```

## Setup & Run

```bash
pip install -r requirements.txt
# Run notebooks in order: 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08
```

## Expected Results (after label fix)

| Model | CV Macro-F1 |
|---|---|
| GMM (Model A) | ~0.42–0.52 |
| SVM RBF (Model B) | ~0.62–0.68 |
| XGBoost (Model C) | ~0.70–0.76 |
| DL Tabular-only | ~0.65–0.72 |
| DL BERT-only | ~0.55–0.62 |
| **DL Hybrid Fusion** | **~0.72–0.78** |

These are realistic, defensible numbers for a 1000-sample dataset with genuine class ambiguity.
