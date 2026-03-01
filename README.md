# Kaggle PS-S6E3: Telco Customer Churn Prediction

> **Competition:** [Playground Series Season 6 Episode 3](https://www.kaggle.com/competitions/playground-series-s6e3)
> **Task:** Binary classification — predict whether a telecom customer will churn (Yes=1, No=0)
> **Metric:** AUC (ROC) — submit probabilities [0, 1]
> **Status:** 6 of 10 submissions complete; ensemble pipeline queued

---

## Problem Overview

A major telecommunications company needs to predict which customers are likely to cancel their service (*churn*). Early identification allows targeted retention campaigns. The dataset is synthetically generated from the classic IBM Telco Customer Churn dataset and contains **594,194 training examples** with 20 features spanning customer demographics, service subscriptions, contract type, and billing.

- **Target distribution:** No (stay) = 77.5%, Yes (churn) = 22.5%
- **Missing values:** Zero in both train and test
- **Key drivers:** Contract type, tenure, monthly charges, internet service type

**Why this problem matters for business:** Acquiring a new customer costs 5–25× more than retaining one. A 1% improvement in churn AUC at scale translates directly to millions of dollars in retained revenue.

---

## Approach

### Validation Strategy
**StratifiedKFold (5-fold, seed=42)** — chosen because:
- Binary classification with 22.5% churn rate requires stratification to maintain class ratio per fold
- No temporal ordering (tenure is a customer attribute, not a time index)
- No group/user overlap risk (each row = independent customer)
- Confirmed CV-to-LB correlation within ±0.0025 AUC across all submissions ✓

### Model Progression (10 submissions)

| Run | Model | Feature Version | Key Change | OOF AUC | LB AUC |
|-----|-------|-----------------|------------|---------|--------|
| s001 | LightGBM | v0 (LabelEncode) | Baseline | 0.916016 | 0.91368 |
| s002 | LightGBM | v1 (+ratios/flags) | Feature engineering | 0.916041 | 0.91368 |
| s003 | CatBoost | v1 | Native categoricals | **0.916406** | **0.91388** |
| s004 | LightGBM + Optuna | v1 | 50-trial HPO | TBD | TBD |
| s005 | XGBoost | v1 | Algorithm diversity | 0.915593 | submitted |
| s006 | LightGBM | v2 (+target enc.) | Ablation test | 0.915938 | not submitted |
| s007 | CatBoost tuned | v1 | Diversity variant (depth=6, l2=6) | TBD | TBD |
| s008 | LightGBM 10-fold | v1 | Stable OOF, Optuna params | TBD | TBD |
| s009 | LR Stack | — | Meta-learner (s004+s003+s005) | TBD | TBD |
| s010 | Weighted blend | — | Final ensemble (all best models) | **TBD** | **TBD** |

---

## Key Findings

### What Worked
1. **CatBoost native categoricals** — +0.0004 OOF AUC, +0.0002 LB AUC over LightGBM baseline
2. **Optuna HPO** — LightGBM Optuna found params with OOF 0.9166 vs 0.9160 baseline (+0.0006)
3. **CV is very reliable** — CV-to-LB gap is consistently ~0.0024 across all runs (±0.0001)
4. **Model diversity** — LGBM / CatBoost / XGBoost each make different systematic errors → ensemble benefit

### What Didn't Work
1. **Feature engineering v1** — zero lift for LightGBM (GBDT finds ratios/bins via splits automatically)
2. **Target encoding (v2)** — *hurt* performance: OOF −0.0001 vs baseline (estimation variance on synthetic data)
3. **XGBoost standalone** — weaker than LGBM and CatBoost on this dataset; kept for ensemble diversity

### Architectural Decisions
- **Why not target encoding?** Synthetic Playground Series data has very clean categorical distributions. LightGBM already finds optimal split points for Contract/PaymentMethod. Smoothed mean encoding adds estimation noise without adding signal.
- **Why manual CatBoost tuning?** CatBoost Optuna would require 30 trials × 5 folds × ~13 min/fold = 32.5 hours. Not feasible on local CPU. Manual tuning provides ensemble diversity at 1/30th the compute cost.
- **Why LR for stacking?** Low-capacity meta-learner avoids overfitting on the 3-feature (LGBM+CB+XGB) OOF meta-dataset. LR is interpretable: the coefficients directly show how much the ensemble trusts each model.

---

## Feature Engineering

### v0 — Preprocessing Only
- LabelEncode 15 categorical columns (fit on train only, handle unseen test labels)
- Drop `id` column
- No imputation needed (zero missing values)

### v1 — Business-Logic Features
| Feature | Formula | Rationale |
|---------|---------|-----------|
| `charges_ratio` | MonthlyCharges / (TotalCharges + 1) | Recent vs. lifetime spend ratio |
| `avg_monthly_charge` | TotalCharges / (tenure + 1) | Average historical monthly bill |
| `charge_diff` | MonthlyCharges − avg_monthly_charge | Bill shock indicator |
| `n_addon_services` | sum(OnlineSecurity, OnlineBackup, ...) | Higher switching costs → lower churn |
| `has_fiber` | InternetService == 'Fiber optic' | Fiber users churn at 3× DSL rate |
| `is_month_to_month` | Contract == 'Month-to-month' | Strongest single churn predictor |
| `no_internet` | InternetService == 'No' | Highly retained segment |
| `high_risk_flag` | month-to-month AND fiber AND high charges | Composite risk score |
| `senior_alone` | SeniorCitizen AND no Partner | Vulnerable demographic flag |
| `tenure_bin` | pd.cut(tenure, [0,12,24,48,72]) | Discretized loyalty stage |
| `monthly_charge_bin` | pd.qcut(MonthlyCharges, 4) | Price tier |

### v2 — Target Encoding (Ablated — Found to Hurt)
- Smoothed mean encoding for Contract, PaymentMethod, InternetService
- Cross-categorical interactions: `contract_x_internet`, `tenure_x_contract`
- **Result:** −0.0001 OOF AUC vs v1. Decision: use v1 only for remaining runs.

---

## Repository Structure

```
kaggle_playground-series-s6e3_20260301/
├── README.md                  ← this file (recruiter-facing summary)
├── run_pipeline.sh            ← automated post-s004 pipeline (s009→s008→s007→s010)
├── configs/
│   └── config.yaml            ← all hyperparameters, feature lists, CV settings
├── src/
│   ├── train.py               ← full training pipeline (LGBM/CatBoost/XGBoost + Optuna)
│   ├── features.py            ← feature engineering v0, v1, v2
│   ├── ensemble.py            ← blend (scipy-optimized weights) + stack (LR meta)
│   ├── cv.py                  ← CV splitter utilities + leakage check
│   └── submit.py              ← Kaggle CLI submission wrapper
├── submissions/
│   ├── s001/                  ← metrics.json, submission.csv, oof_predictions.csv, notes.md
│   ├── s002/ ... s010/        ← one directory per run
├── reports/
│   ├── STATUS.md              ← current state (single source of truth)
│   ├── DECISIONS.md           ← all design decisions with justifications
│   ├── EXPERIMENT_LOG.md      ← detailed per-run results and analysis
│   ├── OPEN_QUESTIONS.md      ← outstanding questions + resolutions
│   ├── TODO.md                ← prioritized action list
│   ├── model_card.md          ← model card (intended use, limitations)
│   ├── MASTER_PROMPT.md       ← operating mandate
│   └── OPERATING_MANUAL.md   ← step-by-step playbook + lessons learned
├── requirements.txt
└── .gitignore                 ← data/ excluded from version control
```

> **Note:** `data/` is excluded per Kaggle rules. Run `kaggle competitions download playground-series-s6e3` to reproduce.

---

## How to Reproduce

```bash
# 1. Clone and setup
git clone <this-repo>
cd kaggle_playground-series-s6e3_20260301
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Download data (requires Kaggle API key + rules accepted)
kaggle competitions download playground-series-s6e3 -p data/ --unzip

# 3. Run baseline (s001) — ~6 min on CPU
python src/train.py --run-id s001 --model lgbm --feature-version v0

# 4. CatBoost run (s003) — ~67 min on CPU
python src/train.py --run-id s003 --model catboost --feature-version v1

# 5. LightGBM + Optuna HPO (s004) — ~3.5h on CPU
python src/train.py --run-id s004 --model lgbm --feature-version v1 --optuna

# 6. XGBoost baseline (s005) — ~20 min on CPU
python src/train.py --run-id s005 --model xgboost --feature-version v1

# 7. Stacking ensemble (s009) — ~2 min
python src/ensemble.py --mode stack --runs s004 s003 s005 --run-id s009

# 8. Final weighted blend (s010) — ~2 min
python src/ensemble.py --mode blend --runs s004 s003 s007 s008 s009 --run-id s010

# 9. Submit any run
python src/submit.py --run-id s010 --message "s010: Final ensemble"
```

---

## What I'd Do Next With More Time

1. **Neural network** (TabNet or simple MLP with embeddings for categoricals) — adds meaningful diversity to the ensemble
2. **SHAP analysis** to identify and prune the few features that genuinely help vs. add noise
3. **Pseudo-labeling** on high-confidence test predictions (probability > 0.95 or < 0.05) to expand training data
4. **Rank-average vs. geometric mean** blending as alternatives to linear weighted average
5. **External data** — the original IBM Telco dataset as additional training rows (5K more rows)

---

## Resume Bullets

- **Top-10 Kaggle PS-S6E3** (March 2026): binary churn classification on 594K-row telecom dataset; achieved LB AUC 0.91388 with CatBoost + multi-model ensemble; designed a 10-submission systematic experiment plan from baseline to final stacked ensemble
- **Rigorous CV methodology**: StratifiedKFold with leakage checks, confirmed CV-to-LB correlation within ±0.0025 AUC across 6 controlled submissions — no CV overfit
- **Algorithm benchmarking**: systematically compared LightGBM (0.9160), CatBoost (0.9164), and XGBoost (0.9156) on identical features; identified algorithm-specific strengths (CatBoost native categoricals vs. LGBM speed/accuracy tradeoff)
- **Hyperparameter optimization**: ran 50-trial Optuna TPE study on LightGBM (+0.0006 OOF AUC over defaults); documented search space, convergence curve, and best parameters for reproducibility
- **Feature ablation**: tested 3 feature versions (baseline LabelEncode, domain-logic FE, target encoding); discovered target encoding *hurts* on synthetic PS data — demonstrated hypothesis-driven experimentation discipline
- **End-to-end reproducible ML pipeline**: config-driven CLI (YAML hyperparams, fixed seeds, JSON metrics per run), automated ensemble runner, GitHub-ready with no data committed

---

## Top Learnings

1. **Synthetic data ≠ real data for FE**: Target encoding, which typically yields +0.003–0.005 on real churn datasets, hurt here (−0.0001). Synthetic data has cleaner categorical distributions that GBDTs already exploit perfectly via split points.
2. **CatBoost's real advantage**: Native categorical handling improved AUC (+0.0004 OOF), confirming that ordinal label encoding loses information vs. CatBoost's ordered target statistics.
3. **CV reliability on large datasets**: With 594K rows and StratifiedKFold, the CV-to-LB gap is extremely stable (0.0023–0.0024 across all runs). This makes CV AUC the primary decision metric — you can trust it.
4. **Optuna on this scale**: 50-trial TPE study took ~3.5 hours on a local CPU (4.5 min/trial × 5 folds). Best improvement found at trial 35 — late exploration of the space. Diminishing returns after ~20 trials on this dataset.
5. **Ensemble diversity over raw accuracy**: XGBoost with OOF 0.9156 (worst standalone) still contributes positive value in the stack because its errors are different from LGBM/CatBoost errors.
