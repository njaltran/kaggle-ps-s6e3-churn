# DECISIONS.md — Competition, CV, Model Choices

> Maintained by: Competition Scout + Validation Guardian + Model Engineer
> Last updated: 2026-03-01

---

## Competition Selection

### Chosen: `playground-series-s6e3` (Kaggle Playground Series Season 6 Episode 3)

**Selection rationale (Competition Scout):**

| Factor | Assessment |
|--------|-----------|
| Recruiter signal | High — tabular ML is the most hirable skill; Kaggle PS series is widely recognized |
| Recognizable task | Telco Customer Churn — every recruiter knows this problem |
| Leaderboard opportunity | Competition launched 2026-02-24 with effectively 0 public teams; early entry advantage |
| Compute needs | CPU only — LightGBM/CatBoost on 594K rows fits in <8GB RAM |
| Metric clarity | AUC (ROC) — standard, interpretable, no ambiguity |
| Deadline | 2026-03-31 — 30 days remaining |
| Data quality | Zero missing values, clean CSV format, standard sklearn-compatible |

**Rejected alternatives:**
- `march-machine-learning-mania-2026`: Domain-specific (NCAA basketball), requires sports domain knowledge, deadline 2026-03-19 (18 days), niche appeal
- `stanford-rna-3d-folding-2`: Specialized bioinformatics, high compute (GPU), niche recruiter signal
- `hull-tactical-market-prediction`: Custom evaluation framework (gRPC relay), time-series with 3677 teams, much more competitive
- `ai-mathematical-olympiad-progress-prize-3`: LLM reasoning task, not traditional ML portfolio

---

## Dataset Analysis (Deep Research — 2026-03-01)

**Task:** Binary classification — predict customer churn (Yes=1, No=0)
**Metric:** AUC (ROC) — inferred from leaderboard baseline 0.90504
**Train:** 594,194 rows × 21 columns (incl. target)
**Test:** 254,655 rows × 20 columns
**Origin:** Synthetically generated from the classic IBM Telco Customer Churn dataset

### Features

| Column | Type | Notes |
|--------|------|-------|
| id | int | Drop — not predictive |
| gender | str (binary) | Male/Female |
| SeniorCitizen | int (0/1) | Binary |
| Partner | str (Yes/No) | |
| Dependents | str (Yes/No) | |
| tenure | int | 1–72 months |
| PhoneService | str (Yes/No) | |
| MultipleLines | str (Yes/No/No phone) | |
| InternetService | str (DSL/Fiber optic/No) | High importance expected |
| OnlineSecurity | str (Yes/No/No internet) | |
| OnlineBackup | str (Yes/No/No internet) | |
| DeviceProtection | str (Yes/No/No internet) | |
| TechSupport | str (Yes/No/No internet) | |
| StreamingTV | str (Yes/No/No internet) | |
| StreamingMovies | str (Yes/No/No internet) | |
| Contract | str (Month-to-month/One year/Two year) | Highest importance expected |
| PaperlessBilling | str (Yes/No) | |
| PaymentMethod | str (4 values) | |
| MonthlyCharges | float | 18.25–118.75 |
| TotalCharges | float | ≈ tenure × MonthlyCharges |
| **Churn** | str (Yes/No) | **TARGET** → encode Yes=1, No=0 |

**Target distribution:** No=460,377 (77.5%) / Yes=133,817 (22.5%)
**Missing values:** None in train or test ✓

### Leakage Analysis (Validation Guardian)

1. **TotalCharges ≈ tenure × MonthlyCharges** — This is a near-deterministic relationship within the data. NOT external leakage; it's internal correlation. We can exploit this for features but must not let it inflate CV (it exists identically in both train and test).
2. **id column** — Sequential integers, no temporal signal. Safe to drop.
3. **No target leakage identified** — All features are customer attributes known at prediction time.
4. **No group leakage risk** — Each row is an independent customer. StratifiedKFold is safe.

---

## Cross-Validation Strategy

**Decision:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`

**Justification (Validation Guardian):**
- Binary classification with 22.5% positive rate → stratification ensures each fold has the same class ratio
- No temporal ordering in data (tenure is a customer attribute, not a time index)
- No customer grouping structure (each row = independent customer)
- 5 folds balances variance in OOF estimate vs. training time (each fold has ~119K train rows)
- GroupKFold not needed (no user/group overlap risk)
- TimeSeriesSplit not applicable (no time ordering)

---

## Model Strategy

### Run Plan (10 submissions)

| Run | Model | Feature Version | Goal | OOF AUC | LB AUC |
|-----|-------|-----------------|------|---------|--------|
| s001 | LightGBM baseline | v0 | Establish valid LB entry | 0.916016 | 0.91368 |
| s002 | LightGBM + FE v1 | v1 | Measure feature lift | 0.916041 | 0.91368 |
| s003 | CatBoost baseline | v1 | Compare model family | 0.916406 | **0.91388** |
| s004 | LightGBM + Optuna HPO | v1 | Hyperparameter optimization (50 trials) | TBD | TBD |
| s005 | XGBoost baseline | v1 | Ensemble diversity | 0.915593 | submitted |
| s006 | LightGBM + FE v2 | v2 | Ablation: target encoding | 0.915938 | NOT submitted |
| s007 | CatBoost manual tuning | v1 | Diversity variant for ensemble | TBD | TBD |
| s008 | LightGBM 10-fold tuned | v1 | Stable OOF, best Optuna params | TBD | TBD |
| s009 | LR Stack (s004+s003+s005) | — | Meta-learner ensemble | TBD | TBD |
| s010 | Weighted blend (all best) | — | Final kitchen-sink ensemble | TBD | TBD |

### Decision: CatBoost Optuna (s007) → Manual Tuning
**Date:** 2026-03-01
**Original plan:** Run Optuna 30-trial CatBoost HPO
**Issue found:** CatBoost inner CV = 5 folds × ~13 min/fold = 65 min/trial × 30 trials = **32.5 hours** — completely infeasible on local CPU
**Resolution:** Hand-tune CatBoost with diversity-oriented params vs s003:
- depth: 7 → 6 (different tree structure, different splits)
- l2_leaf_reg: 3.0 → 6.0 (more aggressive regularization)
- bagging_temperature: 0.5 → 0.8 (higher randomness → different errors)
- learning_rate: 0.05 → 0.04 (slightly slower)
**Rationale:** For ensemble, model DIVERSITY matters as much as raw accuracy. A different CatBoost configuration makes systematically different errors, which benefits blending/stacking.
**Added:** `catboost_tuned` section to config.yaml; `run_cv()` updated to use it when `--use-tuned-params`.

### Why LightGBM first?
- Fastest iteration (sub-2min on 594K rows with 5-fold CV locally)
- Strong default performance on tabular data
- Feature importance is interpretable for recruiter-facing writeup
- Industry standard: every data science job uses GBDT

### Why CatBoost in s003?
- Handles categorical features natively (no LabelEncoding needed)
- Often outperforms LGBM on datasets with many categoricals
- Provides symmetric trees (more regularized, better generalization)
- Adds diversity for ensembling

### Feature Engineering Plan (s002)
1. `charges_ratio` = MonthlyCharges / (TotalCharges + 1)
2. `avg_monthly_charge` = TotalCharges / (tenure + 1)
3. `tenure_bin` = pd.cut(tenure, bins=[0,12,24,48,72], labels=[0,1,2,3])
4. `monthly_charge_bin` = pd.qcut(MonthlyCharges, q=4, labels=[0,1,2,3])
5. `n_services` = count of Yes across all service columns (OnlineSecurity, OnlineBackup, etc.)
6. `has_internet` = InternetService != 'No'
7. `is_month_to_month` = Contract == 'Month-to-month'
8. `high_value_churner_risk` = is_month_to_month AND has_internet AND MonthlyCharges > median

---

## Metric Alignment

- **LB metric:** AUC (ROC) — submit probabilities (floats 0–1), NOT binary labels
- **CV metric:** roc_auc_score from sklearn — identical computation
- **Baseline LB score:** 0.90504 (competition organizer's baseline)
- **Target:** Beat 0.90504, aim for top 10% (target AUC ≥ 0.92)
