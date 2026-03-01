# Model Card — PS-S6E3 Churn Prediction

## Model Details
- **Competition:** Kaggle Playground Series Season 6 Episode 3
- **Task:** Binary classification (predict customer churn: Yes=1, No=0)
- **Metric:** AUC (ROC) — submit probabilities [0, 1]
- **Primary models:** LightGBM (Optuna-tuned), CatBoost (baseline + diversity variant)
- **Secondary:** XGBoost (for ensemble diversity)
- **Final prediction:** LR-stacked ensemble → weighted blend (scipy-optimized)
- **Framework:** LightGBM 4.3+, CatBoost, XGBoost, scikit-learn 1.4+, Optuna 3.x

## Intended Use
- Portfolio demonstration of end-to-end tabular ML workflow
- Kaggle public leaderboard competition (PS-S6E3)
- NOT intended for production deployment without further validation, recalibration, and fairness audits

## Data
- **Source:** Synthetic dataset generated from IBM Telco Customer Churn (via Kaggle PS-S6E3)
- **Train:** 594,194 rows × 20 features
- **Test:** 254,655 rows × 20 features
- **Features:** Customer demographics (gender, SeniorCitizen, Partner, Dependents), service subscriptions (Phone, Internet, Online Security, etc.), contract type, billing method, charges
- **Target:** Churn (Yes=1 / No=0), 22.5% positive rate
- **Missing values:** Zero in both train and test

## Training

### Cross-Validation
- **Strategy:** StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
- **Rationale:** Binary classification with class imbalance (22.5% positive) requires stratification; no temporal or group structure justifies other strategies
- **CV-to-LB gap:** Consistently 0.0023–0.0025 AUC across 6 submissions (very reliable)

### Models Trained
| Run | Model | Params | OOF AUC |
|-----|-------|--------|---------|
| s001 | LightGBM | num_leaves=127, lr=0.05, defaults | 0.916016 |
| s002 | LightGBM + FE v1 | same | 0.916041 |
| s003 | CatBoost | depth=7, l2=3, lr=0.05 | 0.916406 |
| s004 | LightGBM + Optuna | 50-trial TPE (best params TBD) | TBD |
| s005 | XGBoost | max_depth=7, subsample=0.8, lr=0.05 | 0.915593 |
| s007 | CatBoost tuned | depth=6, l2=6, lr=0.04 | TBD |
| s008 | LightGBM 10-fold | Optuna best params | TBD |
| s009 | LR Stack | C=1.0, trained on s004+s003+s005 OOF | TBD |
| s010 | Weighted Blend | scipy-optimized weights on OOF | TBD |

### Seeds
- All CV splits: `random_state=42`
- All model training: `seed=42` (or equivalent per framework)
- Optuna: `TPESampler(seed=42)`
- Result: fully reproducible pipeline

## Leakage Safeguards
1. **OOF discipline:** All feature fitting (LabelEncoders) fit on training fold only; applied to val/test
2. **No target leakage:** All features are customer attributes observable before churn event
3. **ID column dropped:** Not predictive; zero id overlap between train and test confirmed
4. **Target encoding safety (v2 ablation):** Smoothed mean encoding computed inside CV folds; BUT found to hurt performance — dropped for all production runs
5. **Ensemble OOF:** Meta-learner (LR stack) trained on OOF predictions only — no test data leakage into meta-training

## Ablations Run
| Ablation | Finding |
|----------|---------|
| FE v0 → v1 (ratios, flags, counts) | No lift for LightGBM; GBDT captures these via splits |
| FE v1 → v2 (target encoding + interactions) | −0.0001 OOF AUC; estimation variance on synthetic data |
| LightGBM → CatBoost | +0.0004 OOF; native categorical handling adds value |
| LightGBM → XGBoost | −0.0004 OOF; XGBoost weaker on this dataset |
| LightGBM default → Optuna HPO | +0.0006 OOF (50 trials); meaningful but diminishing after ~20 |

## Limitations
- Trained on **synthetic** data; may not generalize to real telecom churn distributions
- **No temporal validation** (data has no time index; can't simulate deployment drift)
- **No fairness audit:** Gender, SeniorCitizen are features; model may encode demographic disparities
- **Probability calibration:** Raw probabilities used for ranking (AUC); not calibrated for decision thresholds
- **No adversarial testing:** Unknown robustness to distribution shift or deliberate manipulation

## Evaluation
| Metric | Run | Value |
|--------|-----|-------|
| Best OOF AUC | s003 (interim) | 0.916406 |
| Best LB AUC | s003 | 0.91388 |
| Best LB Rank | s003 | ~10 / ~20 teams (interim) |
| Final OOF AUC | s010 | TBD |
| Final LB AUC | s010 | TBD |
| Final LB Rank | s010 | TBD |

## Compute
- **Hardware:** Local CPU (Apple M-series)
- **Per-run training times:**
  - LightGBM 5-fold: ~6 min
  - CatBoost 5-fold: ~67 min (12× slower than LGBM)
  - XGBoost 5-fold: ~20 min
  - Optuna 50-trial LGBM: ~3.75 hours
  - Ensemble (stack/blend): ~2 min
- **Total compute:** ~7 hours across 10 runs
- **No GPU required**
