# EXPERIMENT LOG — PS-S6E3 Churn Prediction

> One section per run. Updated immediately after each run completes.
> Maintained by: Experiment Manager

---

## s001 — LightGBM Baseline

**Date:** 2026-03-01
**Status:** ✅ COMPLETE

### Configuration
- Model: LightGBM
- CV: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
- Features: v0 (basic LabelEncode of categoricals, median-fill numerics)
- Params: n_estimators=2000, lr=0.05, num_leaves=127, feature_fraction=0.8, bagging_fraction=0.8, early_stopping=100
- Seed: 42

### What changed vs previous
- First run. No previous baseline.

### Expected outcome
- OOF AUC: ~0.84–0.87 (LightGBM default on Telco Churn data is typically 0.84–0.86; with 594K rows synthetic data, may be higher)
- LB AUC: within ±0.003 of OOF (low variance expected with 5-fold stratified CV)

### Results
- OOF AUC: **0.916016** ✅
- Fold AUCs: [0.91559, 0.91689, 0.91601, 0.91710, 0.91455]
- Fold std: **0.000926** (very stable)
- LB AUC: **0.91368** 🏆 (Rank #1 / 3 teams, 2026-03-01)
- Runtime: 330s (5:30 min)

### Conclusions / Next steps
- Outstanding baseline! Exceeds expected 0.84–0.87 range significantly.
- Low fold std (0.0009) confirms CV is highly reliable — LB should track closely.
- Next: add feature engineering v1 (s002) — expect +0.003–0.005 lift.
- LGBM default params are already very strong on this synthetic Telco data.
- CV-LB gap: 0.916016 - 0.91368 = 0.0023 → excellent CV calibration.

---

## s002 — Feature Engineering v1

**Date:** 2026-03-01
**Status:** ✅ COMPLETE

### Planned changes
1. charges_ratio = MonthlyCharges / (TotalCharges + 1)
2. avg_monthly_charge = TotalCharges / (tenure + 1)
3. tenure_bin = pd.cut(tenure, [0,12,24,48,72])
4. n_services = sum of all add-on service Yes flags
5. is_month_to_month = (Contract == 'Month-to-month').astype(int)
6. has_fiber = (InternetService == 'Fiber optic').astype(int)

### Expected lift
+0.002–0.005 AUC from feature interactions (based on Telco churn literature)

### Results
- OOF AUC: **0.916041** (+0.000025 vs s001 — negligible)
- LB AUC: **0.91368** (same as s001; LGBM already captures these features implicitly)
- Conclusion: FE v1 adds zero lift to LGBM. Expected — GBDT finds ratios/bins via splits.

---

## s003 — CatBoost (Native Categoricals)

**Date:** 2026-03-01
**Status:** ✅ COMPLETE

### Planned changes
- Switch to CatBoostClassifier
- Pass categorical column indices to cat_features parameter (no LabelEncoding)
- Keep feature version v1

### Expected lift
CatBoost handles ordered categoricals better; may gain +0.001–0.003 on this feature set

### Results
- OOF AUC: **0.916406** (+0.000390 vs s001 — new best!)
- Fold AUCs: [0.91607, 0.91715, 0.91649, 0.91760, 0.91476]
- Runtime: 4017s (67 min — CatBoost is 12× slower than LGBM)
- LB AUC: **0.91388** (Rank #9 / ~20 teams; +0.0002 over s001)
- CV-LB gap: 0.0025 (consistent with s001 gap of 0.0023 — CV is reliable)

---

## s004 — Optuna HPO (LightGBM)

**Date:** 2026-03-01
**Status:** 🔄 RUNNING (trial ~42/50 — completes ~40 min from session resumption)

### Planned changes
- Run Optuna with 50 trials on LightGBM (TPE sampler, seed=42)
- Search space: num_leaves [31,255], min_child_samples [5,50], feature_fraction [0.6,1.0],
  bagging_fraction [0.6,1.0], reg_alpha [0,2], reg_lambda [0,2], learning_rate [0.01,0.1]
- After study: inject best params → run final 5-fold CV → save submission

### Optuna progress (at session resumption)
- Best found: **0.916591** at trial 35 (vs 0.916016 baseline → +0.000575)
- Previous best: 0.91657 at trial 6; new best at trial 35 confirms TPE still exploring
- Each trial: 5-fold inner CV, ~4.5 min/trial, 50 total → ~3.75h total study time

### Expected lift
+0.0005 OOF AUC over LGBM baseline → ~0.9166

### Results
- OOF AUC: **TBD** (after full CV completes)
- LB AUC: **TBD**
- Best params: **TBD** (saved to submissions/s004/optuna_best_params.json)

---

## s005 — XGBoost Baseline

**Date:** 2026-03-01
**Status:** ✅ COMPLETE

### Planned changes
- XGBoostClassifier, v1 features, 5-fold StratifiedKFold
- Config: max_depth=7, subsample=0.8, colsample_bytree=0.8, lr=0.05, n_estimators=3000
- Added for ensemble diversity (different algorithm family from LGBM/CatBoost)

### Results
- OOF AUC: **0.915593** (weaker than LGBM 0.9160 and CatBoost 0.9164)
- LB AUC: submitted (score TBD from LB page)
- Note: XGBoost predictions required clipping [0,1] — fixed by adding `objective: binary:logistic`

### Conclusion
XGBoost underperforms LGBM/CatBoost on this dataset. Submitted for ensemble diversity value,
not standalone performance. Predictions will contribute meaningfully in stacking/blending.

---

## s006 — Feature Engineering v2 (Target Encoding + Deep Interactions)

**Date:** 2026-03-01
**Status:** ✅ COMPLETE (NOT SUBMITTED — worse than best)

### Planned changes
1. **Target encoding** (smoothed mean encoding with cross-val folds to avoid leakage)
   - `Contract_te`, `PaymentMethod_te`, `InternetService_te`
2. **Interaction pairs** (from s001-s005 feature importance top features)
   - `Contract × InternetService` cross (most discriminative combination)
   - `tenure_bin × is_month_to_month` cross
   - `MonthlyCharges × n_addon_services` product
3. **Statistical aggregations** (grouped by Contract type)
   - Mean/std of MonthlyCharges within each Contract group

### Expected lift
+0.003–0.006 AUC (target encoding on Contract typically big win on churn datasets)

### Results
- OOF AUC: **0.915938** (−0.000078 vs s001 baseline — HURT!)
- LB AUC: **NOT SUBMITTED** (saved daily submission slot)
- Features: 37 total (19 base + 11 v1 + 7 v2)

### Conclusion — Key Learning
Target encoding HURT performance on this dataset. Reasons:
1. Synthetic Playground Series data has very clean categorical distributions.
2. LightGBM already captures optimal split points for Contract, PaymentMethod, InternetService.
3. Smoothed mean encoding adds estimation variance that doesn't generalize.

**Action:** Use v1 features (not v2) for all remaining LGBM runs. Drop target encoding.
**New insight added to OPERATING_MANUAL.md.**

---

## s007 — CatBoost Manual Diversity Variant

**Date:** 2026-03-01
**Status:** ⏳ QUEUED (runs after s008 in pipeline)

### Planned changes (revised from original CatBoost+Optuna plan)
- **Original plan**: CatBoost + Optuna 30 trials
- **Revised plan**: Manual tuning — CatBoost Optuna is infeasible
  (30 trials × 5-fold CV × ~13 min/fold = 32.5 hours wall-clock time)
- **New approach**: Hand-tuned diversity variant with different inductive bias vs s003:
  - depth: 7 → **6** (shallower, more regularized, different split points)
  - l2_leaf_reg: 3.0 → **6.0** (more regularization)
  - learning_rate: 0.05 → **0.04** (slightly slower convergence)
  - random_strength: 1.0 → **0.5** (less split-level randomness)
  - bagging_temperature: 0.5 → **0.8** (more Bayesian-style bagging diversity)
- Feature version: v1 (v2 target encoding hurt: −0.0001 OOF)
- Activated via `--use-tuned-params` flag (reads `catboost_tuned` from config.yaml)

### Rationale
For the ensemble to work, we need diverse predictions from s003. Different depth/l2 combination
creates different decision boundaries. The OOF may be slightly lower than s003, but correlation
with s003 errors will be lower → ensemble benefit.

### Expected OOF
~0.9160–0.9165 (competitive but different from s003's error pattern)

### Results
- OOF AUC: **TBD**
- LB AUC: **TBD**

---

## s008 — LightGBM Best Params + 10-Fold CV

**Date:** 2026-03-01
**Status:** ⏳ QUEUED (runs after s009 in pipeline)

### Planned changes
- Use best LGBM params from s004 Optuna study (`optuna_best_params.json`)
- Increase to **10-fold** StratifiedKFold (more stable OOF, lower fold variance)
- Feature version: **v1** (NOT v2 — target encoding hurt)
- `--use-tuned-params` flag reads s004's best Optuna params

### Rationale
10-fold gives a better OOF estimate:
- 10 folds of 594K rows → ~535K train / ~60K val each fold
- Lower fold std = more reliable test prediction averaging
- Slightly different train/val splits → adds subtle prediction diversity for ensemble

### Expected lift
- OOF AUC: ~0.9166 (similar to s004; fold variance reduction main benefit)
- Fold std: ~0.0007 (lower than s004's ~0.001)

### Results
- OOF AUC: **TBD**
- LB AUC: **TBD**
- Fold std: **TBD**

---

## s009 — Stacking (Level-2 Logistic Regression)

**Date:** 2026-03-01
**Status:** ⏳ QUEUED (runs immediately after s004 completes — fast)

### Planned changes
- Level-1 models: **s004** (LGBM Optuna), **s003** (CatBoost), **s005** (XGBoost)
- Level-2 meta-model: LogisticRegression(C=1.0) trained on 3-column OOF matrix
- Meta-CV: 5-fold StratifiedKFold to produce stacked OOF without leakage
- Final meta-model trained on all training data → test predictions

### Rationale
s003 (CatBoost) + s004 (LGBM) + s005 (XGBoost) are 3 diverse algorithm families.
The LR meta-model learns: "when does LGBM/CatBoost/XGBoost each contribute?"
Meta-coefficients will show which model the ensemble trusts most.

### Expected lift
+0.001–0.002 OOF AUC over best individual model (s004)

### Results
- OOF AUC: **TBD**
- LB AUC: **TBD**
- Meta-model coefficients: **TBD**

---

## s010 — Final Weighted Blend

**Date:** 2026-03-01
**Status:** ⏳ QUEUED (runs last, after all components complete)

### Planned changes
- Blend 5 models: s004 (LGBM Optuna) + s003 (CatBoost) + s007 (CB diversity) + s008 (LGBM 10-fold) + s009 (Stack)
- Weights: optimized via scipy.minimize(neg_auc, SLSQP) on OOF AUC
- Constraint: weights ≥ 0, sum = 1

### Rationale
Final kitchen-sink ensemble combining:
- Best LGBM (s004, s008)
- Diverse CatBoost (s003, s007)
- XGBoost signal captured via stack (s009)
Optimized weighting extracts maximum ensemble benefit.

### Expected lift
+0.001–0.003 OOF AUC over best individual (s004)

### Results
- OOF AUC: **TBD**
- LB AUC: **TBD**
- Final rank: **TBD**
- Optimal weights: **TBD**
