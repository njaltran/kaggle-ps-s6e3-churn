# OPEN QUESTIONS & BLOCKERS

> Record issues here. Resolved items are struck through.

---

## Resolved

### ~~OQ-001: Data download 403~~
- **Issue:** Initial download attempt returned 403 Forbidden
- **Resolution:** Rules were already accepted. Retry succeeded on all files.
- **Status:** RESOLVED ✅

### ~~OQ-002: LB metric confirmation~~
- **Issue:** Confirm AUC (float probabilities) vs. accuracy (binary labels)
- **Resolution:** Confirmed via s001 LB score 0.91368 → AUC (ROC). Submit probabilities.
- **Status:** RESOLVED ✅

### ~~OQ-006: XGBoost predictions out of [0,1]~~
- **Issue:** s005 XGBoost predictions went outside [0,1] range (raw log-odds without sigmoid)
- **Root cause:** `objective: binary:logistic` missing from config
- **Fix:** Added `objective: binary:logistic` to config.yaml; clipped s005 submission to [0,1]
- **Status:** RESOLVED ✅

### ~~OQ-007: CatBoost seed parameter error~~
- **Issue:** `CatBoostClassifier` raises TypeError for `seed` keyword (expects `random_seed`)
- **Fix:** Added key rename in `train_catboost_fold()`: `if "seed" in p: p["random_seed"] = p.pop("seed")`
- **Status:** RESOLVED ✅

### ~~OQ-008: Optuna config key mismatch~~
- **Issue:** Code used `config["optuna"]` but YAML key is `config["optuna_lgbm"]`
- **Fix:** Updated references to `config["optuna_lgbm"]`
- **Status:** RESOLVED ✅

---

## Open

### OQ-003: Kaggle Notebook runner
- **Issue:** Kaggle Notebook `kaggle_playground-series-s6e3_runner` not yet created (requires Kaggle UI)
- **Workaround:** Running locally (594K rows, CPU — local runs produce identical results)
- **Action to resolve:** Open https://www.kaggle.com/competitions/playground-series-s6e3/code → New Notebook → paste src/train.py → run with competition dataset mounted
- **Blocking:** No

### OQ-004: LB score recording after submission
- **Issue:** Cannot programmatically retrieve own LB score via Kaggle API
- **Workaround:** After submitting, check https://www.kaggle.com/competitions/playground-series-s6e3/submissions → manually record Public LB score in EXPERIMENT_LOG.md
- **Action:** Document after each submission run

### OQ-005: GitHub repo not yet created
- **Issue:** Remote GitHub repo not initialized
- **Action:** `gh repo create kaggle-ps-s6e3-churn --public --source . --push` after all runs complete
- **Blocking:** No (can push at any time after submissions finish)

### ~~OQ-009: CatBoost Optuna infeasible on local CPU~~
- **Issue:** Original s007 plan was CatBoost + 30-trial Optuna HPO
- **Calculated cost:** 30 trials × 5-fold CV × ~13 min/fold = 32.5 hours — infeasible
- **Resolution:** Changed s007 to manual diversity tuning (depth=6, l2=6, lr=0.04)
- **Status:** RESOLVED (plan updated) ✅

### ~~OQ-011: s005 OOF/test predictions in wrong scale for ensemble~~
- **Issue:** s005 XGBoost was trained without `objective: binary:logistic` → raw regression scores, not probabilities. Range [-0.18, 1.16] vs expected [0,1]. Ensemble operations (weighted average) are nonsensical with mixed scales.
- **Discovery:** Caught during pre-pipeline sanity check of OOF prediction ranges
- **Fix:** Pipeline step 0 re-runs s005 with binary:logistic, overwriting both OOF and test prediction files with proper probabilities in [0,1]. s005 LB submission is NOT re-submitted. AUC metric is rank-based and identical before/after.
- **Status:** RESOLVED ✅

### OQ-010: s004 best Optuna params not yet known
- **Issue:** s004 Optuna still running (trial ~44/50); best params TBD
- **Impact:** s008 `--use-tuned-params` will read `submissions/s004/optuna_best_params.json`
- **Status:** Will auto-resolve when s004 completes (~20 min)
