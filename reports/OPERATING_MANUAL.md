# OPERATING MANUAL — PS-S6E3 Churn Prediction

> Derived from MASTER_PROMPT.md. Updated as project progresses.
> Last updated: 2026-03-01

---

## SKILLS

| Skill | Tool/Approach |
|-------|--------------|
| Competition scouting | `kaggle competitions list`, leaderboard analysis, deadline/teams check |
| EDA | pandas shape/dtypes/describe, null check, target distribution, leakage check |
| CV design | StratifiedKFold (binary clf, 22.5% positive rate, no time/group structure) |
| Baseline modeling | LightGBM with LabelEncoded categoricals, 5-fold CV, AUC metric |
| Feature engineering | Ratio features, binning, service count aggregations, interaction flags |
| Model diversity | LightGBM → CatBoost (native cats) → XGBoost as fallback |
| HPO | Optuna with TPE sampler, 50–100 trials, CV as objective |
| Ensembling | Simple average → rank average → weighted blend (optimized on OOF) |
| Submission | `kaggle competitions submit` via CLI, record LB in EXPERIMENT_LOG |
| Packaging | README case study, model_card.md, resume bullets, GitHub-ready repo |

---

## RULES

1. **Non-interactive**: Never pause for user input. Document blockers in OPEN_QUESTIONS.md.
2. **Kaggle rules compliance**: No external data, no leakage, no private solution sharing.
3. **Compute policy**: Run locally (CPU) since 594K rows × 20 features fits easily in <4GB RAM and 5-fold LGBM finishes in ~90 seconds. No GPU needed.
4. **Fixed seeds**: All models use seed=42. CV uses shuffle=True, random_state=42.
5. **One change at a time**: Each run changes 1–2 variables. Log the diff clearly.
6. **Context management**: All state written to reports/*.md. Never rely on chat history.
7. **LB metric = AUC**: Submit probabilities (floats), not binary labels.

---

## PROJECT PLAYBOOK

### Phase 0: Setup ✅
- [x] Competition selected: playground-series-s6e3
- [x] Data downloaded: train.csv (594K rows), test.csv (254K rows), sample_submission.csv
- [x] EDA complete: 20 features, binary target, zero nulls, AUC metric
- [x] Repo skeleton created
- [x] MASTER_PROMPT.md, OPERATING_MANUAL.md written
- [x] config.yaml updated with correct target_col and feature lists
- [x] src/train.py, src/features.py, src/cv.py, src/submit.py written

### Phase 1: Baseline (s001) ✅
- [x] Run s001: LightGBM 5-fold baseline → OOF 0.916016, LB 0.91368
- [x] Verified AUC metric from LB score — submit probabilities
- [x] Record in EXPERIMENT_LOG.md

### Phase 2: Feature Engineering (s002) ✅
- [x] Added 11 FE v1 features (ratios, bins, flags, service count)
- [x] Run s002 → OOF 0.916041 (+0.000025 — no meaningful lift)
- [x] Key finding: LGBM already finds these interactions; FE v1 ≈ no-op for tree models

### Phase 3: Model Diversity (s003) ✅
- [x] CatBoost with native cat_features (no LabelEncode needed)
- [x] OOF 0.916406, LB 0.91388 — +0.0004 OOF, +0.0002 LB vs baseline ✓
- [x] CatBoost symmetric tree structure helps regularization on high-cardinality cats

### Phase 4a: Algorithm Ablation (s005) ✅
- [x] XGBoost 5-fold → OOF 0.915593 (weaker than LGBM/CatBoost)
- [x] Kept for ensemble diversity (different error pattern)
- [x] Fixed: must set `objective: binary:logistic` or predictions are raw log-odds

### Phase 4b: Feature Ablation (s006) ✅
- [x] Target encoding v2 → OOF 0.915938 (−0.0001 vs baseline — HURT)
- [x] NOT submitted (saved daily slot). v1 features remain best for LGBM.

### Phase 5: HPO (s004) 🔄
- [🔄] Optuna 50-trial TPE on LGBM → best trial 35: OOF 0.916591
- [ ] Complete final CV → submit
- [ ] Record LB score

### Phase 6: Ensemble (s007–s010) ⏳
- [ ] s007: CatBoost diversity variant (manual, depth=6, l2=6)
- [ ] s008: LGBM 10-fold with Optuna params
- [ ] s009: LR stack (s004+s003+s005 OOF)
- [ ] s010: Weighted blend (all best models)
- [ ] Submit and record all LB scores

### Phase 7: Packaging (final)
- [ ] Update README.md with final numbers
- [ ] Update model_card.md with final OOF/LB/rank
- [ ] git init → add all non-data files → push to GitHub
- [ ] Record final rank for resume bullets

---

## LESSONS LEARNED
*(append as project progresses)*

- **2026-03-01 [Setup]:** Dataset confirmed as synthetic Telco Churn (IBM original). No missing values. TotalCharges ≈ tenure × MonthlyCharges. AUC metric confirmed from LB baseline 0.90504. Always submit probabilities (not binary labels).

- **2026-03-01 [s001→s002]:** Feature engineering v1 (ratios, bins, flags) adds NO lift to LightGBM (OOF +0.000025, essentially zero). GBDT creates optimal split points from raw features; precomputing ratios/bins that the tree would discover anyway is redundant. Key takeaway: FE helps when creating new INFORMATION, not new representations of existing information.

- **2026-03-01 [s006 ablation]:** Target encoding HURT LGBM on this synthetic PS dataset (−0.0001 OOF). Reasons: (1) synthetic data has clean categorical distributions already captured by LGBM splits; (2) smoothed mean encoding adds estimation variance proportional to fold size; (3) PS data is generated from simple tabular distributions where GBDT splits are already near-optimal. Lesson: ALWAYS ablate target encoding before adopting it; never assume it helps, especially on synthetic data.

- **2026-03-01 [s003]:** CatBoost outperforms LightGBM (+0.0004 OOF, +0.0002 LB) despite being 12× slower (~67 min vs ~6 min for 5-fold). CatBoost's native ordered-target-statistics encoding for categoricals extracts more signal than LabelEncoding. Worth running when you have categorical-heavy data and time allows.

- **2026-03-01 [s005]:** XGBoost requires `objective: binary:logistic` in params or predict() returns raw regression-style scores (range can be [-0.2, 1.2]). Always verify prediction range [0,1] before submitting AND before running any ensemble. AUC is rank-based (safe), but weighted blends or stack meta-training with mixed probability/score scales gives nonsensical results. RULE: print `df['oof_pred'].agg(['min','max'])` for each component before any ensemble step. Fixed by: (1) adding `objective: binary:logistic` to config; (2) pipeline step 0 re-runs s005 to get proper probabilities for ensemble components.

- **2026-03-01 [s004 Optuna]:** LGBM Optuna on 594K rows with 5-fold CV inner loop: each trial ~4.5 minutes on M-series CPU. 50-trial study = ~3.75 hours. Best improvement at trial 35 (0.916591 vs 0.916016 baseline). Optuna kept exploring until trial 35 despite trial 6 having the second-best result — TPE sometimes finds better regions late. Do NOT terminate early.

- **2026-03-01 [CatBoost Optuna infeasibility]:** CatBoost 5-fold CV inside Optuna: ~13 min/fold × 5 = 65 min/trial. 30 trials = 32.5 hours on local CPU — infeasible. Always compute estimated compute budget BEFORE running Optuna with CatBoost/XGBoost. Solution: use manual hand-tuning or reduce to 3-5 trials for budget Optuna runs.

- **2026-03-01 [CV reliability]:** With 5-fold StratifiedKFold on 594K rows, CV-to-LB gap is extremely consistent: 0.0023–0.0025 AUC across all runs. This confirms the CV is a reliable proxy for LB. Can confidently make go/no-submit decisions based on OOF AUC alone.
