# TODO — Prioritized Action List

> Updated: 2026-03-01 (after s001–s006, pipeline queued for s007–s010)

---

## P1 — Active NOW (automation running)

- [x] s001: LightGBM baseline → OOF 0.916016, LB 0.91368
- [x] s002: LightGBM + FE v1 → OOF 0.916041, LB 0.91368
- [x] s003: CatBoost → OOF 0.916406, LB 0.91388 (best)
- [🔄] s004: LGBM + Optuna HPO (50 trials) — running at trial ~42/50
- [x] s005: XGBoost → OOF 0.915593, submitted
- [x] s006: LGBM + FE v2 ablation → OOF 0.915938, NOT submitted (target encoding hurt)
- [🔄] Pipeline queued: auto-submits s004 → s009 → s008 → s007 → s010 upon s004 completion

## P2 — After Pipeline Completes

- [ ] Record all LB scores from Kaggle submissions page in EXPERIMENT_LOG.md
- [ ] Update STATUS.md table with final OOF and LB AUC for s004–s010
- [ ] Update README.md results table with real s004–s010 numbers
- [ ] Update resume bullets with final rank and AUC
- [ ] Update model_card.md with final OOF/LB/rank

## P3 — Packaging (once all submissions done)

- [ ] `git init` the project directory
- [ ] `git add` all non-data files (src/, configs/, reports/, README.md, requirements.txt)
- [ ] `git commit -m "10-submission Kaggle PS-S6E3 churn prediction project"`
- [ ] `gh repo create kaggle-ps-s6e3-churn --public --source . --push`
- [ ] Create Kaggle public notebook (copy src/train.py → notebook format)
- [ ] Add leaderboard screenshot to README when competition ends

## Nice-to-Have (if time permits)

- [ ] SHAP analysis on s004 model (feature importance waterfall plots)
- [ ] Calibration plot (reliability diagram) — how well-calibrated are probabilities?
- [ ] Rank-average blending as alternative to linear blend in s010
- [ ] TabNet or simple MLP baseline for additional ensemble diversity
