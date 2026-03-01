# TODO — Prioritized Action List

> Updated: 2026-03-01 (s007 CatBoost running, s010 blend queued)

---

## P1 — Completed Runs ✅

- [x] s001: LightGBM baseline → OOF 0.916016, LB 0.91368
- [x] s002: LightGBM + FE v1 → OOF 0.916041, LB 0.91368
- [x] s003: CatBoost → OOF 0.916406, LB 0.91388 (best LB so far)
- [x] s004: LGBM + Optuna HPO (50 trials) → OOF 0.916597, submitted
- [x] s005: XGBoost re-run (binary:logistic fix) → OOF 0.916334
- [x] s006: LGBM + FE v2 ablation → OOF 0.915938, NOT submitted (target encoding hurt)
- [x] s008: LGBM 10-fold tuned → OOF 0.916587, submitted
- [x] s009: LR stack (s004+s003+s005) → OOF 0.916709, submitted

## P1 — Active NOW (pipeline running)

- [🔄] s007: CatBoost depth=6 l2=6 diversity variant — ~60 min remaining
- [⏳] s010: Final weighted blend (s004+s003+s007+s008+s009) — queued after s007

## P2 — After Pipeline Completes

- [ ] Update EXPERIMENT_LOG.md: s007, s010 Results sections with real OOF AUC
- [ ] Update STATUS.md table: s007, s010 OOF AUC, daily budget count
- [ ] Update README.md: s007, s010 in progression table; update resume bullets
- [ ] Update model_card.md: s010 final evaluation row
- [ ] Record all LB scores from Kaggle submissions page in EXPERIMENT_LOG.md
  (s004, s005, s007, s008, s009, s010)
- [ ] Final git commit: `git add submissions/s004/ submissions/s005/ submissions/s007/ submissions/s008/ submissions/s009/ submissions/s010/ reports/`
- [ ] `git push` to https://github.com/njaltran/kaggle-ps-s6e3-churn

## P3 — Packaging (nice-to-have)

- [ ] Create Kaggle public notebook (copy src/train.py → notebook format)
- [ ] Add leaderboard screenshot to README when competition ends

## Nice-to-Have (if time permits)

- [ ] SHAP analysis on s004 model (feature importance waterfall plots)
- [ ] Calibration plot (reliability diagram) — how well-calibrated are probabilities?
- [ ] Rank-average blending as alternative to linear blend in s010
- [ ] TabNet or simple MLP baseline for additional ensemble diversity
