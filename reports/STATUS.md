# STATUS.md — Single Source of Truth

> Updated: 2026-03-01 (pipeline queued, s004 finishing ~trial 42–50)

---

## Current Step
**s004 Optuna (trial ~42/50) completing. Full pipeline queued to auto-run.**

Pipeline will execute automatically:
`s004 done → submit s004 → s009 stack → s008 10-fold → s007 CatBoost → s010 blend`

## Submission Results

| Run | OOF AUC | LB AUC | Status |
|-----|---------|--------|--------|
| s001 | 0.916016 | 0.91368 | ✅ Submitted |
| s002 | 0.916041 | 0.91368 | ✅ Submitted (FE v1 no lift for LGBM) |
| s003 | 0.916406 | **0.91388** | ✅ Submitted — **best LB so far** |
| s004 | TBD (Optuna best@35trials=0.916591) | TBD | 🔄 Running (trial ~42/50) |
| s005 | 0.915593 | submitted | ✅ Submitted (XGB diversity for ensemble) |
| s006 | 0.915938 | NOT SUBMITTED | ✅ Ablated (target encoding HURT −0.0001) |
| s007 | TBD | TBD | ⏳ Queued (CatBoost depth=6 l2=6, v1 feats) |
| s008 | TBD | TBD | ⏳ Queued (LGBM 10-fold, Optuna params, v1) |
| s009 | TBD | TBD | ⏳ Queued (LR stack: s004+s003+s005 OOF) |
| s010 | TBD | TBD | ⏳ Queued (Blend: s004+s003+s007+s008+s009) |

## Leaderboard Position
- **Our best LB: 0.91388** (CatBoost s003)
- **LB top (seen): 0.91603** (gap: 0.00215)
- **Daily budget used: 4 of 10** → 6 remaining for s004+s007+s008+s009+s010

## Key Learnings
1. LGBM baseline (v0) already very strong (OOF 0.9160)
2. FE v1 adds NO lift to LGBM (GBDT finds these interactions via splits)
3. CatBoost gives small but real improvement (+0.0004 OOF, +0.0002 LB)
4. XGBoost weaker than LGBM/CatBoost, but useful for ensemble diversity
5. Target encoding HURTS on synthetic PS data (estimation variance, not signal)
6. Optuna HPO found improved LGBM params (OOF 0.916591 > baseline 0.91601)
7. CatBoost Optuna too slow (30 trials × 65 min/trial = 32.5h) — use manual tuning
8. s007 CatBoost changed to manual diversity variant: depth=6, l2=6, lr=0.04

## Revised Run Plan (s007–s010)
- **s007**: CatBoost manual tuning (depth=6, l2=6, lr=0.04 for diversity vs s003)
- **s008**: LGBM 10-fold + best Optuna params from s004 (v1 features)
- **s009**: Stacking: s004+s003+s005 OOF → LR meta-model
- **s010**: Weighted blend: s004+s003+s007+s008+s009 (scipy-optimized weights)

## Next Actions
1. 🔄 s004 Optuna completing (auto-submit queued)
2. 🔄 Pipeline will auto-run s009 → s008 → s007 → s010
3. ⏳ Update reports with final results
4. ⏳ Push git repo
