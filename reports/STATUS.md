# STATUS.md — Single Source of Truth

> Updated: 2026-03-01 (ALL 10 RUNS COMPLETE — pipeline finished)

---

## Current Step
**✅ PIPELINE COMPLETE. All 10 submissions done. Awaiting LB scores.**

Next: Record LB scores from https://www.kaggle.com/competitions/playground-series-s6e3/submissions → update EXPERIMENT_LOG.md

## Submission Results

| Run | OOF AUC | LB AUC | Status |
|-----|---------|--------|--------|
| s001 | 0.916016 | 0.91368 | ✅ Submitted |
| s002 | 0.916041 | 0.91368 | ✅ Submitted (FE v1 no lift for LGBM) |
| s003 | 0.916406 | **0.91388** | ✅ Submitted |
| s004 | **0.916597** | TBD | ✅ Submitted (LGBM + Optuna 50 trials) |
| s005 | 0.916334† | submitted | ✅ Submitted (XGB re-run for binary:logistic fix) |
| s006 | 0.915938 | NOT SUBMITTED | ✅ Ablated (target encoding HURT −0.0001) |
| s007 | **0.916530** | TBD | ✅ Submitted (CatBoost depth=6 l2=6 — BEAT s003!) |
| s008 | **0.916587** | TBD | ✅ Submitted (LGBM 10-fold, Optuna params) |
| s009 | **0.916709** | TBD | ✅ Submitted (LR stack: s004+s003+s005 OOF) |
| s010 | **0.916785** | TBD | ✅ Submitted — **BEST OOF** (equal-weight blend 5 models) |

†s005 re-run corrected from original 0.915593 (raw regression scores → proper probabilities)

## Leaderboard Position
- **Best OOF AUC achieved: 0.916785** (s010 equal-weight blend)
- **Best LB AUC so far: 0.91388** (s003 CatBoost — awaiting s004–s010 LB scores)
- **LB top (seen): 0.91603** (our best will likely challenge this)
- **All 10 submission slots used** (but 10 per day → slots reset tomorrow)

## Key Learnings (Final)
1. LGBM baseline (v0) already very strong (OOF 0.9160)
2. FE v1 adds NO lift to LGBM (GBDT finds these interactions via splits)
3. CatBoost gives small but real improvement (+0.0004 OOF, +0.0002 LB)
4. XGBoost weaker standalone but adds ensemble diversity (different errors)
5. Target encoding HURTS on synthetic PS data (estimation variance, not signal)
6. Optuna HPO: +0.0006 OOF AUC over LGBM baseline (converges at num_leaves=46, lr=0.01245)
7. CatBoost Optuna infeasible (32.5h) → manual diversity tuning achieves BETTER AUC
8. **CatBoost depth=6 l2=6 OUTPERFORMED depth=7 l2=3** → stronger regularization helps
9. Stacking (s009): +0.000112 over best single, equal meta-coefs (all models contribute equally)
10. 10-fold (s008): same OOF AUC as 5-fold but smoother test predictions for ensemble
11. **Final blend (s010): equal weights (0.2 each) optimal** — all 5 models contribute equally
12. **Total ensemble lift: +0.000769 OOF AUC** over LightGBM baseline

## OOF Progression
```
s001: 0.916016 (baseline LGBM)
s003: 0.916406 (+0.000390, CatBoost native cats)
s004: 0.916597 (+0.000581, LGBM Optuna HPO)
s007: 0.916530 (+0.000514, CatBoost deeper regularization)
s008: 0.916587 (+0.000571, LGBM 10-fold)
s009: 0.916709 (+0.000693, LR stack s004+s003+s005)
s010: 0.916785 (+0.000769, Equal-weight blend 5 models) ← BEST
```

## Next Actions
1. ⏳ Check Kaggle submissions page → record LB AUC for s004, s005, s007, s008, s009, s010
2. ✅ Final git commit: all metrics + updated reports → push to GitHub
3. ⏳ Update resume bullets with final LB rank
