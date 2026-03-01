# MASTER PROMPT — Kaggle Portfolio Maximizer

> Saved verbatim from the originating prompt. Consult this before every major step.

## PRIMARY OBJECTIVE (recruiter-focused)
- Meaningful leaderboard performance (relative rank matters).
- Reproducible, well-documented notebooks.
- Project maturity signals: correct CV, ablations, error analysis, experiment log.
- Public-facing GitHub artifacts.

## AUTONOMOUS EXECUTION
- Fully non-interactive. Document blockers in OPEN_QUESTIONS.md and continue.
- Fixed seeds, logged configs, scripted steps.

## HARD CONSTRAINTS
- Kaggle rules strictly: no leakage, no disallowed external data.
- Kaggle Notebooks as default compute.
- Evidence-based decisions.

## SUBAGENT ROLES
1. Competition Scout — selection + recruiter signal
2. Validation Guardian — CV, leakage checks, metric alignment
3. Model Engineer — pipeline + features + training
4. Experiment Manager — run bookkeeping + ablation discipline

## COMPETITION SELECTED
`playground-series-s6e3` — Telco Customer Churn, AUC metric, binary classification.
See reports/DECISIONS.md for full rationale.

## RUN PLAN
- s001: LightGBM baseline (5-fold StratifiedKFold)
- s002: LightGBM + feature engineering v1
- s003: CatBoost (native categoricals)
- s004: LightGBM + Optuna HPO
- s005: Weighted blend LGBM + CatBoost

## KEY FILES
- reports/STATUS.md       — current state
- reports/DECISIONS.md    — all design decisions
- reports/EXPERIMENT_LOG.md — per-run results
- reports/OPEN_QUESTIONS.md — blockers
- configs/config.yaml     — all hyperparameters
- src/train.py            — training pipeline
- src/submit.py           — submission CLI
