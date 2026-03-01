"""
train.py — Full training pipeline for PS-S6E3 (Telco Customer Churn).

Subagent: Model Engineer
Competition: playground-series-s6e3
Metric: AUC (ROC) — submit probabilities [0, 1]

Usage
-----
  # s001: LightGBM baseline
  python src/train.py --run-id s001 --model lgbm --feature-version v0

  # s002: LightGBM + feature engineering
  python src/train.py --run-id s002 --model lgbm --feature-version v1

  # s003: CatBoost
  python src/train.py --run-id s003 --model catboost --feature-version v1

  # s004: LightGBM + Optuna
  python src/train.py --run-id s004 --model lgbm --feature-version v1 --optuna

  # s005: XGBoost baseline
  python src/train.py --run-id s005 --model xgboost --feature-version v1

  # s006: LightGBM + feature engineering v2
  python src/train.py --run-id s006 --model lgbm --feature-version v2

  # s007: CatBoost + Optuna
  python src/train.py --run-id s007 --model catboost --feature-version v2 --optuna-catboost

  # s008: LightGBM tuned (from s004) + 10-fold CV
  python src/train.py --run-id s008 --model lgbm --feature-version v2 --n-folds 10 --use-tuned-params

  # s009: Stacking
  python src/ensemble.py --mode stack --runs s004 s007 s005 --run-id s009

  # s010: Final weighted ensemble
  python src/ensemble.py --mode blend --runs s004 s007 s005 s009 --run-id s010

On Kaggle Notebook:
  !python /kaggle/input/<dataset>/src/train.py --run-id s001 --model lgbm
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import (
    basic_preprocessing,
    feature_engineering_v1,
    feature_engineering_v2,
    CATEGORICAL_COLS,
    TARGET_COL,
    ID_COL,
)


# ─────────────────────────────────────────────────────────
#  Config helpers
# ─────────────────────────────────────────────────────────

def load_config(path=None) -> dict:
    if path is None:
        path = ROOT / "configs" / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def get_run_dir(run_id: str, on_kaggle: bool) -> Path:
    if on_kaggle:
        d = Path(f"/kaggle/working/runs/{run_id}")
    else:
        d = ROOT / "submissions" / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


# ─────────────────────────────────────────────────────────
#  Model trainers
# ─────────────────────────────────────────────────────────

def train_lgbm_fold(X_tr, y_tr, X_val, y_val, params: dict):
    import lightgbm as lgb

    p = dict(params)
    n_estimators = p.pop("n_estimators", 3000)
    p.pop("seed", None)

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model = lgb.train(
        p,
        dtrain,
        num_boost_round=n_estimators,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=500),
        ],
    )
    return model


def train_catboost_fold(X_tr, y_tr, X_val, y_val, params: dict, cat_features):
    from catboost import CatBoostClassifier, Pool

    p = dict(params)
    # CatBoost uses random_seed, not seed
    if "seed" in p:
        p["random_seed"] = p.pop("seed")
    model = CatBoostClassifier(**p)
    model.fit(
        Pool(X_tr, y_tr, cat_features=cat_features),
        eval_set=Pool(X_val, y_val, cat_features=cat_features),
        verbose=500,
    )
    return model


def train_xgboost_fold(X_tr, y_tr, X_val, y_val, params: dict):
    import xgboost as xgb

    p = dict(params)
    n_estimators = p.pop("n_estimators", 3000)
    early_stopping = p.pop("early_stopping_rounds", 100)

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval   = xgb.DMatrix(X_val, label=y_val)

    model = xgb.train(
        p,
        dtrain,
        num_boost_round=n_estimators,
        evals=[(dval, "val")],
        early_stopping_rounds=early_stopping,
        verbose_eval=500,
    )
    return model


# ─────────────────────────────────────────────────────────
#  Optuna objective
# ─────────────────────────────────────────────────────────

def optuna_objective(trial, X_full, y_full, n_splits=5, seed=42):
    import lightgbm as lgb

    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "n_jobs": -1,
        "num_leaves":        trial.suggest_int("num_leaves", 31, 255),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "feature_fraction":  trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq":      1,
        "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda":        trial.suggest_float("reg_lambda", 0.0, 2.0),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
    }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for tr_idx, va_idx in skf.split(X_full, y_full):
        dtrain = lgb.Dataset(X_full.iloc[tr_idx], label=y_full[tr_idx])
        dval   = lgb.Dataset(X_full.iloc[va_idx], label=y_full[va_idx], reference=dtrain)
        model  = lgb.train(
            params, dtrain, num_boost_round=3000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(9999)],
        )
        preds = model.predict(X_full.iloc[va_idx])
        scores.append(roc_auc_score(y_full[va_idx], preds))
    return float(np.mean(scores))


# ─────────────────────────────────────────────────────────
#  Main CV loop
# ─────────────────────────────────────────────────────────

def run_cv(train_df, test_df, model_type, feature_version, config, run_dir,
           n_splits=5, use_tuned_params=False):
    """Run StratifiedKFold CV. Save OOF, test preds, feature importance."""
    cv_cfg = config["cv"]
    seed = cv_cfg["seed"]

    # Leakage guard: no id overlap
    train_ids = set(train_df[ID_COL])
    test_ids  = set(test_df[ID_COL])
    overlap = train_ids & test_ids
    if overlap:
        print(f"[LeakageCheck] WARNING: {len(overlap)} ids overlap train/test!")
    else:
        print("[LeakageCheck] PASS — zero id overlap.")

    # Preprocessing
    X_train, X_test, y_train = basic_preprocessing(train_df, test_df)

    if feature_version == "v1":
        X_train, X_test = feature_engineering_v1(X_train, X_test, train_df, test_df)
    elif feature_version == "v2":
        X_train, X_test = feature_engineering_v1(X_train, X_test, train_df, test_df)
        X_train, X_test = feature_engineering_v2(X_train, X_test, train_df, test_df, y_train)

    print(f"[Features] Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # CatBoost needs cat_features indices AFTER preprocessing (re-encoded as int)
    cat_feat_indices = [X_train.columns.tolist().index(c)
                        for c in CATEGORICAL_COLS if c in X_train.columns]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof_preds   = np.zeros(len(train_df))
    test_preds  = np.zeros(len(test_df))
    fold_scores = []
    importances = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\n--- Fold {fold+1}/{n_splits} ---")
        X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
        X_va, y_va = X_train.iloc[va_idx],  y_train[va_idx]

        if model_type == "lgbm":
            params = dict(config.get("lgbm_tuned", config["lgbm_baseline"])) \
                     if use_tuned_params else dict(config["lgbm_baseline"])
            params["seed"] = seed
            model = train_lgbm_fold(X_tr, y_tr, X_va, y_va, params)
            val_p  = model.predict(X_va)
            test_p = model.predict(X_test)
            fi = pd.Series(
                model.feature_importance(importance_type="gain"),
                index=X_tr.columns, name=f"fold_{fold}"
            )

        elif model_type == "catboost":
            params = dict(config.get("catboost_tuned", config["catboost_baseline"])) \
                     if use_tuned_params else dict(config["catboost_baseline"])
            model = train_catboost_fold(X_tr, y_tr, X_va, y_va, params, cat_feat_indices)
            val_p  = model.predict_proba(X_va)[:, 1]
            test_p = model.predict_proba(X_test)[:, 1]
            fi = pd.Series(
                model.get_feature_importance(),
                index=X_tr.columns, name=f"fold_{fold}"
            )

        elif model_type == "xgboost":
            import xgboost as xgb
            params = dict(config["xgboost_baseline"])
            model = train_xgboost_fold(X_tr, y_tr, X_va, y_va, params)
            val_p  = model.predict(xgb.DMatrix(X_va))
            test_p = model.predict(xgb.DMatrix(X_test))
            fi = pd.Series(
                model.get_score(importance_type="gain"),
            ).reindex(X_tr.columns).fillna(0).rename(f"fold_{fold}")

        else:
            raise ValueError(f"Unknown model: {model_type}")

        score = roc_auc_score(y_va, val_p)
        print(f"Fold {fold+1} AUC = {score:.6f}")

        oof_preds[va_idx] += val_p
        test_preds         += test_p / n_splits
        fold_scores.append(score)
        importances.append(fi)

    oof_auc = roc_auc_score(y_train, oof_preds)
    print(f"\n{'='*50}")
    print(f"OOF AUC : {oof_auc:.6f}")
    print(f"Fold AUC: {[round(s,5) for s in fold_scores]}")
    print(f"Fold std: {np.std(fold_scores):.6f}")
    print(f"{'='*50}")

    # Save OOF
    pd.DataFrame({
        ID_COL: train_df[ID_COL].values,
        TARGET_COL: y_train,
        "oof_pred": oof_preds,
    }).to_csv(run_dir / "oof_predictions.csv", index=False)

    # Save feature importance
    fi_df = pd.concat(importances, axis=1)
    fi_df["mean_gain"] = fi_df.mean(axis=1)
    fi_df.sort_values("mean_gain", ascending=False).to_csv(
        run_dir / "feature_importance.csv"
    )

    # Save submission
    sub = pd.DataFrame({
        ID_COL: test_df[ID_COL].values,
        TARGET_COL: test_preds,
    })
    sub.to_csv(run_dir / "submission.csv", index=False)
    print(f"[Saved] submission.csv | preds range [{test_preds.min():.4f}, {test_preds.max():.4f}]")

    metrics = {
        "run_id": None,
        "model": model_type,
        "feature_version": feature_version,
        "oof_auc": round(oof_auc, 6),
        "fold_aucs": [round(s, 6) for s in fold_scores],
        "fold_std": round(float(np.std(fold_scores)), 6),
        "n_splits": n_splits,
        "seed": seed,
        "n_features": X_train.shape[1],
        "train_rows": len(train_df),
    }
    return metrics


# ─────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PS-S6E3 training pipeline")
    parser.add_argument("--run-id",         default="s001")
    parser.add_argument("--model",          default="lgbm", choices=["lgbm", "catboost", "xgboost"])
    parser.add_argument("--feature-version",default="v0",   choices=["v0", "v1", "v2"])
    parser.add_argument("--config",         default=None)
    parser.add_argument("--n-folds",        default=5, type=int, help="Number of CV folds")
    parser.add_argument("--use-tuned-params", action="store_true",
                        help="Load best params from s004/s007 optuna_best_params.json")
    parser.add_argument("--optuna",         action="store_true",
                        help="Run Optuna HPO on LightGBM (s004)")
    parser.add_argument("--optuna-catboost",action="store_true",
                        help="Run Optuna HPO on CatBoost (s007)")
    args = parser.parse_args()

    config = load_config(args.config)

    on_kaggle = os.path.exists("/kaggle/input")
    if on_kaggle:
        comp_slug = config["competition"]["slug"]
        data_dir  = Path(f"/kaggle/input/{comp_slug}")
    else:
        data_dir  = ROOT / "data"

    run_dir = get_run_dir(args.run_id, on_kaggle)

    print(f"[Run]  {args.run_id} | model={args.model} | features={args.feature_version}")
    print(f"[Data] {data_dir}")
    print(f"[Out]  {run_dir}")

    t0 = time.time()
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df  = pd.read_csv(data_dir / "test.csv")
    print(f"[Data] train {train_df.shape}, test {test_df.shape}")

    # ── Optional Optuna HPO (s004) ────────────────────────
    if args.optuna:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        X_tr_full, X_te_optuna, y_tr = basic_preprocessing(train_df, test_df)
        if args.feature_version == "v1":
            X_tr_full, _ = feature_engineering_v1(X_tr_full, X_te_optuna, train_df, test_df)
        elif args.feature_version == "v2":
            X_tr_full, _ = feature_engineering_v1(X_tr_full, X_te_optuna, train_df, test_df)
            X_tr_full, _ = feature_engineering_v2(X_tr_full, _, train_df, test_df, y_tr)

        print("[Optuna] Starting 50-trial HPO study...")
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=config["optuna_lgbm"]["seed"]),
        )
        study.optimize(
            lambda trial: optuna_objective(trial, X_tr_full, y_tr),
            n_trials=config["optuna_lgbm"]["n_trials"],
            show_progress_bar=True,
        )
        best = study.best_params
        print(f"[Optuna] Best params: {best}")
        print(f"[Optuna] Best AUC: {study.best_value:.6f}")

        # Inject best params into lgbm config
        config["lgbm_baseline"].update(best)
        with open(run_dir / "optuna_best_params.json", "w") as f:
            json.dump({"best_params": best, "best_auc": study.best_value}, f, indent=2)

    # ── Optional CatBoost Optuna HPO (s007) ──────────────
    if args.optuna_catboost:
        import optuna
        from catboost import CatBoostClassifier, Pool
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        _, _, y_tr = basic_preprocessing(train_df, test_df)
        X_tr_full, _, _ = basic_preprocessing(train_df, test_df)
        if args.feature_version in ("v1", "v2"):
            X_tr_full, _ = feature_engineering_v1(X_tr_full, X_tr_full, train_df, train_df)
        if args.feature_version == "v2":
            X_tr_full, _ = feature_engineering_v2(X_tr_full, X_tr_full, train_df, train_df, y_tr)

        cat_feat_indices = [X_tr_full.columns.tolist().index(c)
                            for c in CATEGORICAL_COLS if c in X_tr_full.columns]

        def cb_objective(trial):
            params = {
                "iterations": 2000,
                "eval_metric": "AUC",
                "verbose": 0,
                "task_type": "CPU",
                "early_stopping_rounds": 100,
                "depth":             trial.suggest_int("depth", 4, 10),
                "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "l2_leaf_reg":       trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
                "random_strength":   trial.suggest_float("random_strength", 0.5, 3.0),
                "random_seed": config["optuna_catboost"]["seed"],
            }
            skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores2 = []
            for tr2, va2 in skf2.split(X_tr_full, y_tr):
                m2 = CatBoostClassifier(**params)
                m2.fit(Pool(X_tr_full.iloc[tr2], y_tr[tr2], cat_features=cat_feat_indices),
                       eval_set=Pool(X_tr_full.iloc[va2], y_tr[va2], cat_features=cat_feat_indices),
                       verbose=0)
                scores2.append(roc_auc_score(y_tr[va2], m2.predict_proba(X_tr_full.iloc[va2])[:, 1]))
            return float(np.mean(scores2))

        print(f"[Optuna-CatBoost] Starting {config['optuna_catboost']['n_trials']}-trial study...")
        study_cb = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=config["optuna_catboost"]["seed"]),
        )
        study_cb.optimize(cb_objective, n_trials=config["optuna_catboost"]["n_trials"],
                          show_progress_bar=True)
        best_cb = study_cb.best_params
        best_cb.update({"iterations": 3000, "eval_metric": "AUC", "verbose": 0,
                        "task_type": "CPU", "early_stopping_rounds": 100,
                        "seed": config["optuna_catboost"]["seed"]})
        print(f"[Optuna-CatBoost] Best: {best_cb}")
        config["catboost_baseline"].update(best_cb)
        with open(run_dir / "optuna_catboost_best_params.json", "w") as f:
            json.dump({"best_params": best_cb, "best_auc": study_cb.best_value}, f, indent=2)

    # Load tuned LGBM params from s004 if requested (s008)
    if args.use_tuned_params:
        s004_params_path = ROOT / "submissions" / "s004" / "optuna_best_params.json"
        if s004_params_path.exists():
            with open(s004_params_path) as f:
                best_p = json.load(f)["best_params"]
            config["lgbm_tuned"].update(best_p)
            print(f"[s008] Loaded tuned LGBM params from s004: {best_p}")
        else:
            print("[s008] Warning: s004 optuna_best_params.json not found. Using lgbm_baseline.")

    # ── Main CV run ───────────────────────────────────────
    metrics = run_cv(train_df, test_df, args.model, args.feature_version, config, run_dir,
                     n_splits=args.n_folds, use_tuned_params=args.use_tuned_params)
    metrics["run_id"] = args.run_id
    metrics["elapsed_seconds"] = round(time.time() - t0, 1)

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Write notes stub
    notes = run_dir / "notes.md"
    if not notes.exists():
        notes.write_text(
            f"# Run {args.run_id} Notes\n\n"
            f"Model: `{args.model}` | Features: `{args.feature_version}`\n\n"
            f"## What changed vs previous run\n- (fill in)\n\n"
            f"## CV result\n- OOF AUC: **{metrics['oof_auc']}**\n"
            f"- Fold AUCs: {metrics['fold_aucs']}\n"
            f"- Std: {metrics['fold_std']}\n\n"
            f"## LB result\n- Public LB AUC: **(record after submission)**\n\n"
            f"## Conclusion\n- (fill in)\n\n"
            f"## Next steps\n- (fill in)\n"
        )

    print(f"\n[Done] {args.run_id} complete in {metrics['elapsed_seconds']}s")
    print(f"[Done] OOF AUC = {metrics['oof_auc']}")
    print(f"[Next] python src/submit.py --run-id {args.run_id} --message '{args.run_id}: {args.model} {args.feature_version}'")


if __name__ == "__main__":
    main()
