"""
ensemble.py — Blend and stack OOF + test predictions.

Supports two modes:
  blend : Weighted average (weights optimized on OOF AUC)
  stack : Level-2 Logistic Regression trained on OOF probability matrix

Usage:
  # s009: Stack (LR meta-learner)
  python src/ensemble.py --mode stack --runs s004 s003 s005 --run-id s009

  # s010: Final blend (4 models + stacked)
  python src/ensemble.py --mode blend --runs s004 s003 s007 s009 --run-id s010

  # Simple equal-weight blend (quick test)
  python src/ensemble.py --mode blend --runs s003 s004 --run-id s010 --equal-weights
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
ID_COL = "id"
TARGET_COL = "Churn"


def load_run(run_id: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load OOF predictions, test submission, and metrics for a run."""
    run_dir = ROOT / "submissions" / run_id
    oof_path = run_dir / "oof_predictions.csv"
    sub_path = run_dir / "submission.csv"

    if not oof_path.exists():
        print(f"[ERROR] OOF not found for {run_id}: {oof_path}")
        sys.exit(1)

    oof = pd.read_csv(oof_path)
    sub = pd.read_csv(sub_path)

    with open(run_dir / "metrics.json") as f:
        metrics = json.load(f)

    return oof, sub, metrics


def optimize_weights(oof_preds: np.ndarray, y_true: np.ndarray, n_models: int) -> np.ndarray:
    """Find weights that maximize OOF AUC via constrained scipy.optimize."""

    def neg_auc(weights):
        w = np.abs(weights) / np.abs(weights).sum()
        blend = (oof_preds * w[:, None]).sum(axis=0)
        return -roc_auc_score(y_true, blend)

    init = np.ones(n_models) / n_models
    bounds = [(0, 1)] * n_models
    res = minimize(neg_auc, init, method="SLSQP", bounds=bounds,
                   options={"maxiter": 1000, "ftol": 1e-9})
    best_w = np.abs(res.x) / np.abs(res.x).sum()
    return best_w


def blend_mode(runs, y_true, oof_preds, test_preds, run_ids, run_dir, equal_weights):
    """Optimize weights and blend predictions."""
    n = len(runs)
    if equal_weights:
        weights = np.ones(n) / n
        print(f"[Blend] Equal weights: {dict(zip(run_ids, weights.round(4)))}")
    else:
        print("[Blend] Optimizing weights on OOF AUC...")
        weights = optimize_weights(oof_preds, y_true, n)
        print(f"[Blend] Optimal weights: {dict(zip(run_ids, weights.round(4)))}")

    oof_blend  = (oof_preds  * weights[:, None]).sum(axis=0)
    test_blend = (test_preds * weights[:, None]).sum(axis=0)

    oof_auc = roc_auc_score(y_true, oof_blend)
    print(f"[Blend] Blended OOF AUC: {oof_auc:.6f}")

    return oof_blend, test_blend, weights, oof_auc


def stack_mode(runs, y_true, oof_preds, test_preds, run_ids, run_dir):
    """
    Train a Logistic Regression meta-model on OOF probability matrix.
    Uses StratifiedKFold to produce stacked OOF predictions without leakage.
    """
    print(f"[Stack] Training LR meta-model on {oof_preds.shape[0]}-model OOF matrix...")

    # oof_preds shape: (n_models, n_train)
    # We need X = (n_train, n_models)
    X_meta = oof_preds.T   # (n_train, n_models)
    y_meta = y_true

    # Produce stacked OOF predictions via CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    stacked_oof  = np.zeros(len(y_true))
    stacked_test = np.zeros(test_preds.shape[1])

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_meta, y_meta)):
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        lr.fit(X_meta[tr_idx], y_meta[tr_idx])
        stacked_oof[va_idx] = lr.predict_proba(X_meta[va_idx])[:, 1]

    # Final meta-model on all data for test predictions
    lr_final = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr_final.fit(X_meta, y_meta)
    X_test_meta = test_preds.T   # (n_test, n_models)
    stacked_test_preds = lr_final.predict_proba(X_test_meta)[:, 1]

    oof_auc = roc_auc_score(y_true, stacked_oof)
    print(f"[Stack] Stacked OOF AUC: {oof_auc:.6f}")
    print(f"[Stack] Meta-model coefs: {dict(zip(run_ids, lr_final.coef_[0].round(4)))}")

    weights = lr_final.coef_[0]  # for reporting
    return stacked_oof, stacked_test_preds, weights, oof_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs",  nargs="+", required=True, help="e.g. s003 s004 s005")
    parser.add_argument("--run-id",required=True,             help="Output run ID, e.g. s009")
    parser.add_argument("--mode",  default="blend",           choices=["blend", "stack"])
    parser.add_argument("--equal-weights", action="store_true")
    args = parser.parse_args()

    run_dir = ROOT / "submissions" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Ensemble] Mode={args.mode} | Runs={args.runs} | Out={args.run_id}")

    # Load all runs
    oofs, subs, all_metrics = [], [], []
    for rid in args.runs:
        oof, sub, metrics = load_run(rid)
        oofs.append(oof)
        subs.append(sub)
        all_metrics.append(metrics)
        print(f"  {rid}: OOF AUC = {metrics['oof_auc']:.6f}")

    # Align on id order from first run
    base_ids = oofs[0].set_index(ID_COL)
    y_true    = oofs[0][TARGET_COL].values

    # oof_preds: (n_models, n_train)
    oof_preds  = np.array([o.set_index(ID_COL)["oof_pred"].loc[base_ids.index].values
                           for o in oofs])
    # test_preds: (n_models, n_test)
    test_ids   = subs[0][ID_COL].values
    test_preds = np.array([s.set_index(ID_COL)[TARGET_COL].loc[test_ids].values
                           for s in subs])

    # Sanity-check: all predictions must be in [0,1] probability range
    print("\n[Scale Check] OOF / test prediction ranges:")
    for i, rid in enumerate(args.runs):
        lo_oof, hi_oof = oof_preds[i].min(), oof_preds[i].max()
        lo_test, hi_test = test_preds[i].min(), test_preds[i].max()
        ok_oof  = "✓" if -0.01 <= lo_oof  and hi_oof  <= 1.01 else "✗ OUT-OF-RANGE!"
        ok_test = "✓" if -0.01 <= lo_test and hi_test <= 1.01 else "✗ OUT-OF-RANGE!"
        print(f"  {rid} OOF:  [{lo_oof:.4f}, {hi_oof:.4f}] {ok_oof}")
        print(f"  {rid} test: [{lo_test:.4f}, {hi_test:.4f}] {ok_test}")
        if lo_oof < -0.01 or hi_oof > 1.01 or lo_test < -0.01 or hi_test > 1.01:
            print(f"[ERROR] {rid} predictions out of [0,1]. Ensure correct objective (binary:logistic for XGB).")
            sys.exit(1)
    print()

    if args.mode == "blend":
        oof_blend, test_blend, weights, oof_auc = blend_mode(
            args.runs, y_true, oof_preds, test_preds, args.runs, run_dir, args.equal_weights
        )
    else:  # stack
        oof_blend, test_blend, weights, oof_auc = stack_mode(
            args.runs, y_true, oof_preds, test_preds, args.runs, run_dir
        )

    # Save submission
    sub_df = pd.DataFrame({ID_COL: test_ids, TARGET_COL: test_blend})
    sub_df.to_csv(run_dir / "submission.csv", index=False)

    # Save OOF
    oof_df = pd.DataFrame({
        ID_COL: base_ids.index,
        TARGET_COL: y_true,
        "oof_pred": oof_blend,
    })
    oof_df.to_csv(run_dir / "oof_predictions.csv", index=False)

    # Save metrics
    metrics = {
        "run_id": args.run_id,
        "model": f"ensemble_{args.mode}",
        "component_runs": args.runs,
        "component_oof_aucs": {r: m["oof_auc"] for r, m in zip(args.runs, all_metrics)},
        "oof_auc": round(oof_auc, 6),
        "weights_or_coefs": {r: round(float(w), 4)
                             for r, w in zip(args.runs, weights)},
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    notes = run_dir / "notes.md"
    best_single = max(m["oof_auc"] for m in all_metrics)
    notes.write_text(
        f"# Run {args.run_id} — {args.mode.title()} Ensemble\n\n"
        f"Components: {args.runs}\n\n"
        f"## CV result\n"
        f"- Blended OOF AUC: **{oof_auc:.6f}**\n"
        f"- Best single component OOF: {best_single:.6f}\n"
        f"- Delta: {oof_auc - best_single:+.6f}\n\n"
        f"## LB result\n- (record after submission)\n\n"
        f"## Conclusion\n"
        f"- Ensemble {'improved' if oof_auc > best_single else 'did NOT improve'} "
        f"over best single model by {abs(oof_auc - best_single):.6f}\n"
    )

    print(f"\n[Done] {args.run_id} saved. OOF AUC = {oof_auc:.6f}")
    print(f"[Best single component] {best_single:.6f}")
    print(f"[Delta] {oof_auc - best_single:+.6f}")
    print(f"[Next] python src/submit.py --run-id {args.run_id} --message '{args.run_id}: {args.mode} ensemble'")


if __name__ == "__main__":
    main()
