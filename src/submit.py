"""
submit.py — Push a submission to Kaggle via CLI.

Usage:
  python src/submit.py --run-id s001 --message "s001: LightGBM baseline, 5-fold StratifiedKFold"
"""

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMPETITION_SLUG = "playground-series-s6e3"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, help="e.g. s001")
    parser.add_argument("--message", required=True, help="Short submission description")
    args = parser.parse_args()

    run_dir = ROOT / "submissions" / args.run_id
    submission_path = run_dir / "submission.csv"

    if not submission_path.exists():
        print(f"[ERROR] submission.csv not found at {submission_path}")
        print("Run train.py first.")
        raise SystemExit(1)

    # Load metrics for logging
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        oof = metrics.get("oof_auc", "N/A")
        print(f"[Info] Submitting {args.run_id} | OOF AUC: {oof}")

    print(f"[Info] Submission file: {submission_path}")
    print(f"[Info] Message: {args.message}")

    cmd = [
        "kaggle", "competitions", "submit",
        "-c", COMPETITION_SLUG,
        "-f", str(submission_path),
        "-m", args.message,
    ]
    print(f"[Exec] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"[Success] Submission accepted.\n{result.stdout}")
        # Prompt user to update notes.md with LB score
        print(
            f"\n[Action required] Record the Public LB score in:\n"
            f"  submissions/{args.run_id}/notes.md\n"
            f"  reports/EXPERIMENT_LOG.md"
        )
    else:
        print(f"[Error] Submission failed:\n{result.stderr}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
