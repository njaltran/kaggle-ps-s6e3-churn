"""
cv.py — Cross-validation utilities.

Validation Guardian: ensures CV strategy matches competition structure.
Rules:
  - No leakage across folds
  - Stratify on target for binary classification
  - Fixed seed everywhere for reproducibility
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold


def get_cv_splitter(strategy: str, n_splits: int = 5, seed: int = 42):
    """Return the appropriate sklearn CV splitter."""
    if strategy == "stratified_kfold":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    elif strategy == "kfold":
        return KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    elif strategy == "group_kfold":
        return GroupKFold(n_splits=n_splits)
    else:
        raise ValueError(f"Unknown CV strategy: {strategy}")


def make_folds(
    df: pd.DataFrame,
    target_col: str,
    strategy: str = "stratified_kfold",
    n_splits: int = 5,
    seed: int = 42,
    group_col: str = None,
) -> pd.DataFrame:
    """
    Add a 'fold' column to df in-place.
    Returns df with fold assignments (0-indexed).
    """
    df = df.copy()
    df["fold"] = -1

    splitter = get_cv_splitter(strategy, n_splits, seed)

    y = df[target_col].values
    groups = df[group_col].values if group_col else None

    for fold_idx, (_, val_idx) in enumerate(splitter.split(df, y, groups)):
        df.loc[df.index[val_idx], "fold"] = fold_idx

    assert (df["fold"] == -1).sum() == 0, "Some rows not assigned to any fold!"
    return df


def leakage_check(train: pd.DataFrame, test: pd.DataFrame, id_col: str = "id"):
    """
    Basic leakage guard: warn if any test id appears in train.
    """
    if id_col not in train.columns or id_col not in test.columns:
        print("[LeakageCheck] id column not found — skipping id overlap check.")
        return
    train_ids = set(train[id_col])
    test_ids = set(test[id_col])
    overlap = train_ids & test_ids
    if overlap:
        print(f"[LeakageCheck] WARNING: {len(overlap)} id(s) appear in both train and test!")
    else:
        print("[LeakageCheck] PASS — no id overlap between train and test.")
