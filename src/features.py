"""
features.py — Feature engineering for PS-S6E3 (Telco Customer Churn).

Subagent: Model Engineer
EDA date: 2026-03-01

Dataset: 594,194 train rows, 20 features, binary target=Churn, AUC metric.
No missing values. Mix of binary Yes/No, multi-class strings, and numeric.

Feature Versions
----------------
v0 : basic preprocessing only (LabelEncode + median fill) — used in s001
v1 : v0 + engineered interaction/ratio/count features  — used in s002+
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ── Column groups (from EDA) ──────────────────────────────
CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

# Add-on services that have Yes/No/No internet service as values
ADDON_SERVICES = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

TARGET_COL = "Churn"
ID_COL = "id"
POSITIVE_LABEL = "Yes"


def encode_target(series: pd.Series) -> pd.Series:
    """Encode Yes→1, No→0."""
    return (series == POSITIVE_LABEL).astype(int)


def basic_preprocessing(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str = TARGET_COL,
    id_col: str = ID_COL,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    v0 preprocessing:
      1. Drop id column
      2. Encode target (Yes→1, No→0)
      3. LabelEncode categoricals (fit on train only, handle unseen in test)
      4. No numeric imputation needed (zero nulls confirmed)

    Returns (X_train, X_test, y_train)
    """
    y_train = encode_target(train[target_col])

    X_train = train.drop(columns=[id_col, target_col]).copy()
    X_test = test.drop(columns=[id_col]).copy()

    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        encoders[col] = le

        # Handle unseen test categories gracefully
        test_vals = X_test[col].astype(str)
        known = set(le.classes_)
        X_test[col] = test_vals.apply(lambda v: v if v in known else le.classes_[0])
        X_test[col] = le.transform(X_test[col])

    return X_train, X_test, y_train.values


def _smoothed_mean_encode(train_col, train_target, test_col,
                           n_folds=5, smoothing=10, seed=42):
    """
    Smoothed Leave-one-out / KFold target encoding.
    Returns (train_encoded, test_encoded) arrays.
    Prevents leakage: each fold's OOF is encoded using the other folds' stats.
    """
    from sklearn.model_selection import KFold
    global_mean = train_target.mean()
    train_enc = np.zeros(len(train_col))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for tr_idx, va_idx in kf.split(train_col):
        stats = (
            pd.DataFrame({"col": train_col.iloc[tr_idx], "target": train_target.iloc[tr_idx]})
            .groupby("col")["target"]
            .agg(["mean", "count"])
        )
        # Smoothing: blend category mean toward global mean based on count
        stats["smoothed"] = (
            (stats["mean"] * stats["count"] + global_mean * smoothing)
            / (stats["count"] + smoothing)
        )
        train_enc[va_idx] = train_col.iloc[va_idx].map(stats["smoothed"]).fillna(global_mean).values

    # Test encoding: use all train data stats
    all_stats = (
        pd.DataFrame({"col": train_col, "target": train_target})
        .groupby("col")["target"]
        .agg(["mean", "count"])
    )
    all_stats["smoothed"] = (
        (all_stats["mean"] * all_stats["count"] + global_mean * smoothing)
        / (all_stats["count"] + smoothing)
    )
    test_enc = test_col.map(all_stats["smoothed"]).fillna(global_mean).values
    return train_enc, test_enc


def feature_engineering_v1(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    v1 feature engineering (on top of v0 preprocessed DataFrames).
    Use train_raw / test_raw for string-based logic before LabelEncoding.

    New features:
    ─────────────
    charges_ratio        : MonthlyCharges / (TotalCharges + 1)
                           Captures the fraction of total spend from current month.
    avg_monthly_charge   : TotalCharges / (tenure + 1)
                           Average historical monthly spend.
    charge_diff          : MonthlyCharges - avg_monthly_charge
                           Positive = recently upsold; negative = long-term discount.
    tenure_bin           : [0-12], [12-24], [24-48], [48-72] buckets (ordinal)
    monthly_charge_bin   : quartile bins of MonthlyCharges (0-3)
    n_addon_services     : count of add-on services with value "Yes"
    has_fiber            : InternetService == "Fiber optic" (high churn risk)
    is_month_to_month    : Contract == "Month-to-month" (highest churn risk)
    no_internet          : InternetService == "No" (very low churn risk)
    high_risk_flag       : is_month_to_month AND has_fiber (extreme churn risk)
    senior_alone         : SeniorCitizen=1 AND Partner=No AND Dependents=No
    """
    X_tr = X_train.copy()
    X_te = X_test.copy()

    for X, raw in [(X_tr, train_raw), (X_te, test_raw)]:
        # ── Ratio / arithmetic features ──────────────────
        X["charges_ratio"] = X["MonthlyCharges"] / (X["TotalCharges"] + 1)
        X["avg_monthly_charge"] = X["TotalCharges"] / (X["tenure"] + 1)
        X["charge_diff"] = X["MonthlyCharges"] - X["avg_monthly_charge"]

        # ── Binning ───────────────────────────────────────
        X["tenure_bin"] = pd.cut(
            X["tenure"], bins=[0, 12, 24, 48, 72], labels=[0, 1, 2, 3]
        ).astype(float)
        X["monthly_charge_bin"] = pd.qcut(
            X["MonthlyCharges"], q=4, labels=[0, 1, 2, 3], duplicates="drop"
        ).astype(float)

        # ── Service count (using raw strings) ─────────────
        n_services = sum((raw[col] == "Yes").astype(int) for col in ADDON_SERVICES)
        X["n_addon_services"] = n_services.values

        # ── Boolean flags from raw strings ────────────────
        X["has_fiber"]         = (raw["InternetService"] == "Fiber optic").astype(int).values
        X["is_month_to_month"] = (raw["Contract"] == "Month-to-month").astype(int).values
        X["no_internet"]       = (raw["InternetService"] == "No").astype(int).values
        X["high_risk_flag"]    = (X["has_fiber"] & X["is_month_to_month"]).astype(int)

        # ── Demographic interaction ───────────────────────
        X["senior_alone"] = (
            (raw["SeniorCitizen"] == 1) &
            (raw["Partner"] == "No") &
            (raw["Dependents"] == "No")
        ).astype(int).values

    return X_tr, X_te


def feature_engineering_v2(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    y_train: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    v2: Adds target encoding + deep interaction features on top of v1.

    New features:
    ─────────────
    contract_te         : Smoothed target encoding of Contract
    payment_te          : Smoothed target encoding of PaymentMethod
    internet_te         : Smoothed target encoding of InternetService
    contract_x_internet : Cross-feature: Contract (str) + '_' + InternetService (str) → label encoded
    tenure_x_contract   : tenure_bin × is_month_to_month
    charges_x_services  : MonthlyCharges × n_addon_services
    monthly_charge_rank : Rank of MonthlyCharges within Contract group (normalized)
    """
    X_tr = X_train.copy()
    X_te = X_test.copy()

    y_series = pd.Series(y_train, index=train_raw.index)

    # ── Target encodings (leakage-safe via KFold) ─────────
    for col in ["Contract", "PaymentMethod", "InternetService"]:
        safe_name = col.lower().replace(" ", "_") + "_te"
        tr_enc, te_enc = _smoothed_mean_encode(
            train_raw[col].reset_index(drop=True),
            y_series.reset_index(drop=True),
            test_raw[col].reset_index(drop=True),
        )
        X_tr[safe_name] = tr_enc
        X_te[safe_name] = te_enc

    # ── Cross-categorical feature ──────────────────────────
    cross_train = (train_raw["Contract"] + "_" + train_raw["InternetService"]).astype("category")
    cross_test  = (test_raw["Contract"]  + "_" + test_raw["InternetService"]).astype("category")
    le_cross = LabelEncoder()
    le_cross.fit(cross_train.tolist() + cross_test.tolist())
    X_tr["contract_x_internet"] = le_cross.transform(cross_train)
    X_te["contract_x_internet"] = le_cross.transform(
        cross_test.map(lambda v: v if v in le_cross.classes_ else le_cross.classes_[0])
    )

    # ── Interaction products ───────────────────────────────
    X_tr["tenure_x_contract"]   = X_tr["tenure_bin"] * X_tr["is_month_to_month"]
    X_te["tenure_x_contract"]   = X_te["tenure_bin"] * X_te["is_month_to_month"]

    X_tr["charges_x_services"]  = X_tr["MonthlyCharges"] * X_tr["n_addon_services"]
    X_te["charges_x_services"]  = X_te["MonthlyCharges"] * X_te["n_addon_services"]

    # ── Rank within Contract group ─────────────────────────
    # (normalized charge rank per contract type — relative price signal)
    tmp = pd.concat([
        train_raw[["Contract"]].assign(MonthlyCharges=X_tr["MonthlyCharges"].values, split="train"),
        test_raw[["Contract"]].assign( MonthlyCharges=X_te["MonthlyCharges"].values, split="test"),
    ], ignore_index=True)
    tmp["rank"] = tmp.groupby("Contract")["MonthlyCharges"].rank(pct=True)
    X_tr["monthly_charge_rank"] = tmp.loc[tmp["split"] == "train", "rank"].values
    X_te["monthly_charge_rank"] = tmp.loc[tmp["split"] == "test",  "rank"].values

    return X_tr, X_te
