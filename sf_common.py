from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CREDIT_EVENT_CODES = {"02", "03", "09"}
VOLUNTARY_PAYOFF_CODE = "01"
SNAPSHOT_AGE = 12
FORWARD_MONTHS = 12
TEST_VINTAGE = 2022


def get_engine():
    """Reuse the already-working MySQL connection from Phase 1 ETL."""
    try:
        from structured_finance_etl_FINAL import create_database_and_engine
    except ImportError as exc:
        raise ImportError(
            "Put structured_finance_etl_FINAL.py in the same project folder "
            "as these scripts, or add the project folder to PYTHONPATH."
        ) from exc
    return create_database_and_engine()


def period_index(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.zfill(6)
    y = pd.to_numeric(s.str[:4], errors="coerce")
    m = pd.to_numeric(s.str[4:6], errors="coerce")
    return y * 12 + m


def weighted_average(values, weights):
    v = pd.to_numeric(pd.Series(values), errors="coerce")
    w = pd.to_numeric(pd.Series(weights), errors="coerce")
    mask = v.notna() & w.notna() & (w > 0)
    if not mask.any():
        return np.nan
    return float(np.average(v[mask], weights=w[mask]))


def load_12m_snapshot_dataset(engine) -> pd.DataFrame:
    """
    One observation per loan near month-12 seasoning.
    Targets are defined strictly in the next 12 reporting months.
    This avoids using an undefined lifetime/censored target as a 12-month PD.
    """
    sql = """
    WITH snap_candidates AS (
        SELECT
            p.loan_sequence_number,
            p.monthly_reporting_period,
            p.current_actual_upb,
            p.current_loan_delinquency_status,
            p.loan_age,
            p.current_interest_rate,
            p.estimated_ltv,
            ROW_NUMBER() OVER (
                PARTITION BY p.loan_sequence_number
                ORDER BY ABS(p.loan_age - 12), p.monthly_reporting_period
            ) AS rn
        FROM freddie_performance p
        WHERE p.loan_age BETWEEN 10 AND 14
          AND (p.zero_balance_code IS NULL OR p.zero_balance_code = '')
          AND p.current_actual_upb > 0
    ),
    event_dates AS (
        SELECT
            loan_sequence_number,
            MIN(CASE WHEN zero_balance_code IN ('02','03','09') THEN
                CAST(SUBSTRING(monthly_reporting_period,1,4) AS UNSIGNED) * 12
                + CAST(SUBSTRING(monthly_reporting_period,5,2) AS UNSIGNED)
            END) AS first_credit_event_idx,
            MIN(CASE WHEN zero_balance_code = '01' THEN
                CAST(SUBSTRING(monthly_reporting_period,1,4) AS UNSIGNED) * 12
                + CAST(SUBSTRING(monthly_reporting_period,5,2) AS UNSIGNED)
            END) AS first_payoff_idx,
            MAX(
                CAST(SUBSTRING(monthly_reporting_period,1,4) AS UNSIGNED) * 12
                + CAST(SUBSTRING(monthly_reporting_period,5,2) AS UNSIGNED)
            ) AS last_observed_idx
        FROM freddie_performance
        GROUP BY loan_sequence_number
    )
    SELECT
        o.loan_sequence_number,
        o.vintage_year,
        o.credit_score,
        o.first_time_homebuyer_flag,
        o.mortgage_insurance_percentage,
        o.number_of_units,
        o.occupancy_status,
        o.original_cltv,
        o.original_dti_ratio,
        o.original_upb,
        o.original_ltv,
        o.original_interest_rate,
        o.channel,
        o.property_state,
        o.property_type,
        o.loan_purpose,
        o.original_loan_term,
        o.number_of_borrowers,
        o.super_conforming_flag,
        o.harp_indicator,
        o.interest_only_indicator,
        s.monthly_reporting_period AS snapshot_period,
        s.current_actual_upb,
        s.current_loan_delinquency_status,
        s.loan_age,
        s.current_interest_rate,
        s.estimated_ltv,
        e.first_credit_event_idx,
        e.first_payoff_idx,
        e.last_observed_idx
    FROM freddie_origination o
    JOIN snap_candidates s
      ON o.loan_sequence_number = s.loan_sequence_number
     AND s.rn = 1
    JOIN event_dates e
      ON o.loan_sequence_number = e.loan_sequence_number
    """
    df = pd.read_sql(sql, engine)
    df["snapshot_idx"] = period_index(df["snapshot_period"])
    df["horizon_end_idx"] = df["snapshot_idx"] + FORWARD_MONTHS

    # Keep only loans for which the full 12-month forward window is observable.
    # Store the horizon on the dataframe itself so index alignment remains exact
    # after filtering (avoids comparing against a stale pre-filter Series).
    df = df[
        df["last_observed_idx"].notna()
        & df["snapshot_idx"].notna()
        & (df["last_observed_idx"] >= df["horizon_end_idx"])
    ].copy()
    df.reset_index(drop=True, inplace=True)

    df["credit_event_12m"] = (
        df["first_credit_event_idx"].notna()
        & (df["first_credit_event_idx"] > df["snapshot_idx"])
        & (df["first_credit_event_idx"] <= df["horizon_end_idx"])
    ).astype(int)

    # Prepayment target excludes loans that credit-event first inside the horizon.
    payoff_in_horizon = (
        df["first_payoff_idx"].notna()
        & (df["first_payoff_idx"] > df["snapshot_idx"])
        & (df["first_payoff_idx"] <= df["horizon_end_idx"])
    )
    credit_before_payoff = (
        df["first_credit_event_idx"].notna()
        & df["first_payoff_idx"].notna()
        & (df["first_credit_event_idx"] <= df["first_payoff_idx"])
    )
    df["payoff_12m"] = (payoff_in_horizon & ~credit_before_payoff).astype(int)

    # Relative coupon proxy available from the dataset itself.
    med = df.groupby("snapshot_period")["current_interest_rate"].transform("median")
    df["rate_spread_to_period_median"] = df["current_interest_rate"] - med

    # Normalize delinquency to a compact numeric signal while retaining raw category as categorical.
    delin = df["current_loan_delinquency_status"].astype(str).str.upper()
    df["delinquency_numeric"] = pd.to_numeric(delin, errors="coerce")
    df.loc[delin.eq("R"), "delinquency_numeric"] = 999
    df.loc[delin.eq("XX"), "delinquency_numeric"] = np.nan

    return df


def build_preprocessor(numeric_features, categorical_features):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
    ])
    return ColumnTransformer([
        ("num", num, numeric_features),
        ("cat", cat, categorical_features),
    ])


def safe_metrics(y_true, prob):
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    y = pd.Series(y_true)
    if y.nunique() < 2:
        auc = np.nan
        ap = np.nan
    else:
        auc = roc_auc_score(y, prob)
        ap = average_precision_score(y, prob)
    return {
        "roc_auc": auc,
        "average_precision": ap,
        "brier_score": brier_score_loss(y, prob),
        "observed_event_rate": float(y.mean()),
        "mean_predicted_probability": float(np.mean(prob)),
        "calibration_ratio_pred_to_obs": float(np.mean(prob) / y.mean()) if y.mean() > 0 else np.nan,
    }
