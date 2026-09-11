
from pathlib import Path
import numpy as np
import pandas as pd

from sf_common import REPORT_DIR

FRED_SERIES = "MORTGAGE30US"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
FRED_CACHE = REPORT_DIR / "fred_mortgage30us_monthly.csv"


def parse_month(series):
    raw = series.astype(str).str.strip()
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    six_digit = raw.str.fullmatch(r"\d{6}")
    if six_digit.any():
        out.loc[six_digit] = pd.to_datetime(
            raw.loc[six_digit] + "01",
            format="%Y%m%d",
            errors="coerce",
        )

    if (~six_digit).any():
        parsed = pd.to_datetime(raw.loc[~six_digit], errors="coerce")
        out.loc[~six_digit] = parsed.dt.to_period("M").dt.to_timestamp()

    return out


def load_monthly_market_rates():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if FRED_CACHE.exists():
        market = pd.read_csv(FRED_CACHE)

        # Support caches created by both the earlier and cleaned Step 5 versions.
        if "calendar_month" not in market.columns:
            if "snapshot_month" in market.columns:
                market = market.rename(
                    columns={"snapshot_month": "calendar_month"}
                )
            else:
                raise KeyError(
                    "Cached FRED file is missing both 'calendar_month' "
                    "and 'snapshot_month'."
                )

        market["calendar_month"] = pd.to_datetime(
            market["calendar_month"], errors="coerce"
        )
        market["market_mortgage_rate_30y"] = pd.to_numeric(
            market["market_mortgage_rate_30y"], errors="coerce"
        )

        market = market.dropna(
            subset=["calendar_month", "market_mortgage_rate_30y"]
        )

        if not market.empty:
            print(f"Using cached PMMS/FRED monthly rates: {FRED_CACHE}")
            return market[
                ["calendar_month", "market_mortgage_rate_30y"]
            ]

    print(f"Downloading FRED {FRED_SERIES} mortgage-rate series...")

    weekly = pd.read_csv(FRED_URL)

    date_col = "DATE" if "DATE" in weekly.columns else weekly.columns[0]

    if FRED_SERIES not in weekly.columns:
        raise KeyError(
            f"Expected column '{FRED_SERIES}' was not found in the FRED file."
        )

    weekly["date"] = pd.to_datetime(weekly[date_col], errors="coerce")
    weekly["rate"] = pd.to_numeric(
        weekly[FRED_SERIES].replace(".", np.nan),
        errors="coerce",
    )

    weekly = weekly.dropna(subset=["date", "rate"]).copy()

    weekly["calendar_month"] = (
        weekly["date"].dt.to_period("M").dt.to_timestamp()
    )

    market = (
        weekly.groupby("calendar_month", as_index=False)["rate"]
        .mean()
        .rename(columns={"rate": "market_mortgage_rate_30y"})
        .sort_values("calendar_month")
    )

    market.to_csv(FRED_CACHE, index=False)

    print(
        f"Cached {len(market):,} monthly market-rate observations to: "
        f"{FRED_CACHE}"
    )

    return market


def find_step3_cpr_file():
    """
    Find the Step 3 monthly SMM/CPR output by inspecting CSV columns.
    Only known downstream/model output files are excluded.
    """
    excluded_files = {
        "fred_mortgage30us_monthly.csv",
        "prepayment_realized_cpr_with_pmms.csv",
        "prepayment_rate_regime_summary.csv",
        "prepayment_cpr_scenarios.csv",
        "prepayment_rate_regime_thresholds.csv",
        "prepayment_oot_vintage_metrics.csv",
        "prepayment_oot_pooled_metrics.csv",
        "prepayment_oot_predictions.csv",
        "prepayment_model_metrics.csv",
        "prepayment_scores_2022.csv",
        "prepayment_market_rate_features.csv",
        "modeling_snapshot_12m.csv",
    }

    candidates = []

    for path in REPORT_DIR.glob("*.csv"):
        if path.name.lower() in excluded_files:
            continue

        try:
            sample = pd.read_csv(path, nrows=20)
        except Exception:
            continue

        columns = [str(c).strip().lower() for c in sample.columns]

        has_cpr = any("cpr" in c for c in columns)
        has_smm = any("smm" in c for c in columns)
        has_period = any(
            any(
                term in c
                for term in [
                    "period",
                    "month",
                    "date",
                    "reporting",
                ]
            )
            for c in columns
        )

        if has_cpr and has_smm and has_period:
            candidates.append(path)

    if not candidates:
        available = sorted(path.name for path in REPORT_DIR.glob("*.csv"))
        raise FileNotFoundError(
            "Could not identify the Step 3 monthly SMM/CPR CSV by its columns.\n"
            f"Reports folder: {REPORT_DIR}\n"
            "CSV files currently present:\n  - "
            + "\n  - ".join(available)
        )

    def score(path):
        name = path.name.lower()
        return (
            10 * ("cpr" in name)
            + 10 * ("smm" in name)
            + 5 * ("monthly" in name)
            + 5 * ("prepay" in name)
            + 3 * ("rate" in name)
        )

    candidates.sort(key=score, reverse=True)
    chosen = candidates[0]

    print(f"Using Step 3 realized SMM/CPR file: {chosen}")
    return chosen


def find_column(columns, keyword):
    for col in columns:
        if keyword in str(col).strip().lower():
            return col
    return None


def load_realized_cpr():
    path = find_step3_cpr_file()
    print(f"Using Step 3 realized SMM/CPR file: {path}")

    df = pd.read_csv(path, low_memory=False)

    cpr_col = find_column(df.columns, "cpr")
    smm_col = find_column(df.columns, "smm")
    vintage_col = find_column(df.columns, "vintage")

    period_col = None
    for col in df.columns:
        low = str(col).strip().lower()
        if "reporting" in low and "period" in low:
            period_col = col
            break

    if period_col is None:
        for col in df.columns:
            low = str(col).strip().lower()
            if any(term in low for term in ["period", "month", "date"]):
                period_col = col
                break

    if cpr_col is None or smm_col is None or period_col is None:
        raise KeyError(
            "Could not identify CPR, SMM, and reporting-period columns "
            f"in {path.name}."
        )

    out = pd.DataFrame()

    out["calendar_month"] = parse_month(df[period_col])
    out["realized_cpr"] = pd.to_numeric(df[cpr_col], errors="coerce")
    out["realized_smm"] = pd.to_numeric(df[smm_col], errors="coerce")

    if vintage_col is not None:
        out["vintage_year"] = pd.to_numeric(
            df[vintage_col], errors="coerce"
        )
    else:
        out["vintage_year"] = np.nan

    out = out.dropna(
        subset=["calendar_month", "realized_cpr", "realized_smm"]
    ).copy()

    # Support either decimal form (0.08) or percentage form (8.0).
    if out["realized_cpr"].median() > 1:
        out["realized_cpr"] = out["realized_cpr"] / 100

    if out["realized_smm"].median() > 1:
        out["realized_smm"] = out["realized_smm"] / 100

    out["realized_cpr"] = out["realized_cpr"].clip(0, 1)
    out["realized_smm"] = out["realized_smm"].clip(0, 1)

    return out


def build_rate_regime_analysis(realized, market):
    merged = realized.merge(
        market,
        on="calendar_month",
        how="inner",
        validate="many_to_one",
    )

    if merged.empty:
        raise RuntimeError(
            "No overlapping months were found between realized CPR data "
            "and the PMMS/FRED mortgage-rate series."
        )

    low_cut = merged["market_mortgage_rate_30y"].quantile(1 / 3)
    high_cut = merged["market_mortgage_rate_30y"].quantile(2 / 3)

    merged["rate_regime"] = np.select(
        [
            merged["market_mortgage_rate_30y"] <= low_cut,
            merged["market_mortgage_rate_30y"] <= high_cut,
        ],
        [
            "Low Market Rate",
            "Mid Market Rate",
        ],
        default="High Market Rate",
    )

    regime_summary = (
        merged.groupby("rate_regime", as_index=False)
        .agg(
            observations=("realized_cpr", "size"),
            months=("calendar_month", "nunique"),
            average_market_rate=("market_mortgage_rate_30y", "mean"),
            average_realized_cpr=("realized_cpr", "mean"),
            median_realized_cpr=("realized_cpr", "median"),
            average_realized_smm=("realized_smm", "mean"),
        )
    )

    return merged, regime_summary


def build_cpr_scenarios(realized_with_rates):
    cpr = realized_with_rates["realized_cpr"].dropna()

    low_cpr = float(cpr.quantile(0.25))
    base_cpr = float(cpr.quantile(0.50))
    high_cpr = float(cpr.quantile(0.75))

    scenarios = pd.DataFrame(
        [
            {
                "scenario": "Low Prepayment",
                "cpr_assumption": low_cpr,
                "smm_assumption": 1 - (1 - low_cpr) ** (1 / 12),
                "basis": "25th percentile of realized monthly CPR",
            },
            {
                "scenario": "Base Prepayment",
                "cpr_assumption": base_cpr,
                "smm_assumption": 1 - (1 - base_cpr) ** (1 / 12),
                "basis": "Median of realized monthly CPR",
            },
            {
                "scenario": "High Prepayment",
                "cpr_assumption": high_cpr,
                "smm_assumption": 1 - (1 - high_cpr) ** (1 / 12),
                "basis": "75th percentile of realized monthly CPR",
            },
        ]
    )

    return scenarios


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nSTEP 5 — PREPAYMENT ANALYTICS & CPR SCENARIOS")
    print("=" * 60)

    market = load_monthly_market_rates()
    realized = load_realized_cpr()

    realized_with_rates, regime_summary = build_rate_regime_analysis(
        realized,
        market,
    )

    scenarios = build_cpr_scenarios(realized_with_rates)

    realized_path = REPORT_DIR / "prepayment_realized_cpr_with_pmms.csv"
    regime_path = REPORT_DIR / "prepayment_rate_regime_summary.csv"
    scenario_path = REPORT_DIR / "prepayment_cpr_scenarios.csv"

    realized_with_rates.to_csv(realized_path, index=False)
    regime_summary.to_csv(regime_path, index=False)
    scenarios.to_csv(scenario_path, index=False)

    print("\nPMMS RATE-REGIME SUMMARY")
    print(regime_summary.to_string(index=False))

    print("\nFINAL CPR SCENARIOS")
    print(scenarios.to_string(index=False))

    print("\nCreated:")
    print(f"  {realized_path}")
    print(f"  {regime_path}")
    print(f"  {scenario_path}")

    print(
        "\nUse prepayment_cpr_scenarios.csv as the prepayment input "
        "for fixed-income and cash-flow analysis."
    )

    print("\nStep 5 complete.")


if __name__ == "__main__":
    main()
