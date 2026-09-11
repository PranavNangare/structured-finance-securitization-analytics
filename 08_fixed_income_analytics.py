
from pathlib import Path
import numpy as np
import pandas as pd

from sf_common import REPORT_DIR

FACE = 100_000_000.0
COUPON = 0.055
YIELD_ASSUMPTION = 0.06
TERM_YEARS = 5

CPR_FILE = REPORT_DIR / "prepayment_cpr_scenarios.csv"


def cpr_to_smm(cpr):
    return 1 - (1 - cpr) ** (1 / 12)


def build_cash_flows(face, coupon, term_years, cpr):
    """
    Stylized monthly amortizing cash-flow model.

    Assumptions:
    - level monthly scheduled principal
    - monthly coupon on beginning balance
    - prepayment applied to remaining balance after scheduled principal
    """
    months = term_years * 12
    monthly_coupon = coupon / 12
    smm = cpr_to_smm(cpr)

    scheduled_principal = face / months
    balance = face

    rows = []

    for month in range(1, months + 1):
        beginning_balance = balance

        scheduled = min(scheduled_principal, beginning_balance)

        remaining_after_scheduled = max(
            beginning_balance - scheduled,
            0.0,
        )

        prepayment = remaining_after_scheduled * smm

        if month == months:
            prepayment = remaining_after_scheduled

        interest = beginning_balance * monthly_coupon

        total_principal = scheduled + prepayment
        cash_flow = interest + total_principal

        ending_balance = max(
            beginning_balance - total_principal,
            0.0,
        )

        rows.append(
            {
                "month": month,
                "beginning_balance": beginning_balance,
                "scheduled_principal": scheduled,
                "prepayment": prepayment,
                "interest": interest,
                "total_cash_flow": cash_flow,
                "ending_balance": ending_balance,
            }
        )

        balance = ending_balance

        if balance <= 1e-8:
            break

    return pd.DataFrame(rows)


def price_from_cash_flows(cash_flows, annual_yield):
    monthly_yield = annual_yield / 12

    months = cash_flows["month"].to_numpy()
    flows = cash_flows["total_cash_flow"].to_numpy()

    discount = (1 + monthly_yield) ** months

    return float(np.sum(flows / discount))


def duration_convexity(cash_flows, annual_yield):
    monthly_yield = annual_yield / 12

    months = cash_flows["month"].to_numpy(dtype=float)
    flows = cash_flows["total_cash_flow"].to_numpy(dtype=float)

    pv = flows / ((1 + monthly_yield) ** months)
    price = pv.sum()

    macaulay_months = np.sum(months * pv) / price
    macaulay_years = macaulay_months / 12

    modified_years = macaulay_years / (1 + monthly_yield)

    convexity_months = (
        np.sum(
            months
            * (months + 1)
            * flows
            / ((1 + monthly_yield) ** (months + 2))
        )
        / price
    )

    convexity_years = convexity_months / (12 ** 2)

    return (
        float(macaulay_years),
        float(modified_years),
        float(convexity_years),
    )


def analyze_scenario(scenario_name, cpr):
    cash_flows = build_cash_flows(
        face=FACE,
        coupon=COUPON,
        term_years=TERM_YEARS,
        cpr=cpr,
    )

    price = price_from_cash_flows(
        cash_flows,
        YIELD_ASSUMPTION,
    )

    macaulay, modified, convexity = duration_convexity(
        cash_flows,
        YIELD_ASSUMPTION,
    )

    return {
        "scenario": scenario_name,
        "face": FACE,
        "coupon": COUPON,
        "yield_assumption": YIELD_ASSUMPTION,
        "term_years": TERM_YEARS,
        "cpr_assumption": cpr,
        "smm_assumption": cpr_to_smm(cpr),
        "price": price,
        "price_pct_of_par": price / FACE * 100,
        "macaulay_duration_years": macaulay,
        "modified_duration_years": modified,
        "convexity": convexity,
    }, cash_flows


def load_cpr_scenarios():
    if not CPR_FILE.exists():
        raise FileNotFoundError(
            f"Missing Step 5 scenario file:\n{CPR_FILE}"
        )

    df = pd.read_csv(CPR_FILE)

    required = {"scenario", "cpr_assumption"}
    missing = required.difference(df.columns)

    if missing:
        raise KeyError(
            f"Step 5 CPR file is missing columns: {sorted(missing)}"
        )

    df["cpr_assumption"] = pd.to_numeric(
        df["cpr_assumption"],
        errors="coerce",
    )

    df = df.dropna(subset=["cpr_assumption"]).copy()

    if df.empty:
        raise ValueError(
            "No valid CPR assumptions were found in the Step 5 file."
        )

    return df


def build_yield_sensitivity(base_cpr):
    yield_grid = [0.05, 0.055, 0.06, 0.065, 0.07]

    cash_flows = build_cash_flows(
        face=FACE,
        coupon=COUPON,
        term_years=TERM_YEARS,
        cpr=base_cpr,
    )

    rows = []

    for yld in yield_grid:
        price = price_from_cash_flows(
            cash_flows,
            yld,
        )

        macaulay, modified, convexity = duration_convexity(
            cash_flows,
            yld,
        )

        rows.append(
            {
                "yield_assumption": yld,
                "cpr_assumption": base_cpr,
                "price": price,
                "price_pct_of_par": price / FACE * 100,
                "macaulay_duration_years": macaulay,
                "modified_duration_years": modified,
                "convexity": convexity,
            }
        )

    return pd.DataFrame(rows)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nSTEP 7 — STYLIZED FIXED-INCOME ANALYTICS")
    print("=" * 60)

    scenarios = load_cpr_scenarios()

    summary_rows = []
    cash_flow_frames = []

    for _, row in scenarios.iterrows():
        result, cash_flows = analyze_scenario(
            row["scenario"],
            float(row["cpr_assumption"]),
        )

        summary_rows.append(result)

        cash_flows = cash_flows.copy()
        cash_flows.insert(0, "scenario", row["scenario"])
        cash_flow_frames.append(cash_flows)

    summary = pd.DataFrame(summary_rows)
    cash_flows_all = pd.concat(
        cash_flow_frames,
        ignore_index=True,
    )

    base_match = scenarios[
        scenarios["scenario"]
        .astype(str)
        .str.lower()
        .str.contains("base")
    ]

    if base_match.empty:
        base_cpr = float(
            scenarios["cpr_assumption"].median()
        )
    else:
        base_cpr = float(
            base_match.iloc[0]["cpr_assumption"]
        )

    yield_sensitivity = build_yield_sensitivity(
        base_cpr
    )

    summary_path = (
        REPORT_DIR
        / "fixed_income_cpr_scenario_summary.csv"
    )
    cashflow_path = (
        REPORT_DIR
        / "fixed_income_cpr_scenario_cashflows.csv"
    )
    yield_path = (
        REPORT_DIR
        / "fixed_income_yield_sensitivity.csv"
    )

    summary.to_csv(summary_path, index=False)
    cash_flows_all.to_csv(cashflow_path, index=False)
    yield_sensitivity.to_csv(yield_path, index=False)

    print("\nCPR SCENARIO SUMMARY")
    print(summary.to_string(index=False))

    print("\nBASE-CPR YIELD SENSITIVITY")
    print(yield_sensitivity.to_string(index=False))

    print("\nCreated:")
    print(f"  {summary_path}")
    print(f"  {cashflow_path}")
    print(f"  {yield_path}")

    print(
        "\nThese are stylized cash-flow assumptions, "
        "not actual Goldman or Freddie security outputs."
    )

    print("\nStep 7 complete.")


if __name__ == "__main__":
    main()
