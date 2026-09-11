import numpy as np
import pandas as pd
from sf_common import REPORT_DIR, weighted_average

TARGET_UPB = 250_000_000.0
TARGET_TOLERANCE = 0.01
MIN_WA_FICO = 740.0
MAX_WA_LTV = 80.0
MAX_WA_DTI = 42.0
MAX_STATE_SHARE = 0.20


def prospective_metric(selected, row, col):
    vals = pd.concat([selected[[col, "current_actual_upb"]], row[[col, "current_actual_upb"]].to_frame().T])
    return weighted_average(vals[col], vals["current_actual_upb"])


def main():
    path = REPORT_DIR / "credit_risk_scores_2022.csv"
    if not path.exists():
        raise FileNotFoundError("Run 05_credit_risk_model.py first.")
    df = pd.read_csv(path)
    need = ["loan_sequence_number", "current_actual_upb", "credit_score", "original_ltv", "original_dti_ratio", "property_state", "pd_12m"]
    df = df.dropna(subset=need).copy()
    df = df[df.current_actual_upb > 0].copy()

    # Risk-quality rank; this is a transparent heuristic, not an issuer optimization claim.
    df["rank_score"] = (
        df["pd_12m"].rank(pct=True)
        + (1 - df["credit_score"].rank(pct=True))
        + df["original_ltv"].rank(pct=True)
        + df["original_dti_ratio"].rank(pct=True)
    )
    df = df.sort_values(["rank_score", "pd_12m", "current_actual_upb"]).reset_index(drop=True)

    selected_idx = []
    selected = df.iloc[0:0].copy()
    state_upb = {}
    state_cap_dollars = MAX_STATE_SHARE * TARGET_UPB
    upper_target = TARGET_UPB * (1 + TARGET_TOLERANCE)
    lower_target = TARGET_UPB * (1 - TARGET_TOLERANCE)

    for idx, row in df.iterrows():
        upb = float(row.current_actual_upb)
        if selected.current_actual_upb.sum() + upb > upper_target:
            continue
        state = row.property_state
        if state_upb.get(state, 0.0) + upb > state_cap_dollars:
            continue

        if len(selected) == 0:
            p_fico, p_ltv, p_dti = row.credit_score, row.original_ltv, row.original_dti_ratio
        else:
            p_fico = prospective_metric(selected, row, "credit_score")
            p_ltv = prospective_metric(selected, row, "original_ltv")
            p_dti = prospective_metric(selected, row, "original_dti_ratio")
        if p_fico < MIN_WA_FICO or p_ltv > MAX_WA_LTV or p_dti > MAX_WA_DTI:
            continue

        selected_idx.append(idx)
        selected = df.loc[selected_idx].copy()
        state_upb[state] = state_upb.get(state, 0.0) + upb
        if selected.current_actual_upb.sum() >= lower_target:
            break

    selected_upb = selected.current_actual_upb.sum()
    if selected_upb < lower_target:
        raise RuntimeError(
            f"Could not fill at least 99% of the ${TARGET_UPB:,.0f} target under constraints; "
            f"selected ${selected_upb:,.0f}. Review feasibility rather than reporting a failed pool as complete."
        )

    state_summary = selected.groupby("property_state", as_index=False)["current_actual_upb"].sum()
    state_summary["pool_share"] = state_summary.current_actual_upb / selected_upb
    max_state_share = state_summary.pool_share.max()

    summary = pd.DataFrame([{
        "selected_loans": len(selected),
        "selected_upb": selected_upb,
        "target_upb": TARGET_UPB,
        "target_fill_pct": selected_upb / TARGET_UPB,
        "weighted_fico": weighted_average(selected.credit_score, selected.current_actual_upb),
        "weighted_ltv": weighted_average(selected.original_ltv, selected.current_actual_upb),
        "weighted_dti": weighted_average(selected.original_dti_ratio, selected.current_actual_upb),
        "weighted_pd_12m": weighted_average(selected.pd_12m, selected.current_actual_upb),
        "max_state_share": max_state_share,
        "constraint_min_fico": MIN_WA_FICO,
        "constraint_max_ltv": MAX_WA_LTV,
        "constraint_max_dti": MAX_WA_DTI,
        "constraint_max_state_share": MAX_STATE_SHARE,
        "target_met": lower_target <= selected_upb <= upper_target,
        "fico_constraint_met": weighted_average(selected.credit_score, selected.current_actual_upb) >= MIN_WA_FICO,
        "ltv_constraint_met": weighted_average(selected.original_ltv, selected.current_actual_upb) <= MAX_WA_LTV,
        "dti_constraint_met": weighted_average(selected.original_dti_ratio, selected.current_actual_upb) <= MAX_WA_DTI,
        "state_constraint_met": max_state_share <= MAX_STATE_SHARE + 1e-12,
    }])

    selected.to_csv(REPORT_DIR / "optimized_pool_2022.csv", index=False)
    state_summary.sort_values("pool_share", ascending=False).to_csv(REPORT_DIR / "optimized_pool_state_concentration.csv", index=False)
    summary.to_csv(REPORT_DIR / "optimized_pool_summary.csv", index=False)
    print("\nCORRECTED $250M POOL SELECTION")
    print(summary.to_string(index=False))
    print("\nState caps are enforced against the target pool dollars, not the temporarily selected balance.")
    print("This is a transparent portfolio-selection heuristic, not an actual issuer pool.")
    print("\nStep 8 complete.")


if __name__ == "__main__":
    main()
