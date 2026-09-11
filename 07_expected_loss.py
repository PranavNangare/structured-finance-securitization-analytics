import numpy as np
import pandas as pd
from sqlalchemy import text

from sf_common import REPORT_DIR, TEST_VINTAGE, get_engine, weighted_average


def pooled_historical_lgd(engine):
    # Use realized event records from vintages before the scoring vintage.
    sql = text("""
        SELECT
            SUM(CASE WHEN actual_loss > 0 THEN actual_loss ELSE 0 END) AS actual_loss,
            SUM(CASE WHEN zero_balance_removal_upb > 0 THEN zero_balance_removal_upb ELSE 0 END) AS removal_upb,
            COUNT(*) AS event_records
        FROM freddie_performance
        WHERE vintage_year < :test_vintage
          AND zero_balance_code IN ('02','03','09')
          AND zero_balance_removal_upb > 0
    """)
    x = pd.read_sql(sql, engine, params={"test_vintage": TEST_VINTAGE}).iloc[0]
    if x["removal_upb"] <= 0:
        raise RuntimeError("No usable historical removal UPB for LGD estimation.")
    lgd = float(x["actual_loss"] / x["removal_upb"])
    return float(np.clip(lgd, 0.0, 1.0)), int(x["event_records"])


def main():
    engine = get_engine()
    scores_path = REPORT_DIR / "credit_risk_scores_2022.csv"
    if not scores_path.exists():
        raise FileNotFoundError("Run 05_credit_risk_model.py first.")

    df = pd.read_csv(scores_path)
    df = df[(df.current_actual_upb > 0) & df.pd_12m.notna()].copy()
    lgd, lgd_event_records = pooled_historical_lgd(engine)

    df["ead"] = df["current_actual_upb"]
    df["lgd"] = lgd
    df["expected_loss_12m"] = df["pd_12m"] * df["lgd"] * df["ead"]

    ead = df["ead"].sum()
    el = df["expected_loss_12m"].sum()
    summary = pd.DataFrame([{
        "vintage_year": TEST_VINTAGE,
        "loans": len(df),
        "ead": ead,
        "expected_loss_12m": el,
        "weighted_pd_12m": weighted_average(df.pd_12m, df.ead),
        "historical_lgd": lgd,
        "lgd_event_records": lgd_event_records,
        "expected_loss_rate_12m": el / ead if ead else np.nan,
    }])

    df[["loan_sequence_number", "vintage_year", "ead", "pd_12m", "lgd", "expected_loss_12m"]].to_csv(
        REPORT_DIR / "expected_loss_loan_level_2022.csv", index=False
    )
    summary.to_csv(REPORT_DIR / "expected_loss_summary.csv", index=False)

    print("\nCORRECTED 12-MONTH EXPECTED LOSS")
    print(summary.to_string(index=False))
    print("\nEL = 12-month PD × realized historical LGD × snapshot EAD.")
    print("LGD is pooled from actual Freddie credit-event removal records before 2022.")
    print("\nStep 6 complete.")


if __name__ == "__main__":
    main()
