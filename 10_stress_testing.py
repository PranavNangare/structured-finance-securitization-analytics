import numpy as np
import pandas as pd
from sf_common import REPORT_DIR, weighted_average

SCENARIOS = {
    "Base": (1.00, 1.00),
    "Moderate Stress": (1.50, 1.20),
    "Severe Stress": (2.00, 1.40),
}


def main():
    path = REPORT_DIR / "expected_loss_loan_level_2022.csv"
    if not path.exists():
        raise FileNotFoundError("Run 07_expected_loss.py first.")
    df = pd.read_csv(path)
    rows = []
    for name, (pd_mult, lgd_mult) in SCENARIOS.items():
        x = df.copy()
        x["pd_stressed"] = np.clip(x.pd_12m * pd_mult, 0, 1)
        x["lgd_stressed"] = np.clip(x.lgd * lgd_mult, 0, 1)
        x["expected_loss_stressed"] = x.ead * x.pd_stressed * x.lgd_stressed
        ead = x.ead.sum()
        rows.append({
            "scenario": name,
            "vintage_year": int(x.vintage_year.iloc[0]),
            "ead": ead,
            "weighted_pd_stressed": weighted_average(x.pd_stressed, x.ead),
            "weighted_lgd_stressed": weighted_average(x.lgd_stressed, x.ead),
            "expected_loss_stressed": x.expected_loss_stressed.sum(),
            "expected_loss_rate_stressed": x.expected_loss_stressed.sum() / ead if ead else np.nan,
            "pd_multiplier": pd_mult,
            "lgd_multiplier": lgd_mult,
        })
    out = pd.DataFrame(rows)
    out.to_csv(REPORT_DIR / "stress_test_summary.csv", index=False)
    print("\nCORRECTED STRESS TEST")
    print(out.to_string(index=False))
    print("\nScenario multipliers are explicit assumptions applied to the corrected 12-month PD/LGD base.")
    print("\nStep 9 complete.")


if __name__ == "__main__":
    main()
