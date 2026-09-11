from __future__ import annotations
from pathlib import Path
from urllib.parse import quote_plus
import pandas as pd
from sqlalchemy import create_engine, text

PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_DIR = PROJECT_ROOT / "reports"
PROFILE_DIR = REPORT_DIR / "profiles"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
PROFILE_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_NAME = "structured_finance"
MYSQL_HOST = "127.0.0.1"
MYSQL_PORT = 3306
MYSQL_USER = "root"
MYSQL_PASSWORD = "123@Pranav"  # local only; remove before GitHub

EXPECTED_YEARS = {2006, 2007, 2008, 2019, 2020, 2021, 2022}
EXPECTED_ORIG_ROWS = 350000
EXPECTED_PERF_ROWS = 17696387
EXPECTED_CMBS_ASSETS = 168
EXPECTED_CMBS_PROPERTIES = 168


def get_engine():
    password = quote_plus(MYSQL_PASSWORD)
    url = f"mysql+pymysql://{MYSQL_USER}:{password}@{MYSQL_HOST}:{MYSQL_PORT}/{DATABASE_NAME}"
    return create_engine(url, echo=False, pool_pre_ping=True, future=True)


def scalar(conn, sql):
    return conn.execute(text(sql)).scalar_one()


def one_row(conn, sql):
    return dict(conn.execute(text(sql)).mappings().one())


def df_query(conn, sql):
    return pd.read_sql(text(sql), conn)


def pct(n, d):
    return None if not d else round(100.0 * n / d, 4)


def add_check(rows, category, check_name, status, observed, threshold, details):
    rows.append({
        "category": category,
        "check_name": check_name,
        "status": status,
        "observed": observed,
        "threshold": threshold,
        "details": details,
    })


def run_validation():
    engine = get_engine()
    checks = []
    profiles = {}

    print("Structured Finance - Step 1 Data Validation")
    print("=" * 70)

    with engine.connect() as conn:
        orig_total = scalar(conn, "SELECT COUNT(*) FROM freddie_origination")
        perf_total = scalar(conn, "SELECT COUNT(*) FROM freddie_performance")

        orig_vintages = df_query(conn, """
            SELECT vintage_year, COUNT(*) AS loans
            FROM freddie_origination
            GROUP BY vintage_year
            ORDER BY vintage_year
        """)
        profiles["freddie_origination_by_vintage"] = orig_vintages

        perf_vintages = df_query(conn, """
            SELECT vintage_year,
                   COUNT(*) AS loan_month_rows,
                   COUNT(DISTINCT loan_sequence_number) AS unique_loans,
                   MIN(monthly_reporting_period) AS first_reporting_period,
                   MAX(monthly_reporting_period) AS last_reporting_period
            FROM freddie_performance
            GROUP BY vintage_year
            ORDER BY vintage_year
        """)
        profiles["freddie_performance_by_vintage"] = perf_vintages

        actual_years = set(orig_vintages["vintage_year"].astype(int).tolist())
        add_check(checks, "Freddie Origination", "Expected vintage coverage",
                  "PASS" if actual_years == EXPECTED_YEARS else "FAIL",
                  ",".join(map(str, sorted(actual_years))),
                  ",".join(map(str, sorted(EXPECTED_YEARS))),
                  "All selected vintages must be present.")
        add_check(checks, "Freddie Origination", "Total origination rows",
                  "PASS" if orig_total == EXPECTED_ORIG_ROWS else "WARN",
                  orig_total, EXPECTED_ORIG_ROWS,
                  "Checkpoint from completed Phase 1 load.")
        add_check(checks, "Freddie Performance", "Total performance rows",
                  "PASS" if perf_total == EXPECTED_PERF_ROWS else "WARN",
                  perf_total, EXPECTED_PERF_ROWS,
                  "Checkpoint from completed Phase 1 load.")

        orig_dupes = scalar(conn, """
            SELECT COUNT(*) FROM (
                SELECT loan_sequence_number
                FROM freddie_origination
                GROUP BY loan_sequence_number
                HAVING COUNT(*) > 1
            ) x
        """)
        perf_dupes = scalar(conn, """
            SELECT COUNT(*) FROM (
                SELECT loan_sequence_number, monthly_reporting_period
                FROM freddie_performance
                GROUP BY loan_sequence_number, monthly_reporting_period
                HAVING COUNT(*) > 1
            ) x
        """)
        unmatched = scalar(conn, """
            SELECT COUNT(DISTINCT p.loan_sequence_number)
            FROM freddie_performance p
            LEFT JOIN freddie_origination o
              ON p.loan_sequence_number = o.loan_sequence_number
            WHERE o.loan_sequence_number IS NULL
        """)
        add_check(checks, "Freddie Keys", "Duplicate origination loan IDs",
                  "PASS" if orig_dupes == 0 else "FAIL", orig_dupes, 0,
                  "Origination should be one row per loan.")
        add_check(checks, "Freddie Keys", "Duplicate performance loan-month keys",
                  "PASS" if perf_dupes == 0 else "FAIL", perf_dupes, 0,
                  "Performance key is loan_sequence_number + monthly_reporting_period.")
        add_check(checks, "Freddie Keys", "Unmatched performance loan IDs",
                  "PASS" if unmatched == 0 else "FAIL", unmatched, 0,
                  "Every performance loan should link to origination.")

        orig_quality = one_row(conn, """
            SELECT
              SUM(credit_score IS NULL) AS missing_fico,
              SUM(credit_score IS NOT NULL AND (credit_score < 300 OR credit_score > 850)) AS bad_fico,
              SUM(vantage_score_4 IS NULL) AS missing_vantage,
              SUM(vantage_score_4 IS NOT NULL AND (vantage_score_4 < 300 OR vantage_score_4 > 850)) AS bad_vantage,
              SUM(original_ltv IS NULL) AS missing_ltv,
              SUM(original_ltv IS NOT NULL AND (original_ltv < 0 OR original_ltv > 200)) AS bad_ltv,
              SUM(original_cltv IS NULL) AS missing_cltv,
              SUM(original_cltv IS NOT NULL AND (original_cltv < 0 OR original_cltv > 250)) AS bad_cltv,
              SUM(original_dti_ratio IS NULL) AS missing_dti,
              SUM(original_dti_ratio IS NOT NULL AND (original_dti_ratio < 0 OR original_dti_ratio > 100)) AS bad_dti,
              SUM(original_upb IS NULL) AS missing_upb,
              SUM(original_upb IS NOT NULL AND original_upb <= 0) AS bad_upb,
              SUM(original_interest_rate IS NULL) AS missing_rate,
              SUM(original_interest_rate IS NOT NULL AND (original_interest_rate <= 0 OR original_interest_rate > 25)) AS bad_rate
            FROM freddie_origination
        """)
        profiles["origination_quality"] = pd.DataFrame([orig_quality])

        for label, bad_key, missing_key in [
            ("FICO", "bad_fico", "missing_fico"),
            ("VantageScore 4.0", "bad_vantage", "missing_vantage"),
            ("Original LTV", "bad_ltv", "missing_ltv"),
            ("Original CLTV", "bad_cltv", "missing_cltv"),
            ("Original DTI", "bad_dti", "missing_dti"),
            ("Original UPB", "bad_upb", "missing_upb"),
            ("Original interest rate", "bad_rate", "missing_rate"),
        ]:
            bad = int(orig_quality[bad_key] or 0)
            missing = int(orig_quality[missing_key] or 0)
            add_check(checks, "Freddie Origination", f"{label} range",
                      "PASS" if bad == 0 else "WARN", bad, 0,
                      f"Missing={missing:,} ({pct(missing, orig_total)}%). Warnings require source-definition review.")

        perf_quality = one_row(conn, """
            SELECT
              SUM(current_actual_upb IS NULL) AS missing_current_upb,
              SUM(current_actual_upb IS NOT NULL AND current_actual_upb < 0) AS negative_current_upb,
              SUM(current_interest_rate IS NULL) AS missing_current_rate,
              SUM(current_interest_rate IS NOT NULL AND (current_interest_rate < 0 OR current_interest_rate > 25)) AS bad_current_rate,
              SUM(loan_age IS NULL) AS missing_loan_age,
              SUM(loan_age IS NOT NULL AND loan_age < 0) AS negative_loan_age,
              SUM(remaining_months_to_legal_maturity IS NOT NULL AND remaining_months_to_legal_maturity < 0) AS negative_remaining_term
            FROM freddie_performance
        """)
        profiles["performance_quality"] = pd.DataFrame([perf_quality])

        for label, key in [
            ("Negative current UPB", "negative_current_upb"),
            ("Current rate outside 0-25%", "bad_current_rate"),
            ("Negative loan age", "negative_loan_age"),
            ("Negative remaining term", "negative_remaining_term"),
        ]:
            observed = int(perf_quality[key] or 0)
            add_check(checks, "Freddie Performance", label,
                      "PASS" if observed == 0 else "WARN", observed, 0,
                      "Do not alter records automatically; inspect source definitions first.")

        profiles["delinquency_status"] = df_query(conn, """
            SELECT current_loan_delinquency_status, COUNT(*) AS loan_month_rows
            FROM freddie_performance
            GROUP BY current_loan_delinquency_status
            ORDER BY loan_month_rows DESC
        """)
        profiles["zero_balance_code"] = df_query(conn, """
            SELECT zero_balance_code, COUNT(*) AS loan_month_rows
            FROM freddie_performance
            GROUP BY zero_balance_code
            ORDER BY loan_month_rows DESC
        """)
        profiles["modification_flag"] = df_query(conn, """
            SELECT modification_flag, COUNT(*) AS loan_month_rows
            FROM freddie_performance
            GROUP BY modification_flag
            ORDER BY loan_month_rows DESC
        """)
        profiles["freddie_vintage_profile"] = df_query(conn, """
            SELECT vintage_year,
                   COUNT(*) AS loans,
                   ROUND(SUM(original_upb),2) AS original_upb,
                   ROUND(AVG(credit_score),2) AS avg_fico,
                   ROUND(AVG(vantage_score_4),2) AS avg_vantage_score_4,
                   ROUND(AVG(original_ltv),2) AS avg_ltv,
                   ROUND(AVG(original_cltv),2) AS avg_cltv,
                   ROUND(AVG(original_dti_ratio),2) AS avg_dti,
                   ROUND(AVG(original_interest_rate),4) AS avg_rate
            FROM freddie_origination
            GROUP BY vintage_year
            ORDER BY vintage_year
        """)

        cmbs_assets = scalar(conn, "SELECT COUNT(*) FROM sec_cmbs_loans")
        cmbs_props = scalar(conn, "SELECT COUNT(*) FROM sec_cmbs_properties")
        cmbs_loan_dupes = scalar(conn, """
            SELECT COUNT(*) FROM (
                SELECT deal_name, asset_number
                FROM sec_cmbs_loans
                GROUP BY deal_name, asset_number
                HAVING COUNT(*) > 1
            ) x
        """)
        cmbs_prop_dupes = scalar(conn, """
            SELECT COUNT(*) FROM (
                SELECT deal_name, asset_number, property_sequence
                FROM sec_cmbs_properties
                GROUP BY deal_name, asset_number, property_sequence
                HAVING COUNT(*) > 1
            ) x
        """)
        add_check(checks, "SEC CMBS", "CMBS asset rows",
                  "PASS" if cmbs_assets == EXPECTED_CMBS_ASSETS else "WARN",
                  cmbs_assets, EXPECTED_CMBS_ASSETS,
                  "Loaded-record checkpoint; hierarchy interpretation is audited separately.")
        add_check(checks, "SEC CMBS", "CMBS property rows",
                  "PASS" if cmbs_props == EXPECTED_CMBS_PROPERTIES else "WARN",
                  cmbs_props, EXPECTED_CMBS_PROPERTIES,
                  "Loaded-record checkpoint.")
        add_check(checks, "SEC CMBS", "Duplicate deal/asset keys",
                  "PASS" if cmbs_loan_dupes == 0 else "FAIL", cmbs_loan_dupes, 0,
                  "CMBS asset table key uniqueness.")
        add_check(checks, "SEC CMBS", "Duplicate property keys",
                  "PASS" if cmbs_prop_dupes == 0 else "FAIL", cmbs_prop_dupes, 0,
                  "CMBS deal + asset + property sequence uniqueness.")

        profiles["cmbs_by_deal"] = df_query(conn, """
            SELECT deal_name,
                   COUNT(*) AS asset_rows,
                   ROUND(SUM(original_loan_amount),2) AS total_original_loan_amount,
                   ROUND(SUM(ending_actual_balance),2) AS total_ending_actual_balance
            FROM sec_cmbs_loans
            GROUP BY deal_name
            ORDER BY deal_name
        """)
        profiles["cmbs_property_missingness"] = df_query(conn, """
            SELECT deal_name,
                   COUNT(*) AS property_rows,
                   SUM(valuation_amount IS NULL) AS missing_valuation,
                   SUM(physical_occupancy IS NULL) AS missing_occupancy,
                   SUM(net_operating_income IS NULL) AS missing_noi,
                   SUM(net_cash_flow IS NULL) AS missing_ncf,
                   SUM(dscr_noi IS NULL) AS missing_dscr_noi,
                   SUM(dscr_ncf IS NULL) AS missing_dscr_ncf
            FROM sec_cmbs_properties
            GROUP BY deal_name
            ORDER BY deal_name
        """)
        profiles["cmbs_property_type"] = df_query(conn, """
            SELECT deal_name, property_type_code,
                   COUNT(*) AS property_rows,
                   ROUND(SUM(valuation_amount),2) AS valuation_amount
            FROM sec_cmbs_properties
            GROUP BY deal_name, property_type_code
            ORDER BY deal_name, property_rows DESC
        """)
        profiles["cmbs_state_concentration"] = df_query(conn, """
            SELECT deal_name, property_state,
                   COUNT(*) AS property_rows,
                   ROUND(SUM(valuation_amount),2) AS valuation_amount
            FROM sec_cmbs_properties
            GROUP BY deal_name, property_state
            ORDER BY deal_name, valuation_amount DESC
        """)

    checks_df = pd.DataFrame(checks)
    checks_path = REPORT_DIR / "data_quality_report.csv"
    checks_df.to_csv(checks_path, index=False)

    for name, df in profiles.items():
        df.to_csv(PROFILE_DIR / f"{name}.csv", index=False)

    if (checks_df["status"] == "FAIL").any():
        overall = "FAIL"
    elif (checks_df["status"] == "WARN").any():
        overall = "PASS WITH WARNINGS"
    else:
        overall = "PASS"

    print("\nVALIDATION SUMMARY")
    print("-" * 70)
    print(checks_df[["category", "check_name", "status", "observed"]].to_string(index=False))
    print(f"\nOverall: {overall}")
    print(f"Main report: {checks_path}")
    print(f"Profiles: {PROFILE_DIR}")
    return checks_df, profiles


if __name__ == "__main__":
    run_validation()
