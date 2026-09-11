from __future__ import annotations

from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy import create_engine, text


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_NAME = "structured_finance"
MYSQL_HOST = "127.0.0.1"
MYSQL_PORT = 3306
MYSQL_USER = "root"

# Local only. Remove before GitHub.
MYSQL_PASSWORD = "123@Pranav"


def get_engine():
    password = quote_plus(MYSQL_PASSWORD)
    return create_engine(
        f"mysql+pymysql://{MYSQL_USER}:{password}"
        f"@{MYSQL_HOST}:{MYSQL_PORT}/{DATABASE_NAME}",
        pool_pre_ping=True,
        future=True,
    )


def df_query(conn, sql: str) -> pd.DataFrame:
    return pd.read_sql(text(sql), conn)


def add_share(df: pd.DataFrame, amount_col: str, group_col: str = "vintage_year"):
    """Add within-group percentage shares without altering source data."""
    out = df.copy()
    totals = out.groupby(group_col)[amount_col].transform("sum")
    out["upb_share_pct"] = (100 * out[amount_col] / totals).round(2)
    return out


def create_clean_view(conn):
    conn.execute(text("DROP VIEW IF EXISTS vw_freddie_origination_clean"))
    conn.execute(text("""
        CREATE VIEW vw_freddie_origination_clean AS
        SELECT
            loan_sequence_number,
            vintage_year,
            CASE
                WHEN credit_score = 9999 THEN NULL
                WHEN credit_score BETWEEN 300 AND 850 THEN credit_score
                ELSE NULL
            END AS credit_score_clean,
            CASE
                WHEN vantage_score_4 = 9999 THEN NULL
                WHEN vantage_score_4 BETWEEN 300 AND 850 THEN vantage_score_4
                ELSE NULL
            END AS vantage_score_4_clean,
            first_payment_date,
            first_time_homebuyer_flag,
            maturity_date,
            msa,
            mortgage_insurance_percentage,
            number_of_units,
            occupancy_status,
            CASE WHEN original_cltv = 999 THEN NULL ELSE original_cltv END AS original_cltv_clean,
            original_dti_ratio,
            original_upb,
            CASE WHEN original_ltv = 999 THEN NULL ELSE original_ltv END AS original_ltv_clean,
            original_interest_rate,
            channel,
            prepayment_penalty_mortgage_flag,
            amortization_type,
            property_state,
            property_type,
            postal_code,
            loan_purpose,
            original_loan_term,
            number_of_borrowers,
            seller_name,
            super_conforming_flag,
            pre_harp_loan_sequence_number,
            special_eligibility_program,
            harp_indicator,
            property_valuation_method,
            interest_only_indicator,
            source_file
        FROM freddie_origination
    """))
    conn.commit()


def run_step2():
    engine = get_engine()

    print("Structured Finance - Step 2 Collateral Analytics")
    print("=" * 70)

    with engine.connect() as conn:
        create_clean_view(conn)
        print("Created analytical view: vw_freddie_origination_clean")

        # ----------------------------------------------------
        # FREDDIE PORTFOLIO METRICS
        # ----------------------------------------------------
        vintage_summary = df_query(conn, """
            SELECT
                vintage_year,
                COUNT(*) AS loans,
                ROUND(SUM(original_upb), 2) AS total_original_upb,
                ROUND(AVG(original_upb), 2) AS average_original_balance,

                ROUND(
                    SUM(CASE WHEN original_interest_rate IS NOT NULL
                             THEN original_upb * original_interest_rate END)
                    / NULLIF(SUM(CASE WHEN original_interest_rate IS NOT NULL
                                      THEN original_upb ELSE 0 END), 0), 4
                ) AS wac,

                ROUND(
                    SUM(CASE WHEN credit_score_clean IS NOT NULL
                             THEN original_upb * credit_score_clean END)
                    / NULLIF(SUM(CASE WHEN credit_score_clean IS NOT NULL
                                      THEN original_upb ELSE 0 END), 0), 2
                ) AS weighted_fico,

                ROUND(
                    SUM(CASE WHEN original_ltv_clean IS NOT NULL
                             THEN original_upb * original_ltv_clean END)
                    / NULLIF(SUM(CASE WHEN original_ltv_clean IS NOT NULL
                                      THEN original_upb ELSE 0 END), 0), 2
                ) AS weighted_ltv,

                ROUND(
                    SUM(CASE WHEN original_cltv_clean IS NOT NULL
                             THEN original_upb * original_cltv_clean END)
                    / NULLIF(SUM(CASE WHEN original_cltv_clean IS NOT NULL
                                      THEN original_upb ELSE 0 END), 0), 2
                ) AS weighted_cltv,

                ROUND(
                    SUM(CASE WHEN original_dti_ratio IS NOT NULL
                             THEN original_upb * original_dti_ratio END)
                    / NULLIF(SUM(CASE WHEN original_dti_ratio IS NOT NULL
                                      THEN original_upb ELSE 0 END), 0), 2
                ) AS weighted_dti

            FROM vw_freddie_origination_clean
            GROUP BY vintage_year
            ORDER BY vintage_year
        """)

        # Overall portfolio metrics.
        portfolio_summary = df_query(conn, """
            SELECT
                COUNT(*) AS loans,
                ROUND(SUM(original_upb), 2) AS total_original_upb,
                ROUND(AVG(original_upb), 2) AS average_original_balance,

                ROUND(
                    SUM(CASE WHEN original_interest_rate IS NOT NULL
                             THEN original_upb * original_interest_rate END)
                    / NULLIF(SUM(CASE WHEN original_interest_rate IS NOT NULL
                                      THEN original_upb ELSE 0 END), 0), 4
                ) AS wac,

                ROUND(
                    SUM(CASE WHEN credit_score_clean IS NOT NULL
                             THEN original_upb * credit_score_clean END)
                    / NULLIF(SUM(CASE WHEN credit_score_clean IS NOT NULL
                                      THEN original_upb ELSE 0 END), 0), 2
                ) AS weighted_fico,

                ROUND(
                    SUM(CASE WHEN original_ltv_clean IS NOT NULL
                             THEN original_upb * original_ltv_clean END)
                    / NULLIF(SUM(CASE WHEN original_ltv_clean IS NOT NULL
                                      THEN original_upb ELSE 0 END), 0), 2
                ) AS weighted_ltv,

                ROUND(
                    SUM(CASE WHEN original_cltv_clean IS NOT NULL
                             THEN original_upb * original_cltv_clean END)
                    / NULLIF(SUM(CASE WHEN original_cltv_clean IS NOT NULL
                                      THEN original_upb ELSE 0 END), 0), 2
                ) AS weighted_cltv,

                ROUND(
                    SUM(CASE WHEN original_dti_ratio IS NOT NULL
                             THEN original_upb * original_dti_ratio END)
                    / NULLIF(SUM(CASE WHEN original_dti_ratio IS NOT NULL
                                      THEN original_upb ELSE 0 END), 0), 2
                ) AS weighted_dti

            FROM vw_freddie_origination_clean
        """)

        fico = df_query(conn, """
            SELECT
                vintage_year,
                CASE
                    WHEN credit_score_clean IS NULL THEN 'Missing'
                    WHEN credit_score_clean < 620 THEN '<620'
                    WHEN credit_score_clean < 660 THEN '620-659'
                    WHEN credit_score_clean < 700 THEN '660-699'
                    WHEN credit_score_clean < 740 THEN '700-739'
                    WHEN credit_score_clean < 780 THEN '740-779'
                    ELSE '780+'
                END AS fico_band,
                COUNT(*) AS loans,
                ROUND(SUM(original_upb), 2) AS original_upb
            FROM vw_freddie_origination_clean
            GROUP BY vintage_year, fico_band
            ORDER BY vintage_year, fico_band
        """)
        fico = add_share(fico, "original_upb")

        ltv = df_query(conn, """
            SELECT
                vintage_year,
                CASE
                    WHEN original_ltv_clean IS NULL THEN 'Missing'
                    WHEN original_ltv_clean <= 60 THEN '<=60'
                    WHEN original_ltv_clean <= 70 THEN '60.01-70'
                    WHEN original_ltv_clean <= 80 THEN '70.01-80'
                    WHEN original_ltv_clean <= 90 THEN '80.01-90'
                    WHEN original_ltv_clean <= 95 THEN '90.01-95'
                    WHEN original_ltv_clean <= 100 THEN '95.01-100'
                    ELSE '>100'
                END AS ltv_band,
                COUNT(*) AS loans,
                ROUND(SUM(original_upb), 2) AS original_upb
            FROM vw_freddie_origination_clean
            GROUP BY vintage_year, ltv_band
            ORDER BY vintage_year, ltv_band
        """)
        ltv = add_share(ltv, "original_upb")

        dti = df_query(conn, """
            SELECT
                vintage_year,
                CASE
                    WHEN original_dti_ratio IS NULL THEN 'Missing'
                    WHEN original_dti_ratio <= 30 THEN '<=30'
                    WHEN original_dti_ratio <= 40 THEN '30.01-40'
                    WHEN original_dti_ratio <= 45 THEN '40.01-45'
                    WHEN original_dti_ratio <= 50 THEN '45.01-50'
                    ELSE '>50'
                END AS dti_band,
                COUNT(*) AS loans,
                ROUND(SUM(original_upb), 2) AS original_upb
            FROM vw_freddie_origination_clean
            GROUP BY vintage_year, dti_band
            ORDER BY vintage_year, dti_band
        """)
        dti = add_share(dti, "original_upb")

        state = df_query(conn, """
            SELECT
                vintage_year,
                property_state,
                COUNT(*) AS loans,
                ROUND(SUM(original_upb), 2) AS original_upb
            FROM vw_freddie_origination_clean
            GROUP BY vintage_year, property_state
            ORDER BY vintage_year, original_upb DESC
        """)
        state = add_share(state, "original_upb")

        purpose = df_query(conn, """
            SELECT
                vintage_year,
                loan_purpose,
                COUNT(*) AS loans,
                ROUND(SUM(original_upb), 2) AS original_upb
            FROM vw_freddie_origination_clean
            GROUP BY vintage_year, loan_purpose
            ORDER BY vintage_year, original_upb DESC
        """)
        purpose = add_share(purpose, "original_upb")

        occupancy = df_query(conn, """
            SELECT
                vintage_year,
                occupancy_status,
                COUNT(*) AS loans,
                ROUND(SUM(original_upb), 2) AS original_upb
            FROM vw_freddie_origination_clean
            GROUP BY vintage_year, occupancy_status
            ORDER BY vintage_year, original_upb DESC
        """)
        occupancy = add_share(occupancy, "original_upb")

        property_type = df_query(conn, """
            SELECT
                vintage_year,
                property_type,
                COUNT(*) AS loans,
                ROUND(SUM(original_upb), 2) AS original_upb
            FROM vw_freddie_origination_clean
            GROUP BY vintage_year, property_type
            ORDER BY vintage_year, original_upb DESC
        """)
        property_type = add_share(property_type, "original_upb")

        # ----------------------------------------------------
        # SEC CMBS ANALYTICS
        # ----------------------------------------------------
        cmbs_deal = df_query(conn, """
            SELECT
                deal_name,
                COUNT(*) AS asset_rows,
                SUM(original_loan_amount IS NOT NULL) AS populated_loan_records,
                ROUND(SUM(original_loan_amount), 2) AS original_loan_amount,
                ROUND(SUM(ending_actual_balance), 2) AS ending_actual_balance,

                ROUND(
                    SUM(CASE WHEN original_loan_amount IS NOT NULL
                             AND original_interest_rate IS NOT NULL
                             THEN original_loan_amount * original_interest_rate END)
                    / NULLIF(SUM(CASE WHEN original_interest_rate IS NOT NULL
                                      THEN original_loan_amount ELSE 0 END), 0), 4
                ) AS original_balance_weighted_coupon,

                SUM(CASE WHEN LOWER(interest_only_indicator) = 'true' THEN 1 ELSE 0 END) AS io_yes_rows,
                SUM(CASE WHEN LOWER(balloon_indicator) = 'true' THEN 1 ELSE 0 END) AS balloon_yes_rows,
                SUM(CASE WHEN LOWER(prepayment_premium_indicator) = 'true' THEN 1 ELSE 0 END) AS prepayment_premium_yes_rows

            FROM sec_cmbs_loans
            GROUP BY deal_name
            ORDER BY deal_name
        """)

        cmbs_property = df_query(conn, """
            SELECT
                deal_name,
                COUNT(*) AS property_rows,
                ROUND(SUM(valuation_amount), 2) AS total_valuation,

                ROUND(
                    SUM(CASE WHEN physical_occupancy IS NOT NULL
                             AND valuation_amount IS NOT NULL
                             THEN valuation_amount * physical_occupancy END)
                    / NULLIF(SUM(CASE WHEN physical_occupancy IS NOT NULL
                                      THEN valuation_amount ELSE 0 END), 0), 4
                ) AS valuation_weighted_occupancy,

                ROUND(
                    SUM(CASE WHEN dscr_noi IS NOT NULL
                             AND valuation_amount IS NOT NULL
                             THEN valuation_amount * dscr_noi END)
                    / NULLIF(SUM(CASE WHEN dscr_noi IS NOT NULL
                                      THEN valuation_amount ELSE 0 END), 0), 3
                ) AS valuation_weighted_dscr_noi,

                ROUND(
                    SUM(CASE WHEN dscr_ncf IS NOT NULL
                             AND valuation_amount IS NOT NULL
                             THEN valuation_amount * dscr_ncf END)
                    / NULLIF(SUM(CASE WHEN dscr_ncf IS NOT NULL
                                      THEN valuation_amount ELSE 0 END), 0), 3
                ) AS valuation_weighted_dscr_ncf,

                ROUND(SUM(net_operating_income), 2) AS total_noi,
                ROUND(SUM(net_cash_flow), 2) AS total_ncf

            FROM sec_cmbs_properties
            GROUP BY deal_name
            ORDER BY deal_name
        """)

        cmbs_property_type = df_query(conn, """
            SELECT
                p.deal_name,
                p.property_type_code,
                COUNT(*) AS properties,
                ROUND(SUM(p.valuation_amount), 2) AS valuation_amount,
                ROUND(
                    100 * SUM(p.valuation_amount)
                    / NULLIF(t.total_valuation, 0), 2
                ) AS valuation_share_pct
            FROM sec_cmbs_properties p
            JOIN (
                SELECT deal_name, SUM(valuation_amount) AS total_valuation
                FROM sec_cmbs_properties
                GROUP BY deal_name
            ) t ON p.deal_name = t.deal_name
            GROUP BY p.deal_name, p.property_type_code, t.total_valuation
            ORDER BY p.deal_name, valuation_amount DESC
        """)

        cmbs_state = df_query(conn, """
            SELECT
                p.deal_name,
                p.property_state,
                COUNT(*) AS properties,
                ROUND(SUM(p.valuation_amount), 2) AS valuation_amount,
                ROUND(
                    100 * SUM(p.valuation_amount)
                    / NULLIF(t.total_valuation, 0), 2
                ) AS valuation_share_pct
            FROM sec_cmbs_properties p
            JOIN (
                SELECT deal_name, SUM(valuation_amount) AS total_valuation
                FROM sec_cmbs_properties
                GROUP BY deal_name
            ) t ON p.deal_name = t.deal_name
            GROUP BY p.deal_name, p.property_state, t.total_valuation
            ORDER BY p.deal_name, valuation_amount DESC
        """)

        cmbs_maturity = df_query(conn, """
            SELECT
                deal_name,
                YEAR(maturity_date) AS maturity_year,
                COUNT(*) AS populated_loan_records,
                ROUND(SUM(original_loan_amount), 2) AS original_loan_amount
            FROM sec_cmbs_loans
            WHERE original_loan_amount IS NOT NULL
            GROUP BY deal_name, YEAR(maturity_date)
            ORDER BY deal_name, maturity_year
        """)

        cmbs_largest = df_query(conn, """
            SELECT
                deal_name,
                asset_number,
                originator_name,
                origination_date,
                maturity_date,
                original_loan_amount,
                original_interest_rate,
                ending_actual_balance,
                interest_only_indicator,
                balloon_indicator,
                payment_status_code
            FROM sec_cmbs_loans
            WHERE original_loan_amount IS NOT NULL
            ORDER BY deal_name, original_loan_amount DESC
        """)

    outputs = {
        "freddie_portfolio_summary.csv": portfolio_summary,
        "freddie_vintage_summary.csv": vintage_summary,
        "freddie_fico_stratification.csv": fico,
        "freddie_ltv_stratification.csv": ltv,
        "freddie_dti_stratification.csv": dti,
        "freddie_state_concentration.csv": state,
        "freddie_loan_purpose.csv": purpose,
        "freddie_occupancy.csv": occupancy,
        "freddie_property_type.csv": property_type,
        "cmbs_deal_summary.csv": cmbs_deal,
        "cmbs_property_summary.csv": cmbs_property,
        "cmbs_property_type_concentration.csv": cmbs_property_type,
        "cmbs_state_concentration.csv": cmbs_state,
        "cmbs_maturity_exposure.csv": cmbs_maturity,
        "cmbs_largest_populated_loan_records.csv": cmbs_largest,
    }

    for filename, df in outputs.items():
        df.to_csv(REPORT_DIR / filename, index=False)

    print("\nFREDDIE VINTAGE SUMMARY")
    print("-" * 70)
    print(vintage_summary.to_string(index=False))

    print("\nSEC CMBS DEAL SUMMARY")
    print("-" * 70)
    print(cmbs_deal.to_string(index=False))

    print("\nSEC CMBS PROPERTY SUMMARY")
    print("-" * 70)
    print(cmbs_property.to_string(index=False))

    print(f"\nCreated {len(outputs)} analytical CSV files in:")
    print(REPORT_DIR)

    print("\nStep 2 completed successfully.")


if __name__ == "__main__":
    run_step2()
