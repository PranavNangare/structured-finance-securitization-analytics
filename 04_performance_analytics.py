from __future__ import annotations

from pathlib import Path
from urllib.parse import quote_plus
import time

import pandas as pd
from sqlalchemy import create_engine, text


# CONFIG

PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_NAME = "structured_finance"
MYSQL_HOST = "127.0.0.1"
MYSQL_PORT = 3306
MYSQL_USER = "root"

# Local development only. Remove before GitHub.
MYSQL_PASSWORD = "123@Pranav"


def get_engine():
    password = quote_plus(MYSQL_PASSWORD)
    return create_engine(
        f"mysql+pymysql://{MYSQL_USER}:{password}"
        f"@{MYSQL_HOST}:{MYSQL_PORT}/{DATABASE_NAME}",
        pool_pre_ping=True,
        future=True,
    )


def query_df(conn, sql: str, label: str) -> pd.DataFrame:
    print(f"\nRunning: {label}")
    start = time.time()
    df = pd.read_sql(text(sql), conn)
    print(f"  rows returned: {len(df):,}")
    print(f"  elapsed: {(time.time() - start):,.1f} sec")
    return df


def add_transition_rates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    denom = out.groupby(["vintage_year", "from_bucket"])["transitions"].transform("sum")
    out["transition_rate_pct"] = (100 * out["transitions"] / denom).round(4)
    return out


def add_delinquency_shares(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    loan_den = out.groupby(["vintage_year", "monthly_reporting_period"])["loan_month_rows"].transform("sum")
    upb_den = out.groupby(["vintage_year", "monthly_reporting_period"])["current_upb"].transform("sum")
    out["loan_share_pct"] = (100 * out["loan_month_rows"] / loan_den).round(4)
    out["upb_share_pct"] = (100 * out["current_upb"] / upb_den.replace(0, pd.NA)).round(4)
    return out


def run_step3():
    engine = get_engine()

    print("Structured Finance - Step 3 Performance Analytics")
    print("=" * 72)
    print("Source tables will not be modified.")
    print("Credit event definition: Zero Balance Code 02, 03, or 09.")
    print("Voluntary payoff definition: Zero Balance Code 01.")

    with engine.connect() as conn:

        # 1. Monthly delinquency stock

        delinquency = query_df(conn, """
            SELECT
                vintage_year,
                monthly_reporting_period,
                CASE
                    WHEN current_loan_delinquency_status IS NULL THEN 'Unknown'
                    WHEN current_loan_delinquency_status IN ('R','XX') THEN 'Special/Unknown'
                    WHEN current_loan_delinquency_status = '0' THEN 'Current'
                    WHEN CAST(current_loan_delinquency_status AS UNSIGNED) = 1 THEN '30 DPD'
                    WHEN CAST(current_loan_delinquency_status AS UNSIGNED) = 2 THEN '60 DPD'
                    WHEN CAST(current_loan_delinquency_status AS UNSIGNED) >= 3 THEN '90+ DPD'
                    ELSE 'Special/Unknown'
                END AS delinquency_bucket,
                COUNT(*) AS loan_month_rows,
                ROUND(SUM(COALESCE(current_actual_upb,0)),2) AS current_upb
            FROM freddie_performance
            GROUP BY vintage_year, monthly_reporting_period, delinquency_bucket
            ORDER BY vintage_year, monthly_reporting_period, delinquency_bucket
        """, "Monthly delinquency stock")
        delinquency = add_delinquency_shares(delinquency)

        # 2. Consecutive-month transition matrix

        transitions = query_df(conn, """
            WITH ordered_perf AS (
                SELECT
                    vintage_year,
                    loan_sequence_number,
                    monthly_reporting_period,
                    current_loan_delinquency_status AS from_status,
                    LEAD(current_loan_delinquency_status) OVER (
                        PARTITION BY loan_sequence_number
                        ORDER BY monthly_reporting_period
                    ) AS to_status,
                    LEAD(monthly_reporting_period) OVER (
                        PARTITION BY loan_sequence_number
                        ORDER BY monthly_reporting_period
                    ) AS next_period
                FROM freddie_performance
            ),
            bucketed AS (
                SELECT
                    vintage_year,
                    monthly_reporting_period,
                    next_period,
                    CASE
                        WHEN from_status IS NULL THEN 'Unknown'
                        WHEN from_status IN ('R','XX') THEN 'Special/Unknown'
                        WHEN from_status = '0' THEN 'Current'
                        WHEN CAST(from_status AS UNSIGNED) = 1 THEN '30 DPD'
                        WHEN CAST(from_status AS UNSIGNED) = 2 THEN '60 DPD'
                        WHEN CAST(from_status AS UNSIGNED) >= 3 THEN '90+ DPD'
                        ELSE 'Special/Unknown'
                    END AS from_bucket,
                    CASE
                        WHEN to_status IS NULL THEN NULL
                        WHEN to_status IN ('R','XX') THEN 'Special/Unknown'
                        WHEN to_status = '0' THEN 'Current'
                        WHEN CAST(to_status AS UNSIGNED) = 1 THEN '30 DPD'
                        WHEN CAST(to_status AS UNSIGNED) = 2 THEN '60 DPD'
                        WHEN CAST(to_status AS UNSIGNED) >= 3 THEN '90+ DPD'
                        ELSE 'Special/Unknown'
                    END AS to_bucket
                FROM ordered_perf
            )
            SELECT
                vintage_year,
                from_bucket,
                to_bucket,
                COUNT(*) AS transitions
            FROM bucketed
            WHERE to_bucket IS NOT NULL
              AND PERIOD_DIFF(next_period, monthly_reporting_period) = 1
            GROUP BY vintage_year, from_bucket, to_bucket
            ORDER BY vintage_year, from_bucket, to_bucket
        """, "Delinquency transition matrix")
        transitions = add_transition_rates(transitions)

        # 3. Monthly prepayment / credit-event rates

        monthly_rates = query_df(conn, """
            WITH perf_lag AS (
                SELECT
                    vintage_year,
                    loan_sequence_number,
                    monthly_reporting_period,
                    current_actual_upb,
                    zero_balance_code,
                    LAG(current_actual_upb) OVER (
                        PARTITION BY loan_sequence_number
                        ORDER BY monthly_reporting_period
                    ) AS prior_upb,
                    LAG(monthly_reporting_period) OVER (
                        PARTITION BY loan_sequence_number
                        ORDER BY monthly_reporting_period
                    ) AS prior_period
                FROM freddie_performance
            ),
            monthly AS (
                SELECT
                    vintage_year,
                    monthly_reporting_period,
                    SUM(
                        CASE
                            WHEN prior_period IS NOT NULL
                             AND PERIOD_DIFF(monthly_reporting_period, prior_period) = 1
                            THEN COALESCE(prior_upb,0)
                            ELSE 0
                        END
                    ) AS beginning_active_upb,
                    SUM(
                        CASE WHEN zero_balance_code = '01'
                             THEN COALESCE(prior_upb,0) ELSE 0 END
                    ) AS voluntary_payoff_balance,
                    SUM(
                        CASE WHEN zero_balance_code IN ('02','03','09')
                             THEN COALESCE(prior_upb,0) ELSE 0 END
                    ) AS credit_event_balance,
                    SUM(zero_balance_code = '01') AS voluntary_payoff_events,
                    SUM(zero_balance_code IN ('02','03','09')) AS credit_events
                FROM perf_lag
                GROUP BY vintage_year, monthly_reporting_period
            )
            SELECT
                vintage_year,
                monthly_reporting_period,
                ROUND(beginning_active_upb,2) AS beginning_active_upb,
                ROUND(voluntary_payoff_balance,2) AS voluntary_payoff_balance,
                voluntary_payoff_events,
                ROUND(voluntary_payoff_balance / NULLIF(beginning_active_upb,0),8) AS smm,
                ROUND(
                    1 - POWER(
                        1 - (voluntary_payoff_balance / NULLIF(beginning_active_upb,0)),
                        12
                    ), 8
                ) AS cpr,
                ROUND(credit_event_balance,2) AS credit_event_balance,
                credit_events,
                ROUND(credit_event_balance / NULLIF(beginning_active_upb,0),8) AS mdr,
                ROUND(
                    1 - POWER(
                        1 - (credit_event_balance / NULLIF(beginning_active_upb,0)),
                        12
                    ), 8
                ) AS cdr
            FROM monthly
            ORDER BY vintage_year, monthly_reporting_period
        """, "Monthly SMM / CPR / MDR / CDR")

        # 4. Zero-balance termination mix

        termination_mix = query_df(conn, """
            SELECT
                vintage_year,
                zero_balance_code,
                COUNT(*) AS events,
                ROUND(SUM(COALESCE(zero_balance_removal_upb,0)),2) AS zero_balance_removal_upb
            FROM freddie_performance
            WHERE zero_balance_code IS NOT NULL
              AND TRIM(zero_balance_code) <> ''
            GROUP BY vintage_year, zero_balance_code
            ORDER BY vintage_year, zero_balance_code
        """, "Zero-balance termination mix")

        # 5. Loss severity and recoveries

        loss_summary = query_df(conn, """
            SELECT
                vintage_year,
                zero_balance_code,
                COUNT(*) AS credit_event_records,
                ROUND(SUM(COALESCE(zero_balance_removal_upb,0)),2) AS removal_upb,
                ROUND(SUM(COALESCE(delinquent_accrued_interest,0)),2) AS delinquent_accrued_interest,
                ROUND(SUM(COALESCE(net_sale_proceeds,0)),2) AS net_sale_proceeds,
                ROUND(SUM(COALESCE(mi_recoveries,0)),2) AS mi_recoveries,
                ROUND(SUM(COALESCE(non_mi_recoveries,0)),2) AS non_mi_recoveries,
                ROUND(SUM(COALESCE(total_expenses,0)),2) AS total_expenses,
                ROUND(SUM(COALESCE(actual_loss,0)),2) AS actual_loss,
                ROUND(
                    SUM(COALESCE(actual_loss,0))
                    / NULLIF(SUM(COALESCE(zero_balance_removal_upb,0)),0),
                    6
                ) AS loss_severity_on_removal_upb
            FROM freddie_performance
            WHERE zero_balance_code IN ('02','03','09')
            GROUP BY vintage_year, zero_balance_code
            ORDER BY vintage_year, zero_balance_code
        """, "Loss severity / recovery summary")

        # 6. Vintage-level event summary
        # Important: aggregate event loans before joining to
        # origination so each originated loan remains one row.

        vintage_events = query_df(conn, """
            WITH event_by_loan AS (
                SELECT
                    loan_sequence_number,
                    MAX(zero_balance_code = '01') AS voluntary_payoff_flag,
                    MAX(zero_balance_code IN ('02','03','09')) AS credit_event_flag
                FROM freddie_performance
                GROUP BY loan_sequence_number
            )
            SELECT
                o.vintage_year,
                COUNT(*) AS originated_loans,
                ROUND(SUM(o.original_upb),2) AS original_upb,
                SUM(COALESCE(e.voluntary_payoff_flag,0)) AS voluntary_payoff_loans,
                SUM(COALESCE(e.credit_event_flag,0)) AS credit_event_loans,
                ROUND(
                    100 * SUM(COALESCE(e.credit_event_flag,0))
                    / NULLIF(COUNT(*),0),
                    4
                ) AS cumulative_credit_event_rate_pct,
                ROUND(
                    100 * SUM(COALESCE(e.voluntary_payoff_flag,0))
                    / NULLIF(COUNT(*),0),
                    4
                ) AS cumulative_voluntary_payoff_rate_pct
            FROM freddie_origination o
            LEFT JOIN event_by_loan e
              ON o.loan_sequence_number = e.loan_sequence_number
            GROUP BY o.vintage_year
            ORDER BY o.vintage_year
        """, "Vintage cumulative event summary")

    outputs = {
        "freddie_monthly_delinquency.csv": delinquency,
        "freddie_delinquency_transitions.csv": transitions,
        "freddie_monthly_prepay_default_rates.csv": monthly_rates,
        "freddie_zero_balance_mix.csv": termination_mix,
        "freddie_loss_severity.csv": loss_summary,
        "freddie_vintage_performance_summary.csv": vintage_events,
    }

    for filename, df in outputs.items():
        df.to_csv(REPORT_DIR / filename, index=False)

    print("\nVINTAGE PERFORMANCE SUMMARY")
    print("-" * 72)
    print(vintage_events.to_string(index=False))

    print("\nLOSS SEVERITY SUMMARY")
    print("-" * 72)
    print(loss_summary.to_string(index=False))

    print("\nTRANSITION MATRIX SAMPLE")
    print("-" * 72)
    print(transitions.head(30).to_string(index=False))

    print(f"\nCreated {len(outputs)} Step 3 CSV files in:")
    print(REPORT_DIR)
    print("\nStep 3 completed successfully.")


if __name__ == "__main__":
    run_step3()
