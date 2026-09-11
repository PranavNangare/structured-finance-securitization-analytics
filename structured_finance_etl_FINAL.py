from __future__ import annotations

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
from urllib.parse import quote_plus

import pandas as pd
from lxml import etree
from sqlalchemy import (
    create_engine, text, MetaData, Table, Column,
    String, Integer, Float, Date, DateTime, Text, BigInteger,
    PrimaryKeyConstraint, UniqueConstraint, Index
)
from sqlalchemy.engine import Engine



# 1. PROJECT CONFIGURATION

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_RAW = PROJECT_ROOT / "data" / "raw"
SEC_ROOT = DATA_RAW / "sec"
FREDDIE_ROOT = DATA_RAW / "freddie"

EXPECTED_FREDDIE_YEARS = [2006, 2007, 2008, 2019, 2020, 2021, 2022]

DATABASE_NAME = "structured_finance"
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = "123@Pranav"

FREDDIE_CHUNK_SIZE = 100_000


# 2. FREDDIE MAC COLUMN LAYOUTS

# The sample origination file uploaded for this project contains 31 fields.
# These are the historical/sample SFLLD column positions.

FREDDIE_ORIG_31 = [
    "credit_score",
    "first_payment_date",
    "first_time_homebuyer_flag",
    "maturity_date",
    "msa",
    "mortgage_insurance_percentage",
    "number_of_units",
    "occupancy_status",
    "original_cltv",
    "original_dti_ratio",
    "original_upb",
    "original_ltv",
    "original_interest_rate",
    "channel",
    "prepayment_penalty_mortgage_flag",
    "amortization_type",
    "property_state",
    "property_type",
    "postal_code",
    "loan_sequence_number",
    "loan_purpose",
    "original_loan_term",
    "number_of_borrowers",
    "seller_name",
    "super_conforming_flag",
    "pre_harp_loan_sequence_number",
    "special_eligibility_program",
    "harp_indicator",
    "property_valuation_method",
    "interest_only_indicator",
    "vantage_score_4",
]

# Some historical documentation/releases contain one additional origination field.
FREDDIE_ORIG_32 = FREDDIE_ORIG_31 + [
    "mortgage_insurance_cancellation_indicator"
]

# Historical/sample performance layout.
FREDDIE_PERF_32 = [
    "loan_sequence_number",
    "monthly_reporting_period",
    "current_actual_upb",
    "current_loan_delinquency_status",
    "loan_age",
    "remaining_months_to_legal_maturity",
    "defect_settlement_date",
    "modification_flag",
    "zero_balance_code",
    "zero_balance_effective_date",
    "current_interest_rate",
    "current_non_interest_bearing_upb",
    "due_date_last_paid_installment",
    "mi_recoveries",
    "net_sale_proceeds",
    "non_mi_recoveries",
    "total_expenses",
    "legal_costs",
    "maintenance_preservation_costs",
    "taxes_and_insurance",
    "miscellaneous_expenses",
    "actual_loss",
    "cumulative_modification_cost",
    "step_modification_flag",
    "deferred_payment_plan",
    "estimated_ltv",
    "zero_balance_removal_upb",
    "delinquent_accrued_interest",
    "delinquency_due_to_disaster",
    "borrower_assistance_status_code",
    "current_month_modification_cost",
    "interest_bearing_upb",
]

# Release 47 Standard Dataset performance layout.
FREDDIE_PERF_35 = FREDDIE_PERF_32 + [
    "mortgage_insurance_cancellation_indicator",
    "servicer_name",
    "bankruptcy_cramdown_costs",
]


# 3. DATABASE ENGINE

def create_database_and_engine() -> Engine:
    password = quote_plus(MYSQL_PASSWORD)

    base_url = (
        f"mysql+pymysql://{MYSQL_USER}:{password}"
        f"@{MYSQL_HOST}:{MYSQL_PORT}/"
    )

    print(f"Connecting as {MYSQL_USER} to {MYSQL_HOST}:{MYSQL_PORT}")

    base_engine = create_engine(
        base_url,
        echo=False,
        pool_pre_ping=True,
        future=True,
    )

    with base_engine.begin() as conn:
        conn.execute(
            text(
                f"CREATE DATABASE IF NOT EXISTS `{DATABASE_NAME}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
            )
        )

    engine = create_engine(
        base_url + DATABASE_NAME,
        echo=False,
        pool_pre_ping=True,
        future=True,
    )

    return engine


# 4. SQLALCHEMY TABLE DEFINITIONS

metadata = MetaData()

raw_file_manifest = Table(
    "raw_file_manifest",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("source", String(30), nullable=False),
    Column("dataset", String(100), nullable=False),
    Column("vintage_year", Integer),
    Column("file_name", String(255), nullable=False),
    Column("file_path", Text, nullable=False),
    Column("file_size_bytes", BigInteger, nullable=False),
    Column("sha256", String(64), nullable=False),
    Column("loaded_at", DateTime, nullable=False, default=datetime.utcnow),
    UniqueConstraint("sha256", name="uq_raw_file_sha256"),
)

freddie_origination = Table(
    "freddie_origination",
    metadata,
    Column("loan_sequence_number", String(20), primary_key=True),
    Column("vintage_year", Integer, nullable=False, index=True),
    Column("credit_score", Integer),
    Column("first_payment_date", String(6)),
    Column("first_time_homebuyer_flag", String(2)),
    Column("maturity_date", String(6)),
    Column("msa", String(10)),
    Column("mortgage_insurance_percentage", Float),
    Column("number_of_units", Integer),
    Column("occupancy_status", String(2)),
    Column("original_cltv", Float),
    Column("original_dti_ratio", Float),
    Column("original_upb", Float),
    Column("original_ltv", Float),
    Column("original_interest_rate", Float),
    Column("channel", String(2)),
    Column("prepayment_penalty_mortgage_flag", String(2)),
    Column("amortization_type", String(10)),
    Column("property_state", String(5)),
    Column("property_type", String(5)),
    Column("postal_code", String(10)),
    Column("loan_purpose", String(5)),
    Column("original_loan_term", Integer),
    Column("number_of_borrowers", Integer),
    Column("seller_name", String(255)),
    Column("servicer_name", String(255)),
    Column("super_conforming_flag", String(2)),
    Column("pre_harp_loan_sequence_number", String(20)),
    Column("special_eligibility_program", String(20)),
    Column("harp_indicator", String(2)),
    Column("property_valuation_method", String(10)),
    Column("interest_only_indicator", String(2)),
    Column("vantage_score_4", Integer),
    Column("mortgage_insurance_cancellation_indicator", String(2)),
    Column("source_file", String(255), nullable=False),
)

freddie_performance = Table(
    "freddie_performance",
    metadata,
    Column("loan_sequence_number", String(20), nullable=False),
    Column("monthly_reporting_period", String(6), nullable=False),
    Column("vintage_year", Integer, nullable=False),
    Column("current_actual_upb", Float),
    Column("current_loan_delinquency_status", String(5)),
    Column("loan_age", Integer),
    Column("remaining_months_to_legal_maturity", Integer),
    Column("defect_settlement_date", String(10)),
    Column("modification_flag", String(2)),
    Column("zero_balance_code", String(5)),
    Column("zero_balance_effective_date", String(10)),
    Column("current_interest_rate", Float),
    Column("current_non_interest_bearing_upb", Float),
    Column("due_date_last_paid_installment", String(10)),
    Column("mi_recoveries", Float),
    Column("net_sale_proceeds", Float),
    Column("non_mi_recoveries", Float),
    Column("total_expenses", Float),
    Column("legal_costs", Float),
    Column("maintenance_preservation_costs", Float),
    Column("taxes_and_insurance", Float),
    Column("miscellaneous_expenses", Float),
    Column("actual_loss", Float),
    Column("cumulative_modification_cost", Float),
    Column("step_modification_flag", String(2)),
    Column("deferred_payment_plan", String(2)),
    Column("estimated_ltv", Float),
    Column("zero_balance_removal_upb", Float),
    Column("delinquent_accrued_interest", Float),
    Column("delinquency_due_to_disaster", String(2)),
    Column("borrower_assistance_status_code", String(5)),
    Column("current_month_modification_cost", Float),
    Column("interest_bearing_upb", Float),
    Column("mortgage_insurance_cancellation_indicator", String(2)),
    Column("servicer_name", String(255)),
    Column("bankruptcy_cramdown_costs", Float),
    Column("source_file", String(255), nullable=False),
    PrimaryKeyConstraint(
        "loan_sequence_number",
        "monthly_reporting_period",
        name="pk_freddie_performance"
    ),
)

Index(
    "ix_freddie_perf_period",
    freddie_performance.c.monthly_reporting_period
)
Index(
    "ix_freddie_perf_zero_balance",
    freddie_performance.c.zero_balance_code
)

sec_cmbs_loans = Table(
    "sec_cmbs_loans",
    metadata,
    Column("deal_name", String(100), nullable=False),
    Column("asset_number", String(40), nullable=False),
    Column("group_id", String(40)),
    Column("reporting_period_begin", Date),
    Column("reporting_period_end", Date),
    Column("originator_name", String(255)),
    Column("origination_date", Date),
    Column("original_loan_amount", Float),
    Column("original_term_months", Integer),
    Column("maturity_date", Date),
    Column("original_amortization_term_months", Integer),
    Column("original_interest_rate", Float),
    Column("securitization_interest_rate", Float),
    Column("original_io_term_months", Integer),
    Column("interest_only_indicator", String(10)),
    Column("balloon_indicator", String(10)),
    Column("prepayment_premium_indicator", String(10)),
    Column("modified_indicator", String(10)),
    Column("scheduled_balance_at_securitization", Float),
    Column("beginning_scheduled_balance", Float),
    Column("ending_actual_balance", Float),
    Column("ending_scheduled_balance", Float),
    Column("scheduled_interest", Float),
    Column("scheduled_principal", Float),
    Column("unscheduled_principal", Float),
    Column("payment_status_code", String(20)),
    Column("primary_servicer_name", String(255)),
    Column("raw_payload_json", Text),
    Column("source_file", String(255), nullable=False),
    PrimaryKeyConstraint("deal_name", "asset_number", name="pk_sec_cmbs_loans"),
)

sec_cmbs_properties = Table(
    "sec_cmbs_properties",
    metadata,
    Column("deal_name", String(100), nullable=False),
    Column("asset_number", String(40), nullable=False),
    Column("property_sequence", Integer, nullable=False),
    Column("property_name", String(255)),
    Column("property_address", String(255)),
    Column("property_city", String(100)),
    Column("property_state", String(10)),
    Column("property_zip", String(20)),
    Column("property_county", String(100)),
    Column("property_type_code", String(20)),
    Column("net_rentable_square_feet", Float),
    Column("year_built", Integer),
    Column("valuation_amount", Float),
    Column("physical_occupancy", Float),
    Column("largest_tenant", String(255)),
    Column("revenue", Float),
    Column("operating_expenses", Float),
    Column("net_operating_income", Float),
    Column("net_cash_flow", Float),
    Column("dscr_noi", Float),
    Column("dscr_ncf", Float),
    Column("raw_payload_json", Text),
    Column("source_file", String(255), nullable=False),
    PrimaryKeyConstraint(
        "deal_name",
        "asset_number",
        "property_sequence",
        name="pk_sec_cmbs_properties"
    ),
)

data_quality_results = Table(
    "data_quality_results",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("source", String(30), nullable=False),
    Column("dataset", String(100), nullable=False),
    Column("check_name", String(100), nullable=False),
    Column("status", String(20), nullable=False),
    Column("record_count", BigInteger),
    Column("details", Text),
    Column("checked_at", DateTime, nullable=False, default=datetime.utcnow),
)


# 5. GENERAL HELPERS


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def register_file(
    engine: Engine,
    path: Path,
    source: str,
    dataset: str,
    vintage_year: int | None = None,
) -> None:
    digest = sha256_file(path)

    sql = text("""
        INSERT IGNORE INTO raw_file_manifest
            (source, dataset, vintage_year, file_name, file_path,
             file_size_bytes, sha256, loaded_at)
        VALUES
            (:source, :dataset, :vintage_year, :file_name, :file_path,
             :file_size_bytes, :sha256, :loaded_at)
    """)

    with engine.begin() as conn:
        conn.execute(
            sql,
            {
                "source": source,
                "dataset": dataset,
                "vintage_year": vintage_year,
                "file_name": path.name,
                "file_path": str(path.resolve()),
                "file_size_bytes": path.stat().st_size,
                "sha256": digest,
                "loaded_at": datetime.utcnow(),
            },
        )


def log_quality(
    engine: Engine,
    source: str,
    dataset: str,
    check_name: str,
    status: str,
    record_count: int | None = None,
    details: str | None = None,
) -> None:
    with engine.begin() as conn:
        conn.execute(
            data_quality_results.insert().values(
                source=source,
                dataset=dataset,
                check_name=check_name,
                status=status,
                record_count=record_count,
                details=details,
                checked_at=datetime.utcnow(),
            )
        )


def normalize_nulls(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace(
        {
            "": None,
            " ": None,
            "NULL": None,
            "null": None,
        }
    )


def numeric(df: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


# 6. FREDDIE FILE DISCOVERY

def discover_freddie_files() -> dict[int, dict[str, Path]]:
    discovered: dict[int, dict[str, Path]] = {}

    for year in EXPECTED_FREDDIE_YEARS:
        year_dir = FREDDIE_ROOT / str(year)

        if not year_dir.exists():
            raise FileNotFoundError(
                f"Missing Freddie folder: {year_dir}"
            )

        orig_candidates = sorted(year_dir.glob(f"*orig*{year}*.txt"))
        perf_candidates = sorted(year_dir.glob(f"*perf*{year}*.txt"))

        if len(orig_candidates) != 1:
            raise RuntimeError(
                f"{year}: expected exactly 1 origination TXT file, "
                f"found {len(orig_candidates)}: {[p.name for p in orig_candidates]}"
            )

        if len(perf_candidates) != 1:
            raise RuntimeError(
                f"{year}: expected exactly 1 performance TXT file, "
                f"found {len(perf_candidates)}: {[p.name for p in perf_candidates]}"
            )

        discovered[year] = {
            "orig": orig_candidates[0],
            "perf": perf_candidates[0],
        }

    return discovered


def detect_pipe_column_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        line = f.readline().rstrip("\r\n")
    return len(line.split("|"))


# 7. FREDDIE ORIGINATION ETL

ORIG_NUMERIC_COLUMNS = [
    "credit_score",
    "mortgage_insurance_percentage",
    "number_of_units",
    "original_cltv",
    "original_dti_ratio",
    "original_upb",
    "original_ltv",
    "original_interest_rate",
    "original_loan_term",
    "number_of_borrowers",
]

def load_freddie_origination(
    engine: Engine,
    year: int,
    path: Path,
) -> int:
    column_count = detect_pipe_column_count(path)

    if column_count == 31:
        columns = FREDDIE_ORIG_31
    elif column_count == 32:
        columns = FREDDIE_ORIG_32
    else:
        raise ValueError(
            f"{path.name}: unsupported origination column count "
            f"{column_count}. Expected 31 or 32."
        )

    print(
        f"  Origination: {path.name} "
        f"({column_count} columns)"
    )

    df = pd.read_csv(
        path,
        sep="|",
        header=None,
        names=columns,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    df = normalize_nulls(df)
    numeric(df, ORIG_NUMERIC_COLUMNS)

    # Freddie sample placeholders.
    if "credit_score" in df:
        df.loc[df["credit_score"] == 9999, "credit_score"] = pd.NA
    if "vantage_score_4" in df:
        df.loc[df["vantage_score_4"] == 9999, "vantage_score_4"] = pd.NA
    if "original_dti_ratio" in df:
        df.loc[df["original_dti_ratio"] == 999, "original_dti_ratio"] = pd.NA
    if "property_valuation_method" in df:
        df["property_valuation_method"] = df["property_valuation_method"].replace({"": None})

    df["vintage_year"] = year
    df["source_file"] = path.name

    if "mortgage_insurance_cancellation_indicator" not in df.columns:
        df["mortgage_insurance_cancellation_indicator"] = None

    expected = {c.name for c in freddie_origination.columns}
    df = df[[c for c in df.columns if c in expected]]

    duplicates = int(df["loan_sequence_number"].duplicated().sum())
    null_ids = int(df["loan_sequence_number"].isna().sum())

    if duplicates or null_ids:
        raise ValueError(
            f"{path.name}: invalid loan keys. "
            f"duplicates={duplicates}, null_ids={null_ids}"
        )

    # Idempotent reload per vintage: remove only this year's rows.
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM freddie_origination WHERE vintage_year = :year"),
            {"year": year},
        )

    df.to_sql(
        "freddie_origination",
        con=engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=5_000,
    )

    register_file(
        engine, path, "Freddie Mac", "SFLLD Origination Sample", year
    )

    log_quality(
        engine,
        "Freddie Mac",
        f"SFLLD Origination {year}",
        "unique_loan_sequence_number",
        "PASS",
        len(df),
        f"duplicates={duplicates}; null_ids={null_ids}; columns={column_count}",
    )

    return len(df)


# 8. FREDDIE PERFORMANCE ETL


PERF_NUMERIC_COLUMNS = [
    "current_actual_upb",
    "loan_age",
    "remaining_months_to_legal_maturity",
    "current_interest_rate",
    "current_non_interest_bearing_upb",
    "mi_recoveries",
    "net_sale_proceeds",
    "non_mi_recoveries",
    "total_expenses",
    "legal_costs",
    "maintenance_preservation_costs",
    "taxes_and_insurance",
    "miscellaneous_expenses",
    "actual_loss",
    "cumulative_modification_cost",
    "estimated_ltv",
    "zero_balance_removal_upb",
    "delinquent_accrued_interest",
    "current_month_modification_cost",
    "interest_bearing_upb",
    "bankruptcy_cramdown_costs",
]

def performance_columns_for_count(column_count: int) -> list[str]:
    if column_count == 32:
        return FREDDIE_PERF_32
    if column_count == 35:
        return FREDDIE_PERF_35

    raise ValueError(
        f"Unsupported Freddie performance column count {column_count}. "
        "This script supports the 32-column historical/sample layout "
        "and the 35-column Release 47 Standard layout."
    )


def load_freddie_performance(
    engine: Engine,
    year: int,
    path: Path,
) -> int:
    column_count = detect_pipe_column_count(path)
    columns = performance_columns_for_count(column_count)

    print(
        f"  Performance: {path.name} "
        f"({column_count} columns, chunk={FREDDIE_CHUNK_SIZE:,})"
    )

    # Make re-runs idempotent by removing only the selected vintage.
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM freddie_performance WHERE vintage_year = :year"),
            {"year": year},
        )

    total_rows = 0
    null_keys = 0
    duplicate_keys = 0

    # Duplicate checking across chunks without retaining full rows.
    seen_keys: set[tuple[str, str]] = set()

    reader = pd.read_csv(
        path,
        sep="|",
        header=None,
        names=columns,
        dtype=str,
        keep_default_na=False,
        chunksize=FREDDIE_CHUNK_SIZE,
        low_memory=False,
    )

    expected = {c.name for c in freddie_performance.columns}

    for chunk_no, df in enumerate(reader, start=1):
        df = normalize_nulls(df)
        numeric(df, PERF_NUMERIC_COLUMNS)

        # Net sale proceeds may contain U = Unknown.
        if "net_sale_proceeds" in df.columns:
            df["net_sale_proceeds"] = pd.to_numeric(
                df["net_sale_proceeds"], errors="coerce"
            )

        df["vintage_year"] = year
        df["source_file"] = path.name

        for optional in [
            "mortgage_insurance_cancellation_indicator",
            "servicer_name",
            "bankruptcy_cramdown_costs",
        ]:
            if optional not in df.columns:
                df[optional] = None

        key_null_mask = (
            df["loan_sequence_number"].isna()
            | df["monthly_reporting_period"].isna()
        )
        null_keys += int(key_null_mask.sum())

        if key_null_mask.any():
            raise ValueError(
                f"{path.name}: null loan/month key found in performance "
                f"chunk {chunk_no}."
            )

        local_dup = int(
            df.duplicated(
                subset=["loan_sequence_number", "monthly_reporting_period"]
            ).sum()
        )
        duplicate_keys += local_dup

        # Cross-chunk duplicate check.
        keys = list(
            zip(
                df["loan_sequence_number"].astype(str),
                df["monthly_reporting_period"].astype(str),
            )
        )
        cross_dup = sum(key in seen_keys for key in keys)
        duplicate_keys += cross_dup
        seen_keys.update(keys)

        if local_dup or cross_dup:
            raise ValueError(
                f"{path.name}: duplicate loan/month keys detected "
                f"in chunk {chunk_no}."
            )

        df = df[[c for c in df.columns if c in expected]]

        df.to_sql(
            "freddie_performance",
            con=engine,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=2_000,
        )

        total_rows += len(df)
        print(
            f"    loaded chunk {chunk_no}: "
            f"{len(df):,} rows | cumulative {total_rows:,}"
        )

    register_file(
        engine, path, "Freddie Mac", "SFLLD Performance Sample", year
    )

    log_quality(
        engine,
        "Freddie Mac",
        f"SFLLD Performance {year}",
        "loan_month_primary_key",
        "PASS",
        total_rows,
        (
            f"null_keys={null_keys}; duplicate_keys={duplicate_keys}; "
            f"columns={column_count}"
        ),
    )

    return total_rows


# 9. SEC / GOLDMAN CMBS XML ETL

def xml_local_name(element) -> str:
    return etree.QName(element).localname


def child(element, tag: str):
    for c in element:
        if xml_local_name(c) == tag:
            return c
    return None


def children(element, tag: str):
    return [c for c in element if xml_local_name(c) == tag]


def xml_value(element, tag: str) -> str | None:
    c = child(element, tag)
    if c is None or c.text is None:
        return None
    value = c.text.strip()
    return value or None


def as_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value):
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def as_date(value):
    if not value:
        return None

    for fmt in ("%m-%d-%Y", "%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass

    return None


def xml_to_dict(element):
    result = {}

    for c in element:
        tag = xml_local_name(c)

        if len(c):
            value = xml_to_dict(c)
        else:
            value = c.text.strip() if c.text else None

        if tag in result:
            if not isinstance(result[tag], list):
                result[tag] = [result[tag]]
            result[tag].append(value)
        else:
            result[tag] = value

    return result


def deal_name_from_folder(path: Path) -> str:
    return path.parent.name.lower()


def load_sec_ex102(engine: Engine, path: Path) -> tuple[int, int]:
    tree = etree.parse(str(path))
    root = tree.getroot()

    if xml_local_name(root) != "assetData":
        print(
            f"  Skipping {path}: root={xml_local_name(root)!r}; "
            "not an EX-102 assetData file."
        )
        return 0, 0

    deal_name = deal_name_from_folder(path)
    assets = [e for e in root if xml_local_name(e) == "assets"]

    loan_rows = []
    property_rows = []

    for asset in assets:
        asset_number = xml_value(asset, "assetNumber")
        if not asset_number:
            continue

        payload = xml_to_dict(asset)

        loan_rows.append(
            {
                "deal_name": deal_name,
                "asset_number": asset_number,
                "group_id": xml_value(asset, "GroupID"),
                "reporting_period_begin": as_date(
                    xml_value(asset, "reportingPeriodBeginningDate")
                ),
                "reporting_period_end": as_date(
                    xml_value(asset, "reportingPeriodEndDate")
                ),
                "originator_name": xml_value(asset, "originatorName"),
                "origination_date": as_date(
                    xml_value(asset, "originationDate")
                ),
                "original_loan_amount": as_float(
                    xml_value(asset, "originalLoanAmount")
                ),
                "original_term_months": as_int(
                    xml_value(asset, "originalTermLoanNumber")
                ),
                "maturity_date": as_date(
                    xml_value(asset, "maturityDate")
                ),
                "original_amortization_term_months": as_int(
                    xml_value(asset, "originalAmortizationTermNumber")
                ),
                "original_interest_rate": as_float(
                    xml_value(asset, "originalInterestRatePercentage")
                ),
                "securitization_interest_rate": as_float(
                    xml_value(asset, "interestRateSecuritizationPercentage")
                ),
                "original_io_term_months": as_int(
                    xml_value(asset, "originalInterestOnlyTermNumber")
                ),
                "interest_only_indicator": xml_value(
                    asset, "interestOnlyIndicator"
                ),
                "balloon_indicator": xml_value(asset, "balloonIndicator"),
                "prepayment_premium_indicator": xml_value(
                    asset, "prepaymentPremiumIndicator"
                ),
                "modified_indicator": xml_value(asset, "modifiedIndicator"),
                "scheduled_balance_at_securitization": as_float(
                    xml_value(
                        asset,
                        "scheduledPrincipalBalanceSecuritizationAmount"
                    )
                ),
                "beginning_scheduled_balance": as_float(
                    xml_value(
                        asset,
                        "reportPeriodBeginningScheduleLoanBalanceAmount"
                    )
                ),
                "ending_actual_balance": as_float(
                    xml_value(asset, "reportPeriodEndActualBalanceAmount")
                ),
                "ending_scheduled_balance": as_float(
                    xml_value(
                        asset,
                        "reportPeriodEndScheduledLoanBalanceAmount"
                    )
                ),
                "scheduled_interest": as_float(
                    xml_value(asset, "scheduledInterestAmount")
                ),
                "scheduled_principal": as_float(
                    xml_value(asset, "scheduledPrincipalAmount")
                ),
                "unscheduled_principal": as_float(
                    xml_value(asset, "unscheduledPrincipalCollectedAmount")
                ),
                "payment_status_code": xml_value(
                    asset, "paymentStatusLoanCode"
                ),
                "primary_servicer_name": xml_value(
                    asset, "primaryServicerName"
                ),
                "raw_payload_json": json.dumps(
                    payload, ensure_ascii=False
                ),
                "source_file": path.name,
            }
        )

        for property_sequence, prop in enumerate(
            children(asset, "property"), start=1
        ):
            property_rows.append(
                {
                    "deal_name": deal_name,
                    "asset_number": asset_number,
                    "property_sequence": property_sequence,
                    "property_name": xml_value(prop, "propertyName"),
                    "property_address": xml_value(prop, "propertyAddress"),
                    "property_city": xml_value(prop, "propertyCity"),
                    "property_state": xml_value(prop, "propertyState"),
                    "property_zip": xml_value(prop, "propertyZip"),
                    "property_county": xml_value(prop, "propertyCounty"),
                    "property_type_code": xml_value(
                        prop, "propertyTypeCode"
                    ),
                    "net_rentable_square_feet": as_float(
                        xml_value(prop, "netRentableSquareFeetNumber")
                    ),
                    "year_built": as_int(
                        xml_value(prop, "yearBuiltNumber")
                    ),
                    "valuation_amount": as_float(
                        xml_value(prop, "valuationSecuritizationAmount")
                    ),
                    "physical_occupancy": as_float(
                        xml_value(
                            prop,
                            "physicalOccupancySecuritizationPercentage"
                        )
                    ),
                    "largest_tenant": xml_value(prop, "largestTenant"),
                    "revenue": as_float(
                        xml_value(prop, "revenueSecuritizationAmount")
                    ),
                    "operating_expenses": as_float(
                        xml_value(
                            prop,
                            "operatingExpensesSecuritizationAmount"
                        )
                    ),
                    "net_operating_income": as_float(
                        xml_value(
                            prop,
                            "netOperatingIncomeSecuritizationAmount"
                        )
                    ),
                    "net_cash_flow": as_float(
                        xml_value(
                            prop,
                            "netCashFlowFlowSecuritizationAmount"
                        )
                    ),
                    "dscr_noi": as_float(
                        xml_value(
                            prop,
                            "debtServiceCoverageNetOperatingIncomeSecuritizationPercentage"
                        )
                    ),
                    "dscr_ncf": as_float(
                        xml_value(
                            prop,
                            "debtServiceCoverageNetCashFlowSecuritizationPercentage"
                        )
                    ),
                    "raw_payload_json": json.dumps(
                        xml_to_dict(prop), ensure_ascii=False
                    ),
                    "source_file": path.name,
                }
            )

    # Idempotent reload for this deal.
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM sec_cmbs_properties WHERE deal_name = :deal"),
            {"deal": deal_name},
        )
        conn.execute(
            text("DELETE FROM sec_cmbs_loans WHERE deal_name = :deal"),
            {"deal": deal_name},
        )

    if loan_rows:
        pd.DataFrame(loan_rows).to_sql(
            "sec_cmbs_loans",
            con=engine,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=1_000,
        )

    if property_rows:
        pd.DataFrame(property_rows).to_sql(
            "sec_cmbs_properties",
            con=engine,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=1_000,
        )

    register_file(
        engine, path, "SEC", f"CMBS EX-102 {deal_name}", None
    )

    log_quality(
        engine,
        "SEC",
        deal_name,
        "asset_number_unique",
        "PASS",
        len(loan_rows),
        f"properties={len(property_rows)}",
    )

    return len(loan_rows), len(property_rows)


def discover_sec_ex102_files() -> list[Path]:
    files = []

    if not SEC_ROOT.exists():
        return files

    for path in SEC_ROOT.rglob("*.xml"):
        try:
            tree = etree.parse(str(path))
            if xml_local_name(tree.getroot()) == "assetData":
                files.append(path)
        except Exception:
            continue

    return sorted(files)


# 10. CROSS-DATASET VALIDATION

def validate_freddie_relationships(engine: Engine) -> None:
    """
    Check whether every performance loan ID exists in origination.
    Performance contains many rows per loan, so this uses SQL rather
    than loading the full tables back into pandas.
    """
    with engine.connect() as conn:
        missing = conn.execute(
            text("""
                SELECT COUNT(DISTINCT p.loan_sequence_number)
                FROM freddie_performance p
                LEFT JOIN freddie_origination o
                  ON p.loan_sequence_number = o.loan_sequence_number
                WHERE o.loan_sequence_number IS NULL
            """)
        ).scalar_one()

        orig_count = conn.execute(
            text("SELECT COUNT(*) FROM freddie_origination")
        ).scalar_one()

        perf_count = conn.execute(
            text("SELECT COUNT(*) FROM freddie_performance")
        ).scalar_one()

    status = "PASS" if missing == 0 else "WARN"

    log_quality(
        engine,
        "Freddie Mac",
        "SFLLD",
        "performance_to_origination_link",
        status,
        int(perf_count),
        (
            f"origination_rows={orig_count}; "
            f"performance_rows={perf_count}; "
            f"performance_loan_ids_missing_from_origination={missing}"
        ),
    )

    print("\n=== FREDDIE RELATIONSHIP CHECK ===")
    print(f"Origination rows : {orig_count:,}")
    print(f"Performance rows : {perf_count:,}")
    print(f"Unmatched loan IDs: {missing:,}")


# 11. DATABASE SUMMARY

def print_database_summary(engine: Engine) -> None:
    print("\n=== DATABASE SUMMARY ===")

    tables = [
        "freddie_origination",
        "freddie_performance",
        "sec_cmbs_loans",
        "sec_cmbs_properties",
        "raw_file_manifest",
        "data_quality_results",
    ]

    with engine.connect() as conn:
        for table_name in tables:
            count = conn.execute(
                text(f"SELECT COUNT(*) FROM `{table_name}`")
            ).scalar_one()
            print(f"{table_name:<30} {count:>12,}")


# 12. MAIN PIPELINE

def main() -> None:
    print("Structured Finance Data Pipeline")
    print("=" * 60)

    engine = create_database_and_engine()
    metadata.create_all(engine)

    print(f"\nMySQL database ready: {DATABASE_NAME}")

    
    # Freddie Mac
    
    print("\n=== FREDDIE MAC SFLLD ===")

    freddie_files = discover_freddie_files()

    for year, paths in freddie_files.items():
        print(f"\n{year}")
        print("-" * 30)

        orig_rows = load_freddie_origination(
            engine, year, paths["orig"]
        )
        print(f"    origination rows loaded: {orig_rows:,}")

        perf_rows = load_freddie_performance(
            engine, year, paths["perf"]
        )
        print(f"    performance rows loaded: {perf_rows:,}")

    validate_freddie_relationships(engine)

    # SEC / Goldman-linked CMBS
    print("\n=== SEC / GOLDMAN-LINKED CMBS ===")

    sec_files = discover_sec_ex102_files()

    if not sec_files:
        print(f"No EX-102 assetData XML found below {SEC_ROOT}")
    else:
        for path in sec_files:
            deal = deal_name_from_folder(path)
            print(f"\n{deal}: {path.name}")

            loan_count, property_count = load_sec_ex102(
                engine, path
            )

            print(f"  CMBS loans loaded      : {loan_count:,}")
            print(f"  CMBS properties loaded : {property_count:,}")

    print_database_summary(engine)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
