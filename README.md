# Structured Finance Securitization & Loan Portfolio Optimization

## Overview

This portfolio project builds an end-to-end structured-finance analytics
workflow using Freddie Mac Single-Family Loan-Level Dataset samples and
SEC ABS-EE commercial mortgage disclosures from Goldman Sachs-linked
CMBS transactions.

The project covers data engineering, mortgage performance analytics,
credit risk, prepayment behavior, expected loss, fixed-income cash-flow
analytics, constrained loan-pool selection, and stress testing.

> **Important:** This is an educational portfolio project. The Freddie
> residential and SEC CMBS datasets are analyzed as separate
> workstreams. The project does not represent an actual Goldman Sachs,
> Freddie Mac, issuer, rating-agency, or investor model.

## Project Highlights

-   Loaded **350,000 Freddie Mac origination records** and **17,696,387
    monthly performance records** into MySQL.
-   Processed SEC ABS-EE XML disclosures for two Goldman Sachs-linked
    CMBS deals.
-   Built delinquency, transition, termination, prepayment, and
    loss-severity analytics.
-   Developed a 12-month credit-risk framework with expanding-window
    out-of-time validation.
-   Integrated Freddie Mac PMMS / FRED mortgage rates with realized SMM
    and CPR behavior.
-   Estimated 12-month expected loss using **PD × LGD × EAD**.
-   Connected empirical CPR scenarios to stylized fixed-income price,
    duration, convexity, and cash flows.
-   Selected a constrained **\~\$250M mortgage pool** using a
    transparent risk-quality heuristic.
-   Applied explicit base, moderate, and severe credit stress scenarios.

## Data

### Freddie Mac Single-Family Loan-Level Dataset

Seven 50,000-loan sample vintages were used:

-   2006
-   2007
-   2008
-   2019
-   2020
-   2021
-   2022

Total sampled originations: **350,000 loans**.

The monthly performance data expands to **17,696,387 records**.

### SEC ABS-EE CMBS Data

Two Goldman Sachs-linked commercial mortgage securitization datasets
were parsed separately:

-   Benchmark 2025-V18 Mortgage Trust
-   GS Mortgage Securities Trust 2018-GS10

The SEC workstream is used for CMBS collateral and concentration
analytics and is not merged loan-to-loan with Freddie residential
mortgages.

## Analytical Pipeline

### 1. ETL and Data Quality

Python and MySQL are used to ingest, validate, normalize, and store
Freddie pipe-delimited files and SEC ABS-EE XML files.

Core database tables include:

-   `freddie_origination`
-   `freddie_performance`
-   `sec_cmbs_loans`
-   `sec_cmbs_properties`
-   `raw_file_manifest`
-   `data_quality_results`

### 2. Portfolio and CMBS Analytics

Residential stratification includes FICO, LTV, DTI, geography, loan
purpose, occupancy, property type, and vintage.

CMBS analysis focuses on commercial mortgage and property attributes,
concentration, maturity exposure, and collateral characteristics.

### 3. Mortgage Performance Analytics

Historical performance analysis includes:

-   delinquency stock
-   delinquency transitions
-   voluntary payoff behavior
-   credit-event terminations
-   realized loss severity
-   vintage-level cumulative event rates
-   monthly SMM and CPR

Credit-event removal codes are defined as Freddie zero-balance codes
**02, 03, and 09**. Voluntary payoff is code **01**.

### 4. 12-Month Credit Risk

A 12-month credit-event target is created from month-12 loan snapshots.

Expanding-window out-of-time validation was used to avoid random
temporal leakage. Pooled genuine OOT results contained **185,990
observations and 24 credit events**, with ROC AUC **0.9446** and
calibration ratio **1.485**.

Because recent vintages contain very few credit events, recent-vintage
AUC/AP values are treated as unstable. The model is presented as a
portfolio analytics framework rather than a production underwriting
model.

### 5. Prepayment Analytics and CPR Scenarios

Realized voluntary prepayment is measured using SMM and annualized CPR.
Freddie Mac PMMS 30-year mortgage rates are integrated to analyze rate
regimes.

Observed average realized CPR:

  Rate regime          Average market rate   Average realized CPR
  ------------------ --------------------- ----------------------
  Low Market Rate                    3.43%                 17.98%
  Mid Market Rate                    4.85%                 14.75%
  High Market Rate                   6.66%                  5.15%

Empirical CPR scenarios are based on realized monthly CPR percentiles:

  Scenario                 CPR       SMM
  ----------------- ---------- ---------
  Low Prepayment       5.2352%   0.4471%
  Base Prepayment     10.6288%   0.9321%
  High Prepayment     16.8998%   1.5309%

### 6. Expected Loss

For the 2022 month-12 snapshot:

-   Loans: **45,357**
-   EAD: **\$13.273B**
-   Weighted 12-month PD: **0.0143%**
-   Historical realized LGD: **45.1011%**
-   Historical LGD event records: **10,180**
-   Expected loss: **\$857,199**
-   Expected loss rate: **0.0065%**

LGD is pooled from actual pre-2022 Freddie credit-event removal records.

### 7. Stylized Fixed-Income Analytics

A stylized \$100M security is evaluated using:

-   5.50% coupon
-   6.00% base yield
-   5-year term
-   empirical Step 5 CPR scenarios

  -----------------------------------------------------------------------
  CPR            Price / Par       Macaulay       Modified      Convexity
  scenario                         duration       duration 
  ----------- -------------- -------------- -------------- --------------
  Low                 99.005          2.005          1.995          5.725

  Base                99.128          1.756          1.748          4.458

  High                99.240          1.530          1.522          3.435
  -----------------------------------------------------------------------

These are stylized cash-flow outputs, not actual security valuations.

### 8. \$250M Pool Selection

The portfolio-selection heuristic selected:

-   **1,291 loans**
-   **\$247.681M UPB**
-   **99.07% target fill**
-   Weighted FICO: **797.51**
-   Weighted LTV: **44.85%**
-   Weighted DTI: **21.89%**
-   Weighted 12-month PD: **0.0006%**
-   Maximum state share: **12.75%**

Constraints:

-   weighted FICO ≥ 740
-   weighted LTV ≤ 80%
-   weighted DTI ≤ 42%
-   maximum state concentration ≤ 20%
-   target balance within ±1% of \$250M

This is a transparent selection heuristic, not an issuer optimization
engine.

### 9. Stress Testing

  Scenario     Stressed PD   Stressed LGD   Expected loss   EL rate
  ---------- ------------- -------------- --------------- ---------
  Base             0.0143%         45.10%          \$857K   0.0065%
  Moderate         0.0215%         54.12%        \$1.543M   0.0116%
  Severe           0.0286%         63.14%        \$2.400M   0.0181%

Moderate stress applies 1.5× PD and 1.2× LGD. Severe stress applies 2.0×
PD and 1.4× LGD.

## Technology

-   Python
-   pandas / NumPy
-   scikit-learn
-   SQL / MySQL
-   SQLAlchemy
-   SEC ABS-EE XML
-   Freddie Mac loan-level performance data
-   FRED / Freddie Mac PMMS
-   Git / GitHub

## Recommended Repository Structure

``` text
structured-finance-securitization/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── etl/
│   ├── analytics/
│   ├── models/
│   └── common/
├── sql/
├── reports/
├── docs/
│   ├── METHODOLOGY.md
│   └── LIMITATIONS.md
└── data/
    └── README.md
```

Raw Freddie and SEC files should not be committed to GitHub.

## Key Limitations

-   Freddie sample vintages are not the entire Freddie mortgage
    universe.
-   Recent-vintage 12-month credit events are sparse.
-   PD outputs are portfolio-project estimates, not production
    underwriting probabilities.
-   Empirical CPR scenarios summarize historical sample behavior and are
    not forecasts.
-   Fixed-income security assumptions are stylized.
-   Stress multipliers are explicit scenario assumptions rather than
    macroeconomically calibrated forecasts.
-   The pool-selection algorithm is a transparent heuristic, not an
    issuer's production optimizer.
-   Freddie residential data and SEC CMBS data are separate analytical
    workstreams.

## Portfolio Purpose

The project demonstrates how loan-level data can be transformed into
securitization-relevant analytics spanning collateral stratification,
credit performance, prepayment behavior, expected loss, fixed-income
cash flows, asset selection, and stress testing.
