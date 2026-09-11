import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sf_common import (
    REPORT_DIR,
    TEST_VINTAGE,
    build_preprocessor,
    get_engine,
    load_12m_snapshot_dataset,
    safe_metrics,
)

NUMERIC = [
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
    "current_actual_upb",
    "current_interest_rate",
    "estimated_ltv",
    "delinquency_numeric",
    "rate_spread_to_period_median",
]

CATEGORICAL = [
    "first_time_homebuyer_flag",
    "occupancy_status",
    "channel",
    "property_state",
    "property_type",
    "loan_purpose",
    "super_conforming_flag",
    "harp_indicator",
    "interest_only_indicator",
    "current_loan_delinquency_status",
]

TARGET = "credit_event_12m"
MIN_EVENTS_FOR_STABLE_DISCRIMINATION = 10


def make_pipeline():
    """Build a fresh model for each out-of-time fold."""
    pre = build_preprocessor(NUMERIC, CATEGORICAL)

    # IMPORTANT:
    # Do not use class_weight='balanced' or any over/under-sampling here.
    # Those methods change the effective class prior and can badly distort
    # raw probability estimates unless followed by explicit calibration.
    model = LogisticRegression(
        max_iter=1500,
        solver="saga",
        penalty="l2",
        C=0.5,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([
        ("prep", pre),
        ("model", model),
    ])


def evaluate_fold(df: pd.DataFrame, validation_vintage: int):
    """
    Expanding-window out-of-time validation.

    Example:
      validation 2008 -> train 2006, 2007
      validation 2019 -> train 2006, 2007, 2008
      validation 2020 -> train everything earlier
      ...
    """
    train = df[df["vintage_year"] < validation_vintage].copy()
    test = df[df["vintage_year"] == validation_vintage].copy()

    if train.empty or test.empty:
        return None, None

    train_events = int(train[TARGET].sum())
    test_events = int(test[TARGET].sum())

    if train[TARGET].nunique() < 2:
        warnings.warn(
            f"Skipping vintage {validation_vintage}: training data has only one target class."
        )
        return None, None

    pipe = make_pipeline()
    pipe.fit(train[NUMERIC + CATEGORICAL], train[TARGET])

    test_prob = pipe.predict_proba(test[NUMERIC + CATEGORICAL])[:, 1]
    train_prob = pipe.predict_proba(train[NUMERIC + CATEGORICAL])[:, 1]

    test_metrics = safe_metrics(test[TARGET], test_prob)
    train_metrics = safe_metrics(train[TARGET], train_prob)

    # AUC/AP can be numerically calculated with very few events, but such
    # estimates are not stable enough to present as strong model evidence.
    discrimination_status = (
        "adequate_event_count"
        if test_events >= MIN_EVENTS_FOR_STABLE_DISCRIMINATION
        else "too_few_events_for_stable_auc_ap"
    )

    fold_metrics = {
        "validation_vintage": validation_vintage,
        "training_vintages": ",".join(
            map(str, sorted(train["vintage_year"].dropna().astype(int).unique()))
        ),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_events_12m": train_events,
        "test_events_12m": test_events,
        "train_event_rate_12m": float(train[TARGET].mean()),
        "test_event_rate_12m": float(test[TARGET].mean()),
        "train_mean_pd_12m": float(np.mean(train_prob)),
        "test_mean_pd_12m": float(np.mean(test_prob)),
        "roc_auc": test_metrics["roc_auc"],
        "average_precision": test_metrics["average_precision"],
        "brier_score": test_metrics["brier_score"],
        "calibration_ratio_pred_to_obs": test_metrics[
            "calibration_ratio_pred_to_obs"
        ],
        "train_calibration_ratio": train_metrics[
            "calibration_ratio_pred_to_obs"
        ],
        "discrimination_status": discrimination_status,
    }

    predictions = test[[
        "loan_sequence_number",
        "vintage_year",
        "snapshot_period",
        "current_actual_upb",
        "credit_score",
        "original_ltv",
        "original_cltv",
        "original_dti_ratio",
        "property_state",
        "current_interest_rate",
        "estimated_ltv",
        TARGET,
    ]].copy()
    predictions["pd_12m"] = test_prob
    predictions["validation_vintage"] = validation_vintage

    return fold_metrics, predictions


def build_pooled_oot_metrics(oot_predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate all genuine out-of-time predictions across validation vintages."""
    m = safe_metrics(oot_predictions[TARGET], oot_predictions["pd_12m"])
    events = int(oot_predictions[TARGET].sum())

    return pd.DataFrame([{
        "oot_rows": len(oot_predictions),
        "oot_events_12m": events,
        "oot_event_rate_12m": float(oot_predictions[TARGET].mean()),
        "oot_mean_pd_12m": float(oot_predictions["pd_12m"].mean()),
        "oot_roc_auc": m["roc_auc"],
        "oot_average_precision": m["average_precision"],
        "oot_brier_score": m["brier_score"],
        "oot_calibration_ratio_pred_to_obs": m[
            "calibration_ratio_pred_to_obs"
        ],
        "discrimination_status": (
            "adequate_event_count"
            if events >= MIN_EVENTS_FOR_STABLE_DISCRIMINATION
            else "too_few_events_for_stable_auc_ap"
        ),
    }])


def save_2022_outputs(predictions_2022: pd.DataFrame, fold_2022: dict):
    """
    Preserve the filenames expected by downstream Step 6 while adding
    richer out-of-time validation files separately.
    """
    if predictions_2022 is None or predictions_2022.empty:
        raise RuntimeError("No 2022 out-of-time predictions were produced.")

    scores = predictions_2022.drop(columns=["validation_vintage"]).copy()
    scores.to_csv(REPORT_DIR / "credit_risk_scores_2022.csv", index=False)

    # Keep downstream/backward-compatible metrics file as the 2022 holdout row.
    pd.DataFrame([fold_2022]).to_csv(
        REPORT_DIR / "credit_risk_model_metrics.csv", index=False
    )

    # Risk-decile diagnostics. With only a tiny number of defaults in a recent
    # vintage, use this mainly for ranking inspection, not event-rate precision.
    try:
        scores["pd_decile"] = (
            pd.qcut(scores["pd_12m"], 10, labels=False, duplicates="drop") + 1
        )
        deciles = scores.groupby("pd_decile", as_index=False).agg(
            loans=("loan_sequence_number", "count"),
            mean_pd_12m=("pd_12m", "mean"),
            observed_event_rate_12m=(TARGET, "mean"),
            events_12m=(TARGET, "sum"),
            ead=("current_actual_upb", "sum"),
        )
        deciles.to_csv(
            REPORT_DIR / "credit_risk_decile_calibration.csv", index=False
        )
    except ValueError:
        pass


def main():
    engine = get_engine()

    print("\nBuilding month-12 snapshot / 12-month forward credit dataset...")
    df = load_12m_snapshot_dataset(engine)
    df.to_csv(REPORT_DIR / "modeling_snapshot_12m.csv", index=False)

    available_vintages = sorted(
        df["vintage_year"].dropna().astype(int).unique().tolist()
    )

    # With the user's sampled vintages this becomes:
    # 2008, 2019, 2020, 2021, 2022.
    # Each validation vintage must have at least two earlier vintages.
    validation_vintages = [
        v for i, v in enumerate(available_vintages) if i >= 2
    ]

    print("\nOUT-OF-TIME VALIDATION PLAN")
    print(f"Available vintages : {available_vintages}")
    print(f"Validation vintages: {validation_vintages}")

    fold_rows = []
    prediction_frames = []
    predictions_2022 = None
    fold_2022 = None

    for vintage in validation_vintages:
        print(f"\nRunning OOT fold: validation vintage {vintage} ...")
        fold, preds = evaluate_fold(df, vintage)

        if fold is None:
            continue

        fold_rows.append(fold)
        prediction_frames.append(preds)

        print(
            f"  train rows/events: {fold['train_rows']:,} / "
            f"{fold['train_events_12m']:,}"
        )
        print(
            f"  test rows/events : {fold['test_rows']:,} / "
            f"{fold['test_events_12m']:,}"
        )
        print(
            f"  observed rate    : {fold['test_event_rate_12m']:.6%}"
        )
        print(
            f"  mean predicted PD: {fold['test_mean_pd_12m']:.6%}"
        )
        print(
            f"  AUC / AP / Brier : {fold['roc_auc']:.6f} / "
            f"{fold['average_precision']:.6f} / "
            f"{fold['brier_score']:.6f}"
        )
        print(
            "  metric status    : "
            f"{fold['discrimination_status']}"
        )

        if vintage == TEST_VINTAGE:
            predictions_2022 = preds.copy()
            fold_2022 = fold.copy()

    if not fold_rows:
        raise RuntimeError("No valid out-of-time folds were produced.")

    fold_metrics = pd.DataFrame(fold_rows)
    oot_predictions = pd.concat(prediction_frames, ignore_index=True)
    pooled = build_pooled_oot_metrics(oot_predictions)

    fold_metrics.to_csv(
        REPORT_DIR / "credit_risk_oot_vintage_metrics.csv", index=False
    )
    oot_predictions.to_csv(
        REPORT_DIR / "credit_risk_oot_predictions.csv", index=False
    )
    pooled.to_csv(
        REPORT_DIR / "credit_risk_oot_pooled_metrics.csv", index=False
    )

    if fold_2022 is None:
        raise RuntimeError(
            f"Expected final holdout vintage {TEST_VINTAGE} was not available."
        )

    save_2022_outputs(predictions_2022, fold_2022)

    print("\n" + "=" * 100)
    print("CREDIT RISK OUT-OF-TIME VALIDATION BY VINTAGE")
    print("=" * 100)

    display_cols = [
        "validation_vintage",
        "test_rows",
        "test_events_12m",
        "test_event_rate_12m",
        "test_mean_pd_12m",
        "roc_auc",
        "average_precision",
        "brier_score",
        "calibration_ratio_pred_to_obs",
        "discrimination_status",
    ]
    print(fold_metrics[display_cols].to_string(index=False))

    print("\nPOOLED GENUINE OUT-OF-TIME PERFORMANCE")
    print(pooled.to_string(index=False))

    print("\nInterpretation rules:")
    print("- Target = credit-event termination (02/03/09) within 12 months after month-12 snapshot.")
    print("- Each validation vintage is scored only by a model trained on earlier vintages.")
    print("- No class_weight='balanced', over-sampling, or under-sampling is used.")
    print("- AUC/AP from a vintage with fewer than 10 events is flagged as statistically unstable.")
    print("- Use pooled OOT results plus vintage-by-vintage calibration as the primary validation evidence.")
    print("- credit_risk_scores_2022.csv remains the downstream PD input for expected loss.")

    print("\nCreated/updated:")
    for name in [
        "credit_risk_oot_vintage_metrics.csv",
        "credit_risk_oot_pooled_metrics.csv",
        "credit_risk_oot_predictions.csv",
        "credit_risk_model_metrics.csv",
        "credit_risk_scores_2022.csv",
        "credit_risk_decile_calibration.csv",
    ]:
        print(f"  {REPORT_DIR / name}")

    print("\nStep 4 complete with expanding-window out-of-time validation.")


if __name__ == "__main__":
    main()
