"""
Script for AT1 predictive models and AT2 exploratory regression

REQUIRED RUN SCRIPTS:
    python src/fetch_data.py
    python src/prepare_tables.py
    python src/prepare_data.py
    python src/analysis.py

The script reads both model-ready files created by ``analysis.py``. 
For AT1 it uses TRAIN (2021-2023) for fitting,
VALIDATION (2024) for all hyperparameter and feature-set selection, 
and evaluates the locked TEST set (2025)  
For AT2 it reports an exploratory Spearman association, 
univariable OLS-HC3, and OLS-HC3 adjusted for province and cyclic calendar month.

"""

# =============================================================================
# 0. Package settings
# =============================================================================

from __future__ import annotations

from itertools import product
from pathlib import Path
import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# =============================================================================
# 1. PROJECT SETTINGS AND FROZEN MODEL CONTRACT
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_project_dir() -> Path:
    explicit = os.environ.get("PROJECT_ROOT", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    if SCRIPT_DIR.name.casefold() == "src":
        return SCRIPT_DIR.parent
    return SCRIPT_DIR


PROJECT_DIR = resolve_project_dir()
MODEL_READY_DIR = PROJECT_DIR / "data" / "processed" / "model_ready"
AT1_INPUT_PATH = MODEL_READY_DIR / "at1_model_ready.csv"
AT2_INPUT_PATH = MODEL_READY_DIR / "at2_model_ready.csv"
MODELING_OUTPUT_DIR = PROJECT_DIR / "outputs" / "modeling"
RESULT_DIR = MODELING_OUTPUT_DIR / "results"
MODEL_DIR = PROJECT_DIR / "outputs" / "models"
ANALYSIS_TABLE_DIR = PROJECT_DIR / "outputs" / "analysis" / "tables"
FIGURE_DIR = PROJECT_DIR / "outputs" / "analysis" / "figures"

CSV_ENCODING = "utf-8-sig"
TARGET = "observed_pm25_median_ugm3"
BASELINE = "persistence_baseline_prediction"
RANDOM_STATE = 1
WARNING_THRESHOLD = 37.5

PM_FEATURES = [
    "observed_pm25_median_lag1",
    "observed_pm25_median_lag2",
    "observed_pm25_median_lag3",
]
WEATHER_FEATURES = [
    "temperature_mean_c_lag1",
    "relative_humidity_mean_pct_lag1",
    "precipitation_mm_lag1",
    "wind_speed_mean_kmh_lag1",
    "wind_direction_sin_lag1",
    "wind_direction_cos_lag1",
    "surface_pressure_mean_hpa_lag1",
]
HOTSPOT_FEATURES = [
    f"hotspot_count_thailand_{band}_lag{lag}"
    for band in ["000_050km", "050_100km", "100_300km", "300_500km"]
    for lag in [1, 2, 3]
]
CALENDAR_FEATURES = ["target_day_of_year_sin", "target_day_of_year_cos"]

NO_HOTSPOT_FEATURES = ["province_key"] + PM_FEATURES + WEATHER_FEATURES + CALENDAR_FEATURES
WITH_HOTSPOT_FEATURES = NO_HOTSPOT_FEATURES + HOTSPOT_FEATURES

MODEL_FEATURES = {
    "RF_NO_HOTSPOTS": NO_HOTSPOT_FEATURES,
    "RF_WITH_HOTSPOTS": WITH_HOTSPOT_FEATURES,
}

# The grid is fully declared before TEST is evaluated. The old report's chosen
# depth=12 and leaf=2 remain candidates, but current results select afresh using
# the rebuilt 2024 validation set.
PARAMETER_GRID = [
    {
        "n_estimators": 300,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "max_features": 1.0,
    }
    for max_depth, min_samples_leaf in product([8, 12, None], [1, 2])
]

FIGURES = {
    "fig04_model_comparison.png": "MAIN_FIGURE_4",
    "fig05_warning_error_diagnostics.png": "MAIN_FIGURE_5",
    "fig06_at2_exploratory_association.png": "MAIN_FIGURE_6",
    "figS07_test_predictions_detail.png": "SUPPLEMENT_FIGURE_S7",
    "figS08_error_by_province_season_detail.png": "SUPPLEMENT_FIGURE_S8",
}

LEGACY_FIGURES = [
    "fig02_model_comparison.png",
    "fig03_test_predictions.png",
    "fig04_error_by_province_season.png",
    "fig05_at2_exploratory_association.png",
]


# =============================================================================
# 2. INPUT AND OUTPUT HELPERS
# =============================================================================

def ensure_directories() -> None:
    for directory in [RESULT_DIR, MODEL_DIR, ANALYSIS_TABLE_DIR, FIGURE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def remove_legacy_figures() -> None:
    """Remove only obsolete files written by the preceding modeling.py version."""
    for filename in LEGACY_FIGURES:
        path = FIGURE_DIR / filename
        if path.is_file():
            path.unlink()


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    output = frame.copy()
    for column in output.select_dtypes(include=["datetime", "datetimetz"]).columns:
        output[column] = output[column].dt.strftime("%Y-%m-%d")
    temporary = path.with_suffix(path.suffix + ".part")
    output.to_csv(temporary, index=False, encoding=CSV_ENCODING)
    temporary.replace(path)


def save_figure(fig: plt.Figure, filename: str) -> None:
    path = FIGURE_DIR / filename
    temporary = path.with_suffix(path.suffix + ".part")
    fig.savefig(
        temporary, format="png", dpi=300, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)
    temporary.replace(path)


def load_model_data() -> pd.DataFrame:
    if not AT1_INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {AT1_INPUT_PATH}. Run python src/analysis.py before modeling.py."
        )
    data = pd.read_csv(AT1_INPUT_PATH, encoding=CSV_ENCODING, low_memory=False)
    data.columns = [str(column).replace("\ufeff", "").strip() for column in data]
    data["analysis_date"] = pd.to_datetime(data["analysis_date"], errors="coerce")
    data["province_key"] = (
        data["province_key"].astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(2)
    )
    for column in [TARGET, BASELINE] + [
        feature for feature in WITH_HOTSPOT_FEATURES if feature != "province_key"
    ]:
        if column in data:
            data[column] = numeric(data[column])
    return data


def load_at2_model_data() -> pd.DataFrame:
    if not AT2_INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {AT2_INPUT_PATH}. Run python src/analysis.py before modeling.py."
        )
    data = pd.read_csv(AT2_INPUT_PATH, encoding=CSV_ENCODING, low_memory=False)
    data.columns = [str(column).replace("\ufeff", "").strip() for column in data]
    data["month_start"] = pd.to_datetime(data["month_start"], errors="coerce")
    data["province_key"] = (
        data["province_key"].astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(2)
    )
    for column in [
        "respiratory_records_per_active_facility",
        "pm25_monthly_median_ugm3",
        "month_sin",
        "month_cos",
    ]:
        if column in data:
            data[column] = numeric(data[column])
    return data


def validate_model_data(data: pd.DataFrame) -> None:
    required = [
        "province_key", "province_name_en", "analysis_date", "data_split",
        TARGET, BASELINE,
    ] + WITH_HOTSPOT_FEATURES
    missing_columns = [column for column in required if column not in data]
    if missing_columns:
        raise ValueError(f"at1_model_ready.csv is missing columns: {missing_columns}")
    if data.duplicated(["province_key", "analysis_date"]).any():
        raise ValueError("Duplicate province-date keys were found in model-ready data")
    required_numeric = [TARGET, BASELINE] + [
        feature for feature in WITH_HOTSPOT_FEATURES if feature != "province_key"
    ]
    if data[required_numeric].isna().any().any():
        failed = data[required_numeric].columns[data[required_numeric].isna().any()].tolist()
        raise ValueError(f"Required model values contain missing data: {failed}")
    split = data["data_split"].astype("string").str.upper()
    unexpected = sorted(set(split.dropna()) - {"TRAIN", "VALIDATION", "TEST"})
    if unexpected:
        raise ValueError(f"Unexpected data_split values: {unexpected}")
    if any((split == label).sum() == 0 for label in ["TRAIN", "VALIDATION", "TEST"]):
        raise ValueError("TRAIN, VALIDATION, and TEST must all contain rows")
    split_ranges = data.assign(data_split=split).groupby("data_split")["analysis_date"].agg(
        ["min", "max"]
    )
    if not (
        split_ranges.loc["TRAIN", "max"] < split_ranges.loc["VALIDATION", "min"]
        <= split_ranges.loc["VALIDATION", "max"] < split_ranges.loc["TEST", "min"]
    ):
        raise ValueError("Chronological splits overlap or are out of order")


def validate_at2_model_data(data: pd.DataFrame) -> None:
    required = [
        "province_key", "province_name_en", "month_start", "year_month",
        "respiratory_records_per_active_facility",
        "pm25_monthly_median_ugm3", "month_sin", "month_cos",
    ]
    missing_columns = [column for column in required if column not in data]
    if missing_columns:
        raise ValueError(f"at2_model_ready.csv is missing columns: {missing_columns}")
    if data.duplicated(["province_key", "year_month"]).any():
        raise ValueError("Duplicate province-month keys were found in AT2 model-ready data")
    if data[required].isna().any().any():
        failed = data[required].columns[data[required].isna().any()].tolist()
        raise ValueError(f"Required AT2 model values contain missing data: {failed}")
    if len(data) < 3:
        raise ValueError("AT2 model-ready data must contain at least three rows")


# =============================================================================
# 3. MODEL FITTING, SELECTION, AND METRICS
# =============================================================================

def make_pipeline(features: list[str], parameters: dict[str, object]) -> Pipeline:
    numeric_features = [feature for feature in features if feature != "province_key"]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "province",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["province_key"],
            ),
            ("numeric", "passthrough", numeric_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    regressor = RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        bootstrap=True,
        **parameters,
    )
    return Pipeline([("preprocessor", preprocessor), ("regressor", regressor)])


def regression_metrics(observed: pd.Series | np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    observed_array = np.asarray(observed, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    return {
        "n_rows": len(observed_array),
        "mae": float(mean_absolute_error(observed_array, predicted_array)),
        "rmse": float(np.sqrt(mean_squared_error(observed_array, predicted_array))),
        "r_squared": float(r2_score(observed_array, predicted_array)),
    }


def tune_random_forests(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]], dict[str, Pipeline]]:
    rows: list[dict[str, object]] = []
    fitted_candidates: dict[tuple[str, int], Pipeline] = {}
    for model_name, features in MODEL_FEATURES.items():
        for candidate_id, parameters in enumerate(PARAMETER_GRID, start=1):
            pipeline = make_pipeline(features, parameters)
            pipeline.fit(train[features], train[TARGET])
            prediction = pipeline.predict(validation[features])
            metrics = regression_metrics(validation[TARGET], prediction)
            rows.append({
                "model": model_name,
                "candidate_id": candidate_id,
                "data_split": "VALIDATION",
                **parameters,
                **metrics,
            })
            fitted_candidates[(model_name, candidate_id)] = pipeline
    tuning = pd.DataFrame(rows).sort_values(
        ["model", "mae", "rmse", "max_depth", "min_samples_leaf"],
        na_position="last",
    ).reset_index(drop=True)

    best_parameters: dict[str, dict[str, object]] = {}
    best_pipelines: dict[str, Pipeline] = {}
    for model_name in MODEL_FEATURES:
        best_row = tuning.loc[tuning["model"].eq(model_name)].iloc[0]
        candidate_id = int(best_row["candidate_id"])
        best_parameters[model_name] = PARAMETER_GRID[candidate_id - 1].copy()
        best_pipelines[model_name] = fitted_candidates[(model_name, candidate_id)]
    return tuning, best_parameters, best_pipelines


def validation_summary(
    validation: pd.DataFrame,
    tuning: pd.DataFrame,
    best_parameters: dict[str, dict[str, object]],
) -> pd.DataFrame:
    rows = [{
        "model": "PERSISTENCE",
        "data_split": "VALIDATION",
        "selected_using": "NOT_APPLICABLE_BASELINE",
        **regression_metrics(validation[TARGET], validation[BASELINE]),
    }]
    for model_name in MODEL_FEATURES:
        parameters = best_parameters[model_name]
        matches = tuning.loc[
            tuning["model"].eq(model_name)
            & tuning["max_depth"].fillna(-1).eq(
                -1 if parameters["max_depth"] is None else parameters["max_depth"]
            )
            & tuning["min_samples_leaf"].eq(parameters["min_samples_leaf"])
        ]
        row = matches.iloc[0]
        rows.append({
            "model": model_name,
            "data_split": "VALIDATION",
            "selected_using": "LOWEST_2024_VALIDATION_MAE_THEN_RMSE",
            "n_estimators": parameters["n_estimators"],
            "max_depth": parameters["max_depth"],
            "min_samples_leaf": parameters["min_samples_leaf"],
            "max_features": parameters["max_features"],
            "n_rows": int(row["n_rows"]),
            "mae": row["mae"],
            "rmse": row["rmse"],
            "r_squared": row["r_squared"],
        })
    return pd.DataFrame(rows)


def select_model_family(validation_results: pd.DataFrame) -> str:
    candidates = validation_results.loc[
        validation_results["model"].isin(MODEL_FEATURES)
    ].sort_values(["mae", "rmse", "model"])
    return str(candidates.iloc[0]["model"])


def refit_and_test(
    development: pd.DataFrame,
    test: pd.DataFrame,
    best_parameters: dict[str, dict[str, object]],
    selected_model: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Pipeline]]:
    predictions = test[[
        "province_key", "province_name_th", "province_name_en",
        "analysis_date", "data_split", TARGET, BASELINE,
    ]].copy()
    rows = [{
        "model": "PERSISTENCE",
        "data_split": "TEST",
        "selected_on_validation": False,
        **regression_metrics(test[TARGET], test[BASELINE]),
    }]
    fitted: dict[str, Pipeline] = {}
    for model_name, features in MODEL_FEATURES.items():
        pipeline = make_pipeline(features, best_parameters[model_name])
        pipeline.fit(development[features], development[TARGET])
        prediction = pipeline.predict(test[features])
        predictions[f"prediction_{model_name.casefold()}"] = prediction
        rows.append({
            "model": model_name,
            "data_split": "TEST",
            "selected_on_validation": model_name == selected_model,
            **best_parameters[model_name],
            **regression_metrics(test[TARGET], prediction),
        })
        fitted[model_name] = pipeline

    selected_column = f"prediction_{selected_model.casefold()}"
    predictions["selected_model"] = selected_model
    predictions["selected_prediction"] = predictions[selected_column]
    predictions["selected_residual_observed_minus_predicted"] = (
        predictions[TARGET] - predictions["selected_prediction"]
    )
    predictions["observed_warning"] = predictions[TARGET].ge(WARNING_THRESHOLD)
    predictions["persistence_warning"] = predictions[BASELINE].ge(WARNING_THRESHOLD)
    predictions["selected_warning"] = predictions["selected_prediction"].ge(WARNING_THRESHOLD)
    predictions["season"] = season_from_month(predictions["analysis_date"].dt.month)
    return pd.DataFrame(rows), predictions, fitted


def season_from_month(month: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [month.isin([11, 12, 1, 2]), month.isin([3, 4, 5])],
            ["COOL", "HOT"],
            default="RAINY",
        ),
        index=month.index,
        dtype="string",
    )


def warning_metrics(observed: pd.Series, predicted: pd.Series) -> dict[str, object]:
    actual = numeric(observed).ge(WARNING_THRESHOLD)
    forecast = numeric(predicted).ge(WARNING_THRESHOLD)
    tp = int((actual & forecast).sum())
    fn = int((actual & ~forecast).sum())
    fp = int((~actual & forecast).sum())
    tn = int((~actual & ~forecast).sum())
    recall = tp / (tp + fn) if tp + fn else np.nan
    precision = tp / (tp + fp) if tp + fp else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else np.nan
    return {
        "threshold_ugm3": WARNING_THRESHOLD,
        "true_positive": tp, "false_negative": fn,
        "false_positive": fp, "true_negative": tn,
        "recall": recall, "precision": precision,
        "specificity": specificity, "f1": f1,
    }


def build_warning_table(predictions: pd.DataFrame) -> pd.DataFrame:
    prediction_columns = {
        "PERSISTENCE": BASELINE,
        "RF_NO_HOTSPOTS": "prediction_rf_no_hotspots",
        "RF_WITH_HOTSPOTS": "prediction_rf_with_hotspots",
        "SELECTED_RF": "selected_prediction",
    }
    return pd.DataFrame([
        {"model": model, "data_split": "TEST", **warning_metrics(predictions[TARGET], predictions[column])}
        for model, column in prediction_columns.items()
    ])


def build_error_table(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["absolute_error"] = frame["selected_residual_observed_minus_predicted"].abs()
    frame["squared_error"] = frame["selected_residual_observed_minus_predicted"].pow(2)
    result = frame.groupby(
        ["province_key", "province_name_en", "season"], observed=True
    ).agg(
        n_rows=(TARGET, "size"),
        mae=("absolute_error", "mean"),
        mean_residual_observed_minus_predicted=(
            "selected_residual_observed_minus_predicted", "mean"
        ),
        mean_squared_error=("squared_error", "mean"),
    ).reset_index()
    result["rmse"] = np.sqrt(result.pop("mean_squared_error"))
    return result.sort_values(["province_key", "season"])


def grouped_time_series_diagnostic(
    pretest: pd.DataFrame,
    model_name: str,
    parameters: dict[str, object],
) -> pd.DataFrame:
    unique_dates = np.array(sorted(pretest["analysis_date"].dropna().unique()))
    if len(unique_dates) < 10:
        return pd.DataFrame()
    splitter = TimeSeriesSplit(n_splits=min(4, len(unique_dates) - 1))
    rows: list[dict[str, object]] = []
    features = MODEL_FEATURES[model_name]
    for fold, (train_index, assessment_index) in enumerate(splitter.split(unique_dates), start=1):
        training_dates = set(unique_dates[train_index])
        assessment_dates = set(unique_dates[assessment_index])
        training = pretest.loc[pretest["analysis_date"].isin(training_dates)]
        assessment = pretest.loc[pretest["analysis_date"].isin(assessment_dates)]
        pipeline = make_pipeline(features, parameters)
        pipeline.fit(training[features], training[TARGET])
        model_prediction = pipeline.predict(assessment[features])
        fold_details = {
            "fold": fold,
            "train_start": training["analysis_date"].min(),
            "train_end": training["analysis_date"].max(),
            "assessment_start": assessment["analysis_date"].min(),
            "assessment_end": assessment["analysis_date"].max(),
            "train_rows": len(training),
            "assessment_rows": len(assessment),
            "assessment_unique_dates": assessment["analysis_date"].nunique(),
        }
        rows.extend([
            {"model": model_name, **fold_details, **regression_metrics(assessment[TARGET], model_prediction)},
            {"model": "PERSISTENCE", **fold_details, **regression_metrics(assessment[TARGET], assessment[BASELINE])},
        ])
    return pd.DataFrame(rows)


def build_feature_importance(pipeline: Pipeline, model_name: str) -> pd.DataFrame:
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importance = pipeline.named_steps["regressor"].feature_importances_
    result = pd.DataFrame({
        "model": model_name,
        "transformed_feature": names,
        "importance": importance,
    })
    result["feature"] = (
        result["transformed_feature"]
        .str.replace(r"^numeric__", "", regex=True)
        .str.replace(r"^province__", "", regex=True)
    )
    return result.sort_values("importance", ascending=False).reset_index(drop=True)


# =============================================================================
# 4. EXPLORATORY AT2 REGRESSION
# =============================================================================

def at2_spearman(model: pd.DataFrame) -> dict[str, float]:
    pairs = model[["pm25_monthly_median_ugm3", "respiratory_rate"]].dropna()
    if (
        len(pairs) < 3
        or pairs["pm25_monthly_median_ugm3"].nunique() < 2
        or pairs["respiratory_rate"].nunique() < 2
    ):
        return {"paired_n": len(pairs), "spearman_rho": np.nan, "spearman_p": np.nan}
    result = stats.spearmanr(
        pairs["pm25_monthly_median_ugm3"], pairs["respiratory_rate"]
    )
    return {
        "paired_n": len(pairs),
        "spearman_rho": float(result.statistic),
        "spearman_p": float(result.pvalue),
    }


def fit_ols_hc3(
    model: pd.DataFrame,
    *,
    adjusted: bool,
) -> dict[str, float]:
    """Fit OLS and calculate HC3 covariance without formula-side effects."""
    pm25 = numeric(model["pm25_monthly_median_ugm3"]).to_numpy(float)
    outcome = numeric(model["respiratory_rate"]).to_numpy(float)
    parts = [np.ones(len(model)), pm25]
    if adjusted:
        province_dummies = pd.get_dummies(
            model["province_key"].astype("string"), drop_first=True, dtype=float
        )
        parts.extend(province_dummies[column].to_numpy(float) for column in province_dummies)
        parts.extend([
            numeric(model["month_sin"]).to_numpy(float),
            numeric(model["month_cos"]).to_numpy(float),
        ])
    design = np.column_stack(parts)
    xtx_inverse = np.linalg.pinv(design.T @ design)
    beta = xtx_inverse @ design.T @ outcome
    fitted = design @ beta
    residual = outcome - fitted
    leverage = np.einsum("ij,jk,ik->i", design, xtx_inverse, design)
    leverage = np.clip(leverage, 0, 1 - np.finfo(float).eps)
    hc3_scale = np.square(residual / (1 - leverage))
    meat = design.T @ (design * hc3_scale[:, None])
    covariance = xtx_inverse @ meat @ xtx_inverse
    standard_error = float(np.sqrt(max(covariance[1, 1], 0)))
    rank = int(np.linalg.matrix_rank(design))
    degrees_freedom = max(len(model) - rank, 1)
    if standard_error > 0:
        statistic = float(beta[1] / standard_error)
        p_value = float(2 * stats.t.sf(abs(statistic), degrees_freedom))
        critical = float(stats.t.ppf(0.975, degrees_freedom))
    else:
        p_value = np.nan
        critical = np.nan
    total_sum_squares = float(np.square(outcome - outcome.mean()).sum())
    r_squared = (
        1 - float(np.square(residual).sum()) / total_sum_squares
        if total_sum_squares > 0 else np.nan
    )
    return {
        "estimate": float(beta[1]),
        "std_error": standard_error,
        "ci_lower": float(beta[1] - critical * standard_error),
        "ci_upper": float(beta[1] + critical * standard_error),
        "p_value": p_value,
        "n_rows": len(model),
        "r_squared": r_squared,
    }


def coefficient_row(
    result: dict[str, float], analysis: str, adjustment: str, n_provinces: int
) -> dict[str, object]:
    return {
        "analysis": analysis,
        "effect_measure": "OLS coefficient per 1 ug/m3 higher monthly median PM2.5",
        "estimate": result["estimate"],
        "std_error": result["std_error"],
        "ci_lower": result["ci_lower"],
        "ci_upper": result["ci_upper"],
        "p_value": result["p_value"],
        "n_rows": int(result["n_rows"]),
        "n_provinces": n_provinces,
        "r_squared": result["r_squared"],
        "adjustment": adjustment,
        "covariance": "HC3",
        "interpretation_limit": "EXPLORATORY_ASSOCIATION; NOT_CAUSAL",
    }


def build_pm25_respi(at2: pd.DataFrame) -> pd.DataFrame:
    model = at2.copy().rename(
        columns={"respiratory_records_per_active_facility": "respiratory_rate"}
    )
    required = [
        "respiratory_rate", "pm25_monthly_median_ugm3",
        "province_key", "month_sin", "month_cos",
    ]
    model = model.dropna(subset=required)
    n_provinces = model["province_key"].nunique()
    rows: list[dict[str, object]] = []
    correlation = at2_spearman(model)
    rows.append({
        "analysis": "UNIVARIABLE_SPEARMAN",
        "effect_measure": "Spearman rho",
        "estimate": correlation["spearman_rho"],
        "std_error": np.nan, "ci_lower": np.nan, "ci_upper": np.nan,
        "p_value": correlation["spearman_p"], "n_rows": correlation["paired_n"],
        "n_provinces": n_provinces, "r_squared": np.nan,
        "adjustment": "NONE", "covariance": "NOT_APPLICABLE",
        "interpretation_limit": "EXPLORATORY_ASSOCIATION; NOT_CAUSAL",
    })
    if len(model) >= 5 and model["pm25_monthly_median_ugm3"].nunique() >= 2:
        univariable = fit_ols_hc3(model, adjusted=False)
        rows.append(coefficient_row(
            univariable, "UNIVARIABLE_OLS_HC3", "NONE", n_provinces
        ))
    minimum_adjusted_rows = n_provinces + 6
    if (
        len(model) >= minimum_adjusted_rows
        and n_provinces >= 2
        and model["pm25_monthly_median_ugm3"].nunique() >= 2
    ):
        adjusted = fit_ols_hc3(model, adjusted=True)
        rows.append(coefficient_row(
            adjusted,
            "MULTIVARIABLE_OLS_HC3",
            "PROVINCE_FIXED_EFFECTS; CYCLIC_CALENDAR_MONTH",
            n_provinces,
        ))
    result = pd.DataFrame(rows)
    result["low_sample_warning"] = len(model) < 100
    result["outcome_definition"] = (
        "RESPIRATORY_DIAGNOSIS_RECORDS_PER_ACTIVE_REPORTING_FACILITY"
    )
    return result


def adjusted_at2_prediction_curve(at2: pd.DataFrame) -> pd.DataFrame:
    """Return marginal adjusted OLS predictions and HC3 confidence intervals."""
    model = at2.copy().rename(
        columns={"respiratory_records_per_active_facility": "respiratory_rate"}
    ).dropna(subset=[
        "respiratory_rate", "pm25_monthly_median_ugm3",
        "province_key", "month_sin", "month_cos",
    ])
    levels = sorted(model["province_key"].astype(str).unique())
    if (
        len(model) < len(levels) + 6
        or len(levels) < 2
        or model["pm25_monthly_median_ugm3"].nunique() < 2
    ):
        return pd.DataFrame()

    def design(frame: pd.DataFrame) -> np.ndarray:
        parts = [
            np.ones(len(frame)),
            numeric(frame["pm25_monthly_median_ugm3"]).to_numpy(float),
        ]
        province = frame["province_key"].astype(str)
        parts.extend(province.eq(level).astype(float).to_numpy() for level in levels[1:])
        parts.extend([
            numeric(frame["month_sin"]).to_numpy(float),
            numeric(frame["month_cos"]).to_numpy(float),
        ])
        return np.column_stack(parts)

    x = design(model)
    y = numeric(model["respiratory_rate"]).to_numpy(float)
    xtx_inverse = np.linalg.pinv(x.T @ x)
    beta = xtx_inverse @ x.T @ y
    residual = y - x @ beta
    leverage = np.einsum("ij,jk,ik->i", x, xtx_inverse, x)
    leverage = np.clip(leverage, 0, 1 - np.finfo(float).eps)
    hc3_scale = np.square(residual / (1 - leverage))
    covariance = xtx_inverse @ (x.T @ (x * hc3_scale[:, None])) @ xtx_inverse
    degrees_freedom = max(len(model) - np.linalg.matrix_rank(x), 1)
    critical = float(stats.t.ppf(0.975, degrees_freedom))

    pm25_grid = np.linspace(
        model["pm25_monthly_median_ugm3"].min(),
        model["pm25_monthly_median_ugm3"].max(),
        80,
    )
    rows = []
    for value in pm25_grid:
        counterfactual = model.copy()
        counterfactual["pm25_monthly_median_ugm3"] = value
        mean_design = design(counterfactual).mean(axis=0)
        estimate = float(mean_design @ beta)
        standard_error = float(np.sqrt(max(mean_design @ covariance @ mean_design, 0)))
        rows.append({
            "pm25_monthly_median_ugm3": value,
            "adjusted_mean": estimate,
            "ci_lower": estimate - critical * standard_error,
            "ci_upper": estimate + critical * standard_error,
        })
    return pd.DataFrame(rows)


# =============================================================================
# 5. REPORT FIGURES 4-6 AND SUPPLEMENTARY FIGURES S7-S8
# =============================================================================

def setup_plot_theme() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 9, "figure.titlesize": 14,
    })


def figure_model_comparison(validation: pd.DataFrame, test: pd.DataFrame) -> None:
    plot = pd.concat([
        validation[["model", "data_split", "mae"]],
        test[["model", "data_split", "mae"]],
    ], ignore_index=True)
    labels = {
        "PERSISTENCE": "Persistence",
        "RF_NO_HOTSPOTS": "RF without hotspots",
        "RF_WITH_HOTSPOTS": "RF with hotspots",
    }
    plot["Model"] = plot["model"].map(labels)
    plot["Evaluation period"] = plot["data_split"].map({
        "VALIDATION": "Validation 2024", "TEST": "Locked test 2025"
    })
    fig, ax = plt.subplots(figsize=(10, 5.8))
    sns.barplot(
        data=plot, x="Model", y="mae", hue="Evaluation period",
        palette=["#65A7F3", "#2864DC"], ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=9)
    ax.set(
        xlabel="", ylabel="MAE (µg/m³; lower is better)",
        title="Out-of-time model comparison",
    )
    ax.legend(title=None, frameon=False, loc="upper right")
    save_figure(fig, "fig04_model_comparison.png")


def figure_test_predictions(predictions: pd.DataFrame, selected_model: str) -> None:
    observed = predictions[TARGET].to_numpy(float)
    predicted = predictions["selected_prediction"].to_numpy(float)
    upper = float(max(
        np.nanmax(observed), np.nanmax(predicted), WARNING_THRESHOLD * 1.15
    ) * 1.04)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axvspan(WARNING_THRESHOLD, upper, ymin=0,
               ymax=min(WARNING_THRESHOLD / upper, 1), color="#FADADD", alpha=0.5)
    ax.axhspan(WARNING_THRESHOLD, upper, xmin=0,
               xmax=min(WARNING_THRESHOLD / upper, 1), color="#FFF4C2", alpha=0.55)
    ax.axvspan(WARNING_THRESHOLD, upper, ymin=min(WARNING_THRESHOLD / upper, 1),
               ymax=1, color="#DDF2DD", alpha=0.4)
    ax.scatter(observed, predicted, s=18, alpha=0.34, color="#3478DB", edgecolors="none")
    ax.plot([0, upper], [0, upper], linestyle="--", color="#6F7C87", label="Ideal 1:1")
    ax.axvline(WARNING_THRESHOLD, linestyle=":", color="#D9534F", linewidth=1.5)
    ax.axhline(WARNING_THRESHOLD, linestyle=":", color="#D9534F", linewidth=1.5)
    ax.text(0.68 * upper, 0.88 * upper, "Correct warning", color="#218739", fontweight="bold")
    ax.text(0.70 * upper, 0.07 * upper, "Missed warning", color="#C83232", fontweight="bold")
    ax.text(0.04 * upper, 0.88 * upper, "False warning", color="#A86C00", fontweight="bold")
    ax.set(
        xlim=(0, upper), ylim=(0, upper),
        xlabel="Observed PM2.5 (µg/m³)", ylabel="Predicted PM2.5 (µg/m³)",
        title=f"Selected {selected_model.replace('_', ' ')} on the locked 2025 test set",
    )
    ax.legend(frameon=False, loc="lower right")
    save_figure(fig, "figS07_test_predictions_detail.png")


def figure_error_heatmaps(errors: pd.DataFrame) -> None:
    province_order = (
        errors[["province_key", "province_name_en"]].drop_duplicates()
        .sort_values("province_key")["province_name_en"].tolist()
    )
    season_order = ["COOL", "HOT", "RAINY"]
    mae = errors.pivot(index="province_name_en", columns="season", values="mae")
    residual = errors.pivot(
        index="province_name_en", columns="season",
        values="mean_residual_observed_minus_predicted",
    )
    mae = mae.reindex(province_order).reindex(columns=season_order)
    residual = residual.reindex(province_order).reindex(columns=season_order)
    limit = float(np.nanmax(np.abs(residual.to_numpy()))) if residual.notna().any().any() else 1
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6))
    sns.heatmap(
        mae, cmap="YlOrRd", annot=True, fmt=".1f", linewidths=0.5,
        cbar_kws={"label": "MAE (µg/m³)"}, ax=axes[0],
    )
    sns.heatmap(
        residual, cmap="RdBu_r", center=0, vmin=-limit, vmax=limit,
        annot=True, fmt=".1f", linewidths=0.5,
        cbar_kws={"label": "Mean residual (µg/m³)"}, ax=axes[1],
    )
    axes[0].set(title="A. Error magnitude", xlabel="Season", ylabel="Province")
    axes[1].set(
        title="B. Error direction (observed - predicted)",
        xlabel="Season", ylabel="",
    )
    for ax in axes:
        ax.set_xticklabels([label.title() for label in season_order], rotation=0)
        ax.tick_params(axis="y", rotation=0)
    fig.suptitle("Selected-model error by province and season, locked test 2025", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "figS08_error_by_province_season_detail.png")


def figure_warning_error_diagnostics(
    predictions: pd.DataFrame, selected_model: str, errors: pd.DataFrame
) -> None:
    """Main Figure 5: warning classification, error magnitude, and error direction."""
    observed = predictions[TARGET].to_numpy(float)
    predicted = predictions["selected_prediction"].to_numpy(float)
    upper = float(max(
        np.nanmax(observed), np.nanmax(predicted), WARNING_THRESHOLD * 1.15
    ) * 1.04)
    province_order = (
        errors[["province_key", "province_name_en"]].drop_duplicates()
        .sort_values("province_key")["province_name_en"].tolist()
    )
    season_order = ["COOL", "HOT", "RAINY"]
    mae = errors.pivot(index="province_name_en", columns="season", values="mae")
    residual = errors.pivot(
        index="province_name_en", columns="season",
        values="mean_residual_observed_minus_predicted",
    )
    mae = mae.reindex(province_order).reindex(columns=season_order)
    residual = residual.reindex(province_order).reindex(columns=season_order)
    residual_limit = (
        float(np.nanmax(np.abs(residual.to_numpy())))
        if residual.notna().any().any() else 1
    )

    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.08, 1], hspace=0.42, wspace=0.34)
    ax_scatter = fig.add_subplot(grid[:, 0])
    ax_mae = fig.add_subplot(grid[0, 1])
    ax_residual = fig.add_subplot(grid[1, 1])

    threshold_fraction = min(WARNING_THRESHOLD / upper, 1)
    ax_scatter.axvspan(
        WARNING_THRESHOLD, upper, ymin=0, ymax=threshold_fraction,
        color="#FADADD", alpha=0.5,
    )
    ax_scatter.axhspan(
        WARNING_THRESHOLD, upper, xmin=0, xmax=threshold_fraction,
        color="#FFF4C2", alpha=0.55,
    )
    ax_scatter.axvspan(
        WARNING_THRESHOLD, upper, ymin=threshold_fraction, ymax=1,
        color="#DDF2DD", alpha=0.4,
    )
    ax_scatter.scatter(
        observed, predicted, s=18, alpha=0.34, color="#3478DB", edgecolors="none",
    )
    ax_scatter.plot(
        [0, upper], [0, upper], linestyle="--", color="#6F7C87", label="Ideal 1:1",
    )
    ax_scatter.axvline(WARNING_THRESHOLD, linestyle=":", color="#D9534F", linewidth=1.5)
    ax_scatter.axhline(WARNING_THRESHOLD, linestyle=":", color="#D9534F", linewidth=1.5)
    ax_scatter.text(0.68 * upper, 0.88 * upper, "Correct warning", color="#218739",
                    fontweight="bold")
    ax_scatter.text(0.70 * upper, 0.07 * upper, "Missed warning", color="#C83232",
                    fontweight="bold")
    ax_scatter.text(0.04 * upper, 0.88 * upper, "False warning", color="#A86C00",
                    fontweight="bold")
    ax_scatter.set(
        xlim=(0, upper), ylim=(0, upper),
        xlabel="Observed PM2.5 (µg/m³)", ylabel="Predicted PM2.5 (µg/m³)",
        title=f"A. Warning classification: {selected_model.replace('_', ' ')}",
    )
    ax_scatter.legend(frameon=False, loc="lower right")

    sns.heatmap(
        mae, cmap="YlOrRd", annot=True, fmt=".1f", linewidths=0.5,
        cbar_kws={"label": "MAE (µg/m³)"}, ax=ax_mae,
    )
    sns.heatmap(
        residual, cmap="RdBu_r", center=0, vmin=-residual_limit, vmax=residual_limit,
        annot=True, fmt=".1f", linewidths=0.5,
        cbar_kws={"label": "Mean residual (µg/m³)"}, ax=ax_residual,
    )
    ax_mae.set(title="B. Error magnitude", xlabel="Season", ylabel="Province")
    ax_residual.set(
        title="C. Error direction (observed - predicted)",
        xlabel="Season", ylabel="Province",
    )
    for ax in [ax_mae, ax_residual]:
        ax.set_xticklabels([label.title() for label in season_order], rotation=0)
        ax.tick_params(axis="y", rotation=0)
    fig.suptitle(
        "Selected-model warning performance and failure pattern, locked test 2025",
        fontweight="bold", y=0.99,
    )
    save_figure(fig, "fig05_warning_error_diagnostics.png")


def figure_respiratory_context(at2: pd.DataFrame) -> None:
    """Create the agreed two-panel exploratory AT2 Figure 5 layout."""
    plot = at2.copy()
    plot["Province"] = plot["province_name_en"]
    plot["Year"] = pd.to_datetime(plot["month_start"], errors="coerce").dt.year.astype("Int64")
    plot["month_label"] = pd.to_datetime(
        plot["month_start"], errors="coerce"
    ).dt.strftime("%b")
    curve = adjusted_at2_prediction_curve(plot)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))

    sns.scatterplot(
        data=plot, x="pm25_monthly_median_ugm3",
        y="respiratory_records_per_active_facility",
        hue="Province", style="Year", s=65, alpha=0.85, ax=axes[0],
    )
    if plot["respiratory_records_per_active_facility"].gt(0).all():
        axes[0].set_yscale("log")
    for row in plot.itertuples(index=False):
        axes[0].annotate(
            row.month_label,
            (row.pm25_monthly_median_ugm3, row.respiratory_records_per_active_facility),
            xytext=(3, 2), textcoords="offset points", fontsize=6, alpha=0.75,
        )
    axes[0].set(
        xlabel="Monthly median observed PM2.5 (µg/m³)",
        ylabel="Respiratory diagnosis records per active facility (log scale)",
        title="A. Observed province-months",
    )
    axes[0].legend(
        title=None, bbox_to_anchor=(1.01, 1), loc="upper left",
        fontsize=7, title_fontsize=8,
    )

    if not curve.empty:
        axes[1].fill_between(
            curve["pm25_monthly_median_ugm3"].to_numpy(float),
            curve["ci_lower"].to_numpy(float),
            curve["ci_upper"].to_numpy(float),
            color="#77AADD", alpha=0.28, label="95% CI",
        )
        axes[1].plot(
            curve["pm25_monthly_median_ugm3"], curve["adjusted_mean"],
            color="#3377CC", linewidth=2.3, label="Adjusted mean",
        )
        axes[1].legend(frameon=False)
    else:
        axes[1].text(
            0.5, 0.5, "Adjusted curve unavailable\n(insufficient variation or rows)",
            ha="center", va="center", transform=axes[1].transAxes,
        )
    axes[1].set(
        xlabel="Monthly median observed PM2.5 (µg/m³)",
        ylabel="Adjusted respiratory records per active facility",
        title="B. Adjusted OLS-HC3 association (95% CI)",
    )
    fig.suptitle(
        "PM2.5 and respiratory diagnosis-record reporting in the QC-passing AT2 sample",
        fontweight="bold", y=1.02,
    )
    fig.text(
        0.5, -0.01,
        "Exploratory association only; diagnosis records are not unique patients or admissions.",
        ha="center", fontsize=9,
    )
    fig.tight_layout()
    save_figure(fig, "fig06_at2_exploratory_association.png")


# =============================================================================
# 6. MANIFEST, SUMMARY, AND ENTRY POINT
# =============================================================================

def write_manifest(
    result_files: dict[str, int | None],
    pm25_respi_rows: int,
) -> None:
    rows = []
    for filename, row_count in result_files.items():
        rows.append({
            "output_group": "RESULT",
            "report_role": "MACHINE_READABLE_MODEL_RESULT",
            "filename": filename,
            "relative_path": (
                f"outputs/modeling/{filename}"
                if filename == "modeling_summary.json"
                else f"outputs/modeling/results/{filename}"
            ),
            "row_count": row_count,
        })
    rows.append({
        "output_group": "MODEL", "report_role": "SELECTED_FITTED_PIPELINE",
        "filename": "selected_random_forest.joblib",
        "relative_path": "outputs/models/selected_random_forest.joblib",
        "row_count": np.nan,
    })
    rows.append({
        "output_group": "TABLE", "report_role": "MAIN_TABLE_3",
        "filename": "pm25_respi.csv",
        "relative_path": "outputs/analysis/tables/pm25_respi.csv",
        "row_count": pm25_respi_rows,
    })
    for filename, role in FIGURES.items():
        rows.append({
            "output_group": "FIGURE", "report_role": role,
            "filename": filename,
            "relative_path": f"outputs/analysis/figures/{filename}",
            "row_count": np.nan,
        })
    save_csv(pd.DataFrame(rows), MODELING_OUTPUT_DIR / "modeling_manifest.csv")


def main() -> None:
    ensure_directories()
    remove_legacy_figures()
    print("Loading and validating AT1 and AT2 model-ready data")
    data = load_model_data()
    at2_data = load_at2_model_data()
    validate_model_data(data)
    validate_at2_model_data(at2_data)
    split = data["data_split"].astype("string").str.upper()
    train = data.loc[split.eq("TRAIN")].copy()
    validation = data.loc[split.eq("VALIDATION")].copy()
    test = data.loc[split.eq("TEST")].copy()

    print("Selecting Random Forest configurations with VALIDATION 2024 only")
    tuning, best_parameters, _validation_pipelines = tune_random_forests(train, validation)
    validation_results = validation_summary(validation, tuning, best_parameters)
    selected_model = select_model_family(validation_results)

    print(f"Frozen selection: {selected_model}; refitting on TRAIN + VALIDATION")
    development = pd.concat([train, validation], ignore_index=True)
    test_results, predictions, fitted_models = refit_and_test(
        development, test, best_parameters, selected_model
    )
    warnings_table = build_warning_table(predictions)
    error_table = build_error_table(predictions)
    cv_results = grouped_time_series_diagnostic(
        development, selected_model, best_parameters[selected_model]
    )
    importance = build_feature_importance(fitted_models[selected_model], selected_model)

    result_frames = {
        "validation_tuning_results.csv": tuning,
        "validation_model_summary.csv": validation_results,
        "test_results.csv": test_results,
        "test_predictions.csv": predictions,
        "warning_metrics.csv": warnings_table,
        "cross_validation_results.csv": cv_results,
        "province_season_errors.csv": error_table,
        "feature_importance.csv": importance,
    }
    for filename, frame in result_frames.items():
        save_csv(frame, RESULT_DIR / filename)

    joblib.dump(fitted_models[selected_model], MODEL_DIR / "selected_random_forest.joblib")

    print("Fitting exploratory AT2 regression")
    pm25_respi = build_pm25_respi(at2_data)
    save_csv(pm25_respi, ANALYSIS_TABLE_DIR / "pm25_respi.csv")

    setup_plot_theme()
    print("Creating report Figures 4-6 and Supplementary Figures S7-S8")
    figure_model_comparison(validation_results, test_results)
    figure_warning_error_diagnostics(predictions, selected_model, error_table)
    figure_respiratory_context(at2_data)
    figure_test_predictions(predictions, selected_model)
    figure_error_heatmaps(error_table)

    selected_validation = validation_results.loc[
        validation_results["model"].eq(selected_model)
    ].iloc[0]
    selected_test = test_results.loc[test_results["model"].eq(selected_model)].iloc[0]
    no_hotspot_test = test_results.loc[
        test_results["model"].eq("RF_NO_HOTSPOTS"), "mae"
    ].iloc[0]
    with_hotspot_test = test_results.loc[
        test_results["model"].eq("RF_WITH_HOTSPOTS"), "mae"
    ].iloc[0]
    hotspot_improvement_pct = (
        100 * (no_hotspot_test - with_hotspot_test) / no_hotspot_test
        if no_hotspot_test > 0 else np.nan
    )
    summary = {
        "project_directory": str(PROJECT_DIR),
        "at1_input_path": str(AT1_INPUT_PATH),
        "at2_input_path": str(AT2_INPUT_PATH),
        "selection_rule": "Lowest 2024 validation MAE, then RMSE; TEST never used for selection",
        "selected_model": selected_model,
        "selected_parameters": best_parameters[selected_model],
        "train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test),
        "selected_validation_mae": float(selected_validation["mae"]),
        "selected_test_mae": float(selected_test["mae"]),
        "selected_test_rmse": float(selected_test["rmse"]),
        "selected_test_r_squared": float(selected_test["r_squared"]),
        "hotspot_test_mae_relative_improvement_pct": (
            float(hotspot_improvement_pct) if np.isfinite(hotspot_improvement_pct) else None
        ),
        "warning_threshold_ugm3": WARNING_THRESHOLD,
        "test_evaluated_after_selection": True,
        "at2_model_ready_rows": len(at2_data),
        "at2_result_rows": len(pm25_respi),
        "at2_methods": [
            "Spearman correlation",
            "Univariable OLS with HC3 robust covariance",
            "OLS-HC3 adjusted for province and cyclic calendar month",
        ],
        "at2_interpretation": "Exploratory association; not causal",
        "figures": list(FIGURES),
    }
    (MODELING_OUTPUT_DIR / "modeling_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    result_files = {filename: len(frame) for filename, frame in result_frames.items()}
    result_files["modeling_summary.json"] = None
    write_manifest(result_files, len(pm25_respi))

    print("Modeling completed successfully.")
    print(f"Selected model: {selected_model}")
    print(f"Validation MAE: {selected_validation['mae']:.3f} µg/m³")
    print(f"Locked-test MAE: {selected_test['mae']:.3f} µg/m³")
    print(f"AT2 model-ready rows: {len(at2_data):,}")
    print(f"AT2 result table: {ANALYSIS_TABLE_DIR / 'pm25_respi.csv'}")
    print(f"Results: {RESULT_DIR}")
    print(f"Figures 4-6 and S7-S8: {FIGURE_DIR}")
    print(f"Selected pipeline: {MODEL_DIR / 'selected_random_forest.joblib'}")


if __name__ == "__main__":
    main()
