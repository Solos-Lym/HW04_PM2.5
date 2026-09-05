"""
Descriptive analytic scripts

REQUIRED RUN SCRIPTS:
    python src/fetch_data.py
    python src/prepare_tables.py
    python src/prepare_data.py


This script reads only ``data/processed/analysis_ready`` (plus the optional
source-table coordinates used for the sensor-location figure). It never edits
raw, source-aligned, or analysis-ready inputs.

Outputs
-------
1. Model ready tables
2. Descriptive analysis
3. Data visualization

"""

# =============================================================================
# 0. Package settings
# =============================================================================

from __future__ import annotations

from pathlib import Path
import json
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


# =============================================================================
# 1. PROJECT SETTINGS AND PRE-SPECIFIED CONTRACT
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_project_dir() -> Path:
    """Resolve the project root independently of directory name and cwd."""
    explicit = os.environ.get("PROJECT_ROOT", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    if SCRIPT_DIR.name.casefold() == "src":
        return SCRIPT_DIR.parent
    return SCRIPT_DIR


PROJECT_DIR = resolve_project_dir()
ANALYSIS_READY_DIR = PROJECT_DIR / "data" / "processed" / "analysis_ready"
SOURCE_TABLE_DIR = PROJECT_DIR / "data" / "processed" / "source_tables"
MODEL_READY_DIR = PROJECT_DIR / "data" / "processed" / "model_ready"
ANALYSIS_OUTPUT_DIR = PROJECT_DIR / "outputs" / "analysis"
TABLE_DIR = ANALYSIS_OUTPUT_DIR / "tables"
FIGURE_DIR = ANALYSIS_OUTPUT_DIR / "figures"

CSV_ENCODING = "utf-8-sig"
TARGET = "observed_pm25_median_ugm3"

PM_LAG_FEATURES = [
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

CALENDAR_FEATURES = [
    "target_day_of_year_sin",
    "target_day_of_year_cos",
]

# Province is categorical in modeling and is deliberately excluded from the
# numeric correlation screen. Feature inclusion never depends on p-values.
AT1_NUMERIC_FEATURES = (
    PM_LAG_FEATURES + WEATHER_FEATURES + HOTSPOT_FEATURES + CALENDAR_FEATURES
)
AT1_MODEL_FEATURES = ["province_key"] + AT1_NUMERIC_FEATURES

REPORT_TABLES = {
    "province_pm25.csv": "MAIN_TABLE_1",
    "month_pm25.csv": "SUPPLEMENT_TABLE_S1",
    "exceed_pm25.csv": "SUPPLEMENT_TABLE_S2",
    "hotspot_summary.csv": "SUPPLEMENT_TABLE_S3",
    "correlation.csv": "MAIN_TABLE_2",
    "correlation_comparison.csv": "SUPPLEMENT_TABLE_S4",
    "province_respi.csv": "SUPPLEMENT_TABLE_S5",
    "hospital_capacity.csv": "SUPPLEMENT_TABLE_S6",
}

FIGURES = {
    "fig01_pm25_coverage_and_patterns.png": "MAIN_FIGURE_1",
    "fig02_pm25_exceedance.png": "MAIN_FIGURE_2",
    "fig03_environmental_factors.png": "MAIN_FIGURE_3",
    "figS01_pm25_descriptive_overview.png": "SUPPLEMENT_FIGURE_S1",
    "figS02_data_coverage.png": "SUPPLEMENT_FIGURE_S2",
    "figS03_pm25_distribution_by_province.png": "SUPPLEMENT_FIGURE_S3",
    "figS04_observed_modeled_benchmark.png": "SUPPLEMENT_FIGURE_S4",
    "figS05_hospital_capacity.png": "SUPPLEMENT_FIGURE_S5",
    "figS06_model_eligibility_by_split.png": "SUPPLEMENT_FIGURE_S6",
}

LEGACY_FIGURES = [
    "fig01_pm25_descriptive_overview.png",
    "figS01_data_coverage.png",
    "figS02_pm25_temporal_trend.png",
    "figS03_hotspot_pm25_relationship.png",
    "figS04_pm25_distribution_by_province.png",
    "figS05_pm25_exceedance.png",
    "figS06_weather_pm25_relationship.png",
    "figS07_predictor_correlation.png",
    "figS08_observed_modeled_benchmark.png",
    "figS09_hospital_capacity.png",
    "figS10_sensor_location_map.png",
    "figS11_model_eligibility_by_split.png",
    "figS12_pm25_observation_coverage.png",
    "fig01_data_coverage.png",
    "fig02_pm25_yearly_comparison.png",
    "fig03_pm25_monthly_seasonality.png",
    "fig04_pm25_temporal_trend.png",
    "fig05_hotspot_pm25_relationship.png",
    "fig06_respiratory_pm25_context.png",
    "figS01_pm25_distribution_by_province.png",
    "figS02_pm25_exceedance.png",
    "figS03_weather_pm25_relationship.png",
    "figS04_correlation_heatmap.png",
    "figS05_observed_modeled_comparison.png",
    "figS06_hospital_capacity.png",
    "figS07_sensor_location_map.png",
    "figS08_model_eligibility_by_split.png",
]


# =============================================================================
# 2. INPUT, VALIDATION, AND OUTPUT HELPERS
# =============================================================================

def ensure_directories() -> None:
    for directory in [MODEL_READY_DIR, TABLE_DIR, FIGURE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def remove_legacy_figures() -> None:
    """Remove only obsolete files written by the preceding analysis.py version."""
    for filename in LEGACY_FIGURES:
        path = FIGURE_DIR / filename
        if path.is_file():
            path.unlink()


def read_csv(path: Path, date_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run python src/prepare_data.py before analysis.py."
        )
    frame = pd.read_csv(path, encoding=CSV_ENCODING, low_memory=False)
    frame.columns = [str(column).replace("\ufeff", "").strip() for column in frame]
    if "province_key" in frame:
        frame["province_key"] = (
            frame["province_key"].astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(2)
        )
    for column in date_columns:
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False).astype(bool)
    return (
        series.astype("string").str.strip().str.casefold().map(
            {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}
        ).fillna(False).astype(bool)
    )


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    output = frame.copy()
    for column in output.select_dtypes(include=["datetime", "datetimetz"]).columns:
        output[column] = output[column].dt.strftime("%Y-%m-%d")
    output.to_csv(path, index=False, encoding=CSV_ENCODING)


def save_figure(fig: plt.Figure, filename: str) -> None:
    path = FIGURE_DIR / filename
    temporary = path.with_suffix(path.suffix + ".part")
    fig.savefig(
        temporary, format="png", dpi=300, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)
    temporary.replace(path)


def assert_unique(frame: pd.DataFrame, keys: list[str], label: str) -> None:
    duplicates = frame.duplicated(keys, keep=False)
    if duplicates.any():
        examples = frame.loc[duplicates, keys].head(5).to_dict("records")
        raise ValueError(f"{label} has duplicate analytical keys {keys}: {examples}")


def load_inputs() -> dict[str, pd.DataFrame]:
    data = {
        "daily": read_csv(
            ANALYSIS_READY_DIR / "data_per_day.csv", ("analysis_date",)
        ),
        "env_full": read_csv(
            ANALYSIS_READY_DIR / "env_data_full.csv", ("analysis_date",)
        ),
        "env_notnull": read_csv(
            ANALYSIS_READY_DIR / "env_data_notnull.csv", ("analysis_date",)
        ),
        "respi": read_csv(
            ANALYSIS_READY_DIR / "respi_disease_data.csv", ("month_start", "month_end")
        ),
    }
    provinces_path = SOURCE_TABLE_DIR / "provinces.csv"
    locations_path = SOURCE_TABLE_DIR / "location.csv"
    data["provinces"] = (
        read_csv(provinces_path) if provinces_path.is_file() else pd.DataFrame()
    )
    data["locations"] = (
        read_csv(locations_path) if locations_path.is_file() else pd.DataFrame()
    )
    return data


def validate_inputs(data: dict[str, pd.DataFrame]) -> None:
    daily = data["daily"]
    env_full = data["env_full"]
    env_notnull = data["env_notnull"]
    respi = data["respi"]

    require_columns(
        daily,
        [
            "province_key", "province_name_en", "analysis_date", TARGET,
            "modeled_pm25_mean_ugm3", "year_ce", "month",
            "thai_24h_standard_ugm3",
        ],
        "data_per_day.csv",
    )
    require_columns(
        env_full,
        ["province_key", "analysis_date", "data_split", "at1_one_day_ahead_eligible"],
        "env_data_full.csv",
    )
    require_columns(
        env_notnull,
        [
            "province_key", "province_name_en", "analysis_date", "data_split",
            TARGET, "persistence_baseline_prediction",
        ] + AT1_MODEL_FEATURES,
        "env_data_notnull.csv",
    )
    require_columns(
        respi,
        [
            "province_key", "province_name_en", "month_start", "year_month",
            "respiratory_record_count_any_position",
            "respiratory_records_per_active_facility",
            "pm25_monthly_median_ugm3", "active_reporting_facility_count",
            "at2_model_eligible",
        ],
        "respi_disease_data.csv",
    )

    assert_unique(daily, ["province_key", "analysis_date"], "data_per_day.csv")
    assert_unique(env_full, ["province_key", "analysis_date"], "env_data_full.csv")
    assert_unique(env_notnull, ["province_key", "analysis_date"], "env_data_notnull.csv")
    assert_unique(respi, ["province_key", "year_month"], "respi_disease_data.csv")

    if daily["analysis_date"].isna().any() or env_notnull["analysis_date"].isna().any():
        raise ValueError("Invalid analysis dates were found")
    if env_notnull[TARGET].isna().any():
        raise ValueError("env_data_notnull.csv contains a missing target")
    numeric_features = env_notnull[AT1_NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    if numeric_features.isna().any().any():
        failed = numeric_features.columns[numeric_features.isna().any()].tolist()
        raise ValueError(f"Model-ready AT1 features contain missing values: {failed}")


# =============================================================================
# 3. MODEL-READY DATASETS
# =============================================================================

def build_model_ready(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    env = data["env_notnull"].copy()
    at1_columns = [
        "province_key", "province_name_th", "province_name_en",
        "analysis_date", "data_split", TARGET,
        "persistence_baseline_prediction",
    ] + [column for column in AT1_MODEL_FEATURES if column != "province_key"]
    for optional in [
        "modeled_pm25_mean_lag1", "at1_source_complete",
        "at1_one_day_ahead_eligible", "at1_one_day_ahead_qc_status",
    ]:
        if optional in env and optional not in at1_columns:
            at1_columns.append(optional)
    at1 = env[[column for column in at1_columns if column in env]].copy()
    for column in [TARGET, "persistence_baseline_prediction"] + AT1_NUMERIC_FEATURES:
        at1[column] = numeric(at1[column])
    at1 = at1.sort_values(["analysis_date", "province_key"]).reset_index(drop=True)

    respi = data["respi"].copy()
    eligible = to_bool(respi["at2_model_eligible"])
    at2 = respi.loc[eligible].copy()
    at2["month_number"] = pd.to_datetime(at2["month_start"], errors="coerce").dt.month
    at2["month_sin"] = np.sin(2 * np.pi * at2["month_number"] / 12)
    at2["month_cos"] = np.cos(2 * np.pi * at2["month_number"] / 12)
    at2_columns = [
        "province_key", "province_name_th", "province_name_en",
        "month_start", "year_month", "year_ce", "month_number",
        "respiratory_record_count_any_position",
        "respiratory_record_count_principal",
        "respiratory_records_per_active_facility",
        "active_reporting_facility_count",
        "pm25_monthly_mean_ugm3", "pm25_monthly_median_ugm3",
        "pm25_days_observed", "pm25_day_coverage_pct",
        "total_population", "opened_beds_total",
        "opened_beds_per_1000_population", "month_sin", "month_cos",
        "at2_model_eligible", "at2_qc_status",
    ]
    at2 = at2[[column for column in at2_columns if column in at2]].copy()
    required_at2 = [
        "respiratory_records_per_active_facility",
        "pm25_monthly_median_ugm3", "month_sin", "month_cos",
    ]
    for column in required_at2:
        at2[column] = numeric(at2[column])
    at2 = at2.dropna(subset=required_at2).sort_values(
        ["month_start", "province_key"]
    ).reset_index(drop=True)

    save_csv(at1, MODEL_READY_DIR / "at1_model_ready.csv")
    save_csv(at2, MODEL_READY_DIR / "at2_model_ready.csv")
    return at1, at2


# =============================================================================
# 4. REPORT TABLES
# =============================================================================

def build_province_pm25(daily: pd.DataFrame) -> pd.DataFrame:
    observed = daily.loc[daily[TARGET].notna()].copy()
    observed[TARGET] = numeric(observed[TARGET])
    total_days = daily.groupby(
        ["province_key", "province_name_th", "province_name_en"], observed=True
    ).size().rename("calendar_days").reset_index()
    summary = observed.groupby(
        ["province_key", "province_name_th", "province_name_en"], observed=True
    )[TARGET].agg(
        observed_days="count", pm25_mean_ugm3="mean", pm25_sd_ugm3="std",
        pm25_min_ugm3="min", pm25_q1_ugm3=lambda x: x.quantile(0.25),
        pm25_median_ugm3="median", pm25_q3_ugm3=lambda x: x.quantile(0.75),
        pm25_max_ugm3="max",
    ).reset_index()
    result = total_days.merge(
        summary,
        on=["province_key", "province_name_th", "province_name_en"],
        how="left",
    )
    result["observed_day_coverage_pct"] = 100 * result["observed_days"] / result["calendar_days"]
    fixed = observed.assign(exceed=observed[TARGET].gt(37.5)).groupby(
        "province_key", observed=True
    )["exceed"].agg(
        days_above_37_5="sum", pct_days_above_37_5="mean"
    ).reset_index()
    fixed["pct_days_above_37_5"] *= 100
    return result.merge(fixed, on="province_key", how="left").sort_values("province_key")


def build_month_pm25(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.loc[daily[TARGET].notna()].copy()
    frame[TARGET] = numeric(frame[TARGET])
    frame["above_37_5"] = frame[TARGET].gt(37.5)
    result = frame.groupby("month", observed=True).agg(
        province_day_rows=(TARGET, "size"),
        distinct_years=("year_ce", "nunique"),
        pm25_mean_ugm3=(TARGET, "mean"),
        pm25_sd_ugm3=(TARGET, "std"),
        pm25_q1_ugm3=(TARGET, lambda x: x.quantile(0.25)),
        pm25_median_ugm3=(TARGET, "median"),
        pm25_q3_ugm3=(TARGET, lambda x: x.quantile(0.75)),
        pm25_min_ugm3=(TARGET, "min"),
        pm25_max_ugm3=(TARGET, "max"),
        days_above_37_5=("above_37_5", "sum"),
        pct_days_above_37_5=("above_37_5", "mean"),
    ).reset_index()
    result["pct_days_above_37_5"] *= 100
    return result.sort_values("month")


def build_exceed_pm25(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.loc[daily[TARGET].notna()].copy()
    frame[TARGET] = numeric(frame[TARGET])
    frame["thai_24h_standard_ugm3"] = numeric(frame["thai_24h_standard_ugm3"])
    frame["applicable_threshold_available"] = frame["thai_24h_standard_ugm3"].notna()
    frame["above_applicable_threshold"] = (
        frame[TARGET].gt(frame["thai_24h_standard_ugm3"])
        & frame["applicable_threshold_available"]
    )
    frame["above_37_5"] = frame[TARGET].gt(37.5)
    group_keys = ["province_key", "province_name_en", "year_ce"]
    result = frame.groupby(group_keys, observed=True).agg(
        observed_days=(TARGET, "size"),
        applicable_threshold_days=("applicable_threshold_available", "sum"),
        applicable_exceedance_days=("above_applicable_threshold", "sum"),
        fixed_37_5_exceedance_days=("above_37_5", "sum"),
        annual_pm25_mean_ugm3=(TARGET, "mean"),
        annual_pm25_median_ugm3=(TARGET, "median"),
    ).reset_index()
    result["applicable_exceedance_pct"] = np.where(
        result["applicable_threshold_days"].gt(0),
        100 * result["applicable_exceedance_days"] / result["applicable_threshold_days"],
        np.nan,
    )
    result["fixed_37_5_exceedance_pct"] = (
        100 * result["fixed_37_5_exceedance_days"] / result["observed_days"]
    )
    return result.sort_values(group_keys)


def safe_correlations(x: pd.Series, y: pd.Series) -> dict[str, float]:
    pairs = pd.DataFrame({"x": numeric(x), "y": numeric(y)}).dropna()
    if len(pairs) < 3 or pairs["x"].nunique() < 2 or pairs["y"].nunique() < 2:
        return {
            "paired_n": len(pairs), "pearson_r": np.nan, "pearson_p": np.nan,
            "spearman_rho": np.nan, "spearman_p": np.nan,
        }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pearson = stats.pearsonr(pairs["x"], pairs["y"])
        spearman = stats.spearmanr(pairs["x"], pairs["y"])
    return {
        "paired_n": len(pairs), "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def feature_group(feature: str) -> str:
    if feature in PM_LAG_FEATURES:
        return "LAGGED_OBSERVED_PM25"
    if feature in WEATHER_FEATURES:
        return "PRIOR_DAY_WEATHER"
    if feature in HOTSPOT_FEATURES:
        return "LAGGED_THAI_HOTSPOTS"
    if feature in CALENDAR_FEATURES:
        return "CALENDAR"
    return "OTHER"


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    p = numeric(p_values)
    output = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna()
    if valid.empty:
        return output
    ordered = valid.sort_values()
    m = len(ordered)
    ranks = np.arange(1, m + 1)
    adjusted = ordered.to_numpy() * m / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output.loc[ordered.index] = np.clip(adjusted, 0, 1)
    return output


def build_correlation(env_notnull: pd.DataFrame) -> pd.DataFrame:
    train = env_notnull.loc[
        env_notnull["data_split"].astype("string").str.upper().eq("TRAIN")
    ].copy()
    if train.empty:
        raise ValueError("No TRAIN rows are available for the leakage-safe correlation analysis")
    rows = []
    for order, predictor in enumerate(AT1_NUMERIC_FEATURES, start=1):
        values = safe_correlations(train[predictor], train[TARGET])
        rows.append({
            "feature_order": order,
            "feature": predictor,
            "feature_group": feature_group(predictor),
            "data_split": "TRAIN",
            **values,
        })
    result = pd.DataFrame(rows)
    result["pearson_fdr_q"] = benjamini_hochberg(result["pearson_p"])
    result["spearman_fdr_q"] = benjamini_hochberg(result["spearman_p"])
    result["pearson_fdr_significant_0_05"] = result["pearson_fdr_q"].lt(0.05)
    result["spearman_fdr_significant_0_05"] = result["spearman_fdr_q"].lt(0.05)
    result["selection_rule"] = "ALL_PRE_SPECIFIED_NUMERIC_FEATURES; NO_P_VALUE_SELECTION"
    result["primary_correlation"] = "SPEARMAN"
    return result.sort_values("feature_order")


def build_hotspot_summary(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()
    country_columns: dict[str, list[str]] = {}
    for country in ["thailand", "laos", "myanmar", "china"]:
        columns = [
            column for column in frame
            if column.startswith(f"hotspot_count_{country}_") and column.endswith("_lag0")
        ]
        country_columns[country] = columns
        frame[f"hotspot_count_{country}_total_lag0"] = (
            frame[columns].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
            if columns else np.nan
        )
    rows = []
    for keys, group in frame.groupby(
        ["province_key", "province_name_en", "year_ce"], observed=True
    ):
        row = {
            "province_key": keys[0], "province_name_en": keys[1], "year_ce": keys[2],
            "province_days": len(group),
        }
        for country in country_columns:
            column = f"hotspot_count_{country}_total_lag0"
            row[f"{country}_hotspot_total"] = numeric(group[column]).sum(min_count=1)
            row[f"{country}_hotspot_daily_mean"] = numeric(group[column]).mean()
            row[f"{country}_hotspot_active_days"] = numeric(group[column]).gt(0).sum()
        correlation = safe_correlations(
            group["hotspot_count_thailand_total_lag0"], group[TARGET]
        )
        row["paired_pm25_days"] = correlation["paired_n"]
        row["thai_hotspot_pm25_spearman_rho"] = correlation["spearman_rho"]
        row["thai_hotspot_pm25_spearman_p"] = correlation["spearman_p"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["province_key", "year_ce"])


def comparison_metrics(group: pd.DataFrame, label: dict[str, object]) -> dict[str, object]:
    pairs = group[[TARGET, "modeled_pm25_mean_ugm3"]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()
    result: dict[str, object] = dict(label)
    result["paired_n"] = len(pairs)
    if pairs.empty:
        result.update({
            "observed_mean_ugm3": np.nan, "modeled_mean_ugm3": np.nan,
            "mean_bias_modeled_minus_observed_ugm3": np.nan,
            "mae_ugm3": np.nan, "rmse_ugm3": np.nan,
            "pearson_r": np.nan, "spearman_rho": np.nan,
        })
        return result
    error = pairs["modeled_pm25_mean_ugm3"] - pairs[TARGET]
    correlations = safe_correlations(pairs["modeled_pm25_mean_ugm3"], pairs[TARGET])
    result.update({
        "observed_mean_ugm3": pairs[TARGET].mean(),
        "modeled_mean_ugm3": pairs["modeled_pm25_mean_ugm3"].mean(),
        "mean_bias_modeled_minus_observed_ugm3": error.mean(),
        "mae_ugm3": error.abs().mean(),
        "rmse_ugm3": float(np.sqrt(np.mean(np.square(error)))),
        "pearson_r": correlations["pearson_r"],
        "spearman_rho": correlations["spearman_rho"],
    })
    return result


def build_correlation_comparison(daily: pd.DataFrame) -> pd.DataFrame:
    rows = [comparison_metrics(daily, {
        "summary_level": "OVERALL", "province_key": "ALL",
        "province_name_en": "All provinces", "year_ce": "ALL",
    })]
    for keys, group in daily.groupby(
        ["province_key", "province_name_en", "year_ce"], observed=True
    ):
        rows.append(comparison_metrics(group, {
            "summary_level": "PROVINCE_YEAR", "province_key": keys[0],
            "province_name_en": keys[1], "year_ce": keys[2],
        }))
    return pd.DataFrame(rows)


def build_province_respi(respi: pd.DataFrame) -> pd.DataFrame:
    frame = respi.copy()
    diagnosis_available = (
        to_bool(frame["diagnosis_data_available"])
        if "diagnosis_data_available" in frame else frame["respiratory_record_count_any_position"].notna()
    )
    frame["diagnosis_available_numeric"] = diagnosis_available.astype(int)
    aggregation: dict[str, tuple[str, object]] = {
        "calendar_months": ("year_month", "nunique"),
        "diagnosis_available_months": ("diagnosis_available_numeric", "sum"),
        "respiratory_records_any_position_total": ("respiratory_record_count_any_position", "sum"),
        "active_reporting_facilities_median": ("active_reporting_facility_count", "median"),
        "pm25_monthly_median_ugm3": ("pm25_monthly_median_ugm3", "median"),
        "months_at_least_75_pct_pm25_days": ("pm25_day_coverage_pct", lambda x: numeric(x).ge(75).sum()),
    }
    if "respiratory_record_count_principal" in frame:
        aggregation["principal_respiratory_records_total"] = (
            "respiratory_record_count_principal", "sum"
        )
    if "total_population" in frame:
        aggregation["total_population_median"] = ("total_population", "median")
    if "opened_beds_total" in frame:
        aggregation["opened_beds_total_median"] = ("opened_beds_total", "median")
    result = frame.groupby(
        ["province_key", "province_name_th", "province_name_en"], observed=True
    ).agg(**aggregation).reset_index()
    result["diagnosis_month_coverage_pct"] = (
        100 * result["diagnosis_available_months"] / result["calendar_months"]
    )
    denominator = frame.groupby("province_key", observed=True)[
        "active_reporting_facility_count"
    ].sum(min_count=1)
    result["respiratory_records_per_reporting_facility_month"] = (
        result.set_index("province_key")["respiratory_records_any_position_total"]
        .div(denominator).to_numpy()
    )
    return result.sort_values("province_key")


def build_hospital_capacity(respi: pd.DataFrame) -> pd.DataFrame:
    aggregation: dict[str, tuple[str, object]] = {
        "months_with_capacity_data": ("opened_beds_total", lambda x: numeric(x).notna().sum()),
        "hospital_count_with_bed_data_median": ("hospital_count_with_bed_data", "median"),
        "opened_beds_total_median": ("opened_beds_total", "median"),
        "opened_beds_per_1000_population_median": ("opened_beds_per_1000_population", "median"),
        "respiratory_records_any_position_total": ("respiratory_record_count_any_position", "sum"),
        "active_reporting_facilities_median": ("active_reporting_facility_count", "median"),
    }
    result = respi.groupby(
        ["province_key", "province_name_th", "province_name_en"], observed=True
    ).agg(**aggregation).reset_index()
    result["respiratory_records_per_opened_bed_context"] = (
        result["respiratory_records_any_position_total"]
        / result["opened_beds_total_median"].replace(0, np.nan)
    )
    result["capacity_interpretation"] = (
        "OPENED_BED_SNAPSHOT_CONTEXT; NOT_HISTORICAL_OCCUPANCY_OR_DEMAND"
    )
    return result.sort_values("province_key")


def build_tables(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    tables = {
        "province_pm25.csv": build_province_pm25(data["daily"]),
        "month_pm25.csv": build_month_pm25(data["daily"]),
        "exceed_pm25.csv": build_exceed_pm25(data["daily"]),
        "hotspot_summary.csv": build_hotspot_summary(data["daily"]),
        "correlation.csv": build_correlation(data["env_notnull"]),
        "correlation_comparison.csv": build_correlation_comparison(data["daily"]),
        "province_respi.csv": build_province_respi(data["respi"]),
        "hospital_capacity.csv": build_hospital_capacity(data["respi"]),
    }
    for filename, frame in tables.items():
        save_csv(frame, TABLE_DIR / filename)
    return tables


# =============================================================================
# 5. FIGURES
# =============================================================================

def setup_plot_theme() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
    })


def figure_data_coverage(data: dict[str, pd.DataFrame]) -> None:
    daily = data["daily"]
    env_full = data["env_full"]
    components = {
        "Observed PM2.5 target": daily[TARGET].notna().mean(),
        "Weather": to_bool(daily["weather_source_available"]).mean()
        if "weather_source_available" in daily else np.nan,
        "Hotspots": to_bool(daily["hotspot_source_available"]).mean()
        if "hotspot_source_available" in daily else np.nan,
        "Modeled PM2.5 benchmark": daily["modeled_pm25_mean_ugm3"].notna().mean(),
        "One-day-ahead eligible": to_bool(env_full["at1_one_day_ahead_eligible"]).mean(),
    }
    plot = pd.DataFrame({"component": components.keys(), "coverage": components.values()}).dropna()
    plot["coverage_pct"] = 100 * plot["coverage"]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    sns.barplot(data=plot, y="component", x="coverage_pct", color="#3977B8", ax=ax)
    for index, value in enumerate(plot["coverage_pct"]):
        ax.text(min(value + 1, 98), index, f"{value:.1f}%", va="center", fontsize=10)
    ax.set(xlim=(0, 105), xlabel="Coverage (%)", ylabel="", title="Analytical data coverage")
    save_figure(fig, "figS02_data_coverage.png")


def figure_pm25_descriptive_overview(daily: pd.DataFrame) -> None:
    """Recreate the agreed three-panel descriptive Figure 1."""
    plot = daily.dropna(subset=[TARGET]).copy()
    plot[TARGET] = numeric(plot[TARGET])
    plot["above_37_5"] = plot[TARGET].gt(37.5)
    province_order = (
        plot[["province_key", "province_name_en"]]
        .drop_duplicates()
        .sort_values("province_key")["province_name_en"]
        .tolist()
    )
    year_pivot = plot.pivot_table(
        index="province_name_en", columns="year_ce", values=TARGET, aggfunc="median"
    ).reindex(province_order)
    month_pivot = plot.pivot_table(
        index="province_name_en", columns="month", values=TARGET, aggfunc="median"
    ).reindex(province_order).reindex(columns=range(1, 13))
    exceed_pivot = (
        plot.pivot_table(
            index="province_name_en", columns="month", values="above_37_5", aggfunc="mean"
        ).reindex(province_order).reindex(columns=range(1, 13)) * 100
    )
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    month_pivot.columns = month_labels
    exceed_pivot.columns = month_labels

    fig, axes = plt.subplots(
        1, 3, figsize=(18, 6.2),
        gridspec_kw={"width_ratios": [0.82, 1.45, 1.45]},
    )
    sns.heatmap(
        year_pivot, cmap="YlOrRd", annot=True, fmt=".1f", linewidths=0.35,
        cbar_kws={"label": "Median PM2.5 (µg/m³)", "shrink": 0.75}, ax=axes[0],
    )
    sns.heatmap(
        month_pivot, cmap="YlOrRd", annot=True, fmt=".1f", linewidths=0.35,
        cbar_kws={"label": "Median PM2.5 (µg/m³)", "shrink": 0.75}, ax=axes[1],
    )
    sns.heatmap(
        exceed_pivot, cmap="OrRd", annot=True, fmt=".0f", linewidths=0.35,
        vmin=0, vmax=100,
        cbar_kws={"label": "Days above 37.5 µg/m³ (%)", "shrink": 0.75}, ax=axes[2],
    )
    panel_settings = [
        ("A. Between-year comparison", "Year"),
        ("B. Calendar-month pattern", "Calendar month"),
        ("C. Reference-threshold exceedance", "Calendar month"),
    ]
    for ax, (title, xlabel) in zip(axes, panel_settings):
        ax.set(title=title, xlabel=xlabel, ylabel="")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)
    axes[0].set_ylabel("Province")
    fig.suptitle(
        "Observed PM2.5 by province, year and calendar month, 2021-2025",
        fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, "figS01_pm25_descriptive_overview.png")


def figure_pm25_trend(daily: pd.DataFrame) -> None:
    plot = daily.dropna(subset=[TARGET]).copy()
    plot["year_month"] = plot["analysis_date"].dt.to_period("M").dt.to_timestamp()
    plot = plot.groupby(
        ["year_month", "province_name_en"], observed=True
    )[TARGET].median().reset_index(name="monthly_median_pm25")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(
        data=plot, x="year_month", y="monthly_median_pm25",
        hue="province_name_en", linewidth=1.6, ax=ax,
    )
    ax.set(
        xlabel="Month", ylabel="Monthly median observed PM2.5 (µg/m³)",
        title="Province-level PM2.5 temporal trend",
    )
    ax.legend(title="Province", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_figure(fig, "figS02_pm25_temporal_trend.png")


def add_thai_hotspot_total(daily: pd.DataFrame) -> pd.DataFrame:
    plot = daily.copy()
    columns = [
        column for column in plot
        if column.startswith("hotspot_count_thailand_") and column.endswith("_lag0")
    ]
    plot["thai_hotspots_total"] = (
        plot[columns].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
        if columns else np.nan
    )
    return plot


def figure_hotspot_relationship(daily: pd.DataFrame) -> None:
    plot = add_thai_hotspot_total(daily).dropna(subset=["thai_hotspots_total", TARGET])
    plot["log1p_thai_hotspots"] = np.log1p(plot["thai_hotspots_total"])
    scatter = plot.sample(min(len(plot), 5000), random_state=20260902)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=scatter, x="log1p_thai_hotspots", y=TARGET,
        alpha=0.18, s=18, color="#4C72B0", edgecolor=None, ax=ax,
    )
    if plot["log1p_thai_hotspots"].nunique() >= 4:
        plot["hotspot_bin"] = pd.qcut(
            plot["log1p_thai_hotspots"], q=min(10, plot["log1p_thai_hotspots"].nunique()),
            duplicates="drop",
        )
        trend = plot.groupby("hotspot_bin", observed=True).agg(
            x=("log1p_thai_hotspots", "median"), y=(TARGET, "median")
        ).reset_index(drop=True)
        ax.plot(trend["x"], trend["y"], color="#C44E52", marker="o", linewidth=2.2,
                label="Binned median")
        ax.legend(frameon=False)
    ax.set(
        xlabel="log(1 + same-day Thailand hotspot count)",
        ylabel="Observed PM2.5 (µg/m³)",
        title="Descriptive relationship between hotspots and PM2.5",
    )
    save_figure(fig, "figS03_hotspot_pm25_relationship.png")


def figure_pm25_distribution(daily: pd.DataFrame) -> None:
    plot = daily.dropna(subset=[TARGET]).copy()
    order = plot.groupby("province_name_en", observed=True)[TARGET].median().sort_values().index
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=plot, y="province_name_en", x=TARGET, order=order,
        color="#72B7B2", showfliers=False, ax=ax,
    )
    ax.set(
        xlabel="Observed PM2.5 (µg/m³)", ylabel="Province",
        title="Distribution of observed PM2.5 by province",
    )
    save_figure(fig, "figS03_pm25_distribution_by_province.png")


def figure_exceedance(exceed: pd.DataFrame) -> None:
    pivot = exceed.pivot(
        index="province_name_en", columns="year_ce", values="fixed_37_5_exceedance_pct"
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt=".1f", cbar_kws={"label": "% days"}, ax=ax)
    ax.set(
        xlabel="Year", ylabel="Province",
        title="Observed days above 37.5 µg/m³",
    )
    save_figure(fig, "figS05_pm25_exceedance.png")


def figure_weather_relationship(env: pd.DataFrame) -> None:
    variables = [
        ("temperature_mean_c_lag1", "Prior-day temperature (°C)"),
        ("relative_humidity_mean_pct_lag1", "Prior-day relative humidity (%)"),
        ("precipitation_mm_lag1", "Prior-day precipitation (mm)"),
        ("wind_speed_mean_kmh_lag1", "Prior-day wind speed (km/h)"),
    ]
    train = env.loc[env["data_split"].astype("string").str.upper().eq("TRAIN")]
    sample = train.sample(min(len(train), 5000), random_state=20260902)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    for ax, (column, label) in zip(axes.flat, variables):
        sns.scatterplot(
            data=sample, x=column, y=TARGET, alpha=0.15, s=14,
            color="#4C72B0", edgecolor=None, ax=ax,
        )
        ax.set(xlabel=label, ylabel="Observed PM2.5 (µg/m³)")
    fig.suptitle("Prior-day weather and next-day observed PM2.5 (TRAIN only)")
    fig.tight_layout()
    save_figure(fig, "figS06_weather_pm25_relationship.png")


def pretty_feature_label(feature: str) -> str:
    labels = {
        "observed_pm25_median_lag1": "Observed PM2.5 (lag 1 day)",
        "observed_pm25_median_lag2": "Observed PM2.5 (lag 2 days)",
        "observed_pm25_median_lag3": "Observed PM2.5 (lag 3 days)",
        "temperature_mean_c_lag1": "Temperature (lag 1 day)",
        "relative_humidity_mean_pct_lag1": "Relative humidity (lag 1 day)",
        "precipitation_mm_lag1": "Precipitation (lag 1 day)",
        "wind_speed_mean_kmh_lag1": "Wind speed (lag 1 day)",
        "wind_direction_sin_lag1": "Wind direction sine (lag 1 day)",
        "wind_direction_cos_lag1": "Wind direction cosine (lag 1 day)",
        "surface_pressure_mean_hpa_lag1": "Surface pressure (lag 1 day)",
        "target_day_of_year_sin": "Target day-of-year sine",
        "target_day_of_year_cos": "Target day-of-year cosine",
    }
    if feature in labels:
        return labels[feature]
    prefix = "hotspot_count_thailand_"
    if feature.startswith(prefix) and "_lag" in feature:
        band_code, lag = feature.removeprefix(prefix).rsplit("_lag", 1)
        band_labels = {
            "000_050km": "0-50 km",
            "050_100km": "50-100 km",
            "100_300km": "100-300 km",
            "300_500km": "300-500 km",
        }
        day_word = "day" if lag == "1" else "days"
        return f"Thailand hotspots {band_labels.get(band_code, band_code)} (lag {lag} {day_word})"
    return feature.replace("_", " ")


def figure_correlation(correlation: pd.DataFrame) -> None:
    plot = correlation.sort_values("spearman_rho").copy()
    height = max(7, 0.34 * len(plot))
    fig, ax = plt.subplots(figsize=(10, height))
    y = np.arange(len(plot))
    for index, row in enumerate(plot.itertuples(index=False)):
        pearson_significant = bool(row.pearson_fdr_significant_0_05)
        spearman_significant = bool(row.spearman_fdr_significant_0_05)
        ax.scatter(
            row.pearson_r, index, marker="o", s=46,
            facecolors="#4C72B0" if pearson_significant else "white",
            edgecolors="#4C72B0", linewidths=1.4,
        )
        ax.scatter(
            row.spearman_rho, index, marker="D", s=40,
            facecolors="#DD8452" if spearman_significant else "white",
            edgecolors="#DD8452", linewidths=1.4,
        )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y, labels=[pretty_feature_label(value) for value in plot["feature"]])
    ax.set(
        xlabel="Correlation coefficient", ylabel="",
        title="Pre-specified predictor correlations with target (TRAIN only)",
        xlim=(-1.05, 1.05),
    )
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#4C72B0",
               markeredgecolor="#4C72B0", label="Pearson r", markersize=7),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#DD8452",
               markeredgecolor="#DD8452", label="Spearman rho", markersize=6),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#555555",
               markeredgecolor="#555555", label="FDR q < 0.05", markersize=7),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor="#555555", label="FDR q >= 0.05", markersize=7),
    ], frameon=False, loc="lower right")
    save_figure(fig, "figS07_predictor_correlation.png")


def figure_observed_modeled(daily: pd.DataFrame) -> None:
    plot = daily[[TARGET, "modeled_pm25_mean_ugm3"]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()
    fig, ax = plt.subplots(figsize=(7, 7))
    hb = ax.hexbin(
        plot[TARGET], plot["modeled_pm25_mean_ugm3"], gridsize=45,
        mincnt=1, cmap="viridis", bins="log",
    )
    lower = float(min(plot.min())) if not plot.empty else 0
    upper = float(max(plot.max())) if not plot.empty else 1
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="#C44E52", linewidth=1.5)
    ax.set(
        xlabel="Observed PM2.5 (µg/m³)", ylabel="Modeled PM2.5 (µg/m³)",
        title="Observed versus modeled PM2.5 benchmark",
    )
    fig.colorbar(hb, ax=ax, label="Province-day count (log scale)")
    save_figure(fig, "figS04_observed_modeled_benchmark.png")


def figure_hospital_capacity(capacity: pd.DataFrame) -> None:
    plot = capacity.sort_values("opened_beds_per_1000_population_median")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.barplot(
        data=plot, y="province_name_en", x="opened_beds_per_1000_population_median",
        color="#8172B3", ax=ax,
    )
    ax.set(
        xlabel="Opened beds per 1,000 population", ylabel="Province",
        title="Hospital capacity context by province",
    )
    ax.text(
        0.01, -0.16, "Snapshot context only; not historical occupancy.",
        transform=ax.transAxes, fontsize=9,
    )
    save_figure(fig, "figS05_hospital_capacity.png")


def figure_sensor_map(provinces: pd.DataFrame, locations: pd.DataFrame) -> None:
    usable_locations = (
        not locations.empty
        and {"longitude", "latitude"}.issubset(locations.columns)
    )
    if usable_locations:
        sensors = locations.copy()
        sensors["longitude"] = numeric(sensors["longitude"])
        sensors["latitude"] = numeric(sensors["latitude"])
        sensors = sensors.dropna(subset=["longitude", "latitude"])
    else:
        sensors = pd.DataFrame(columns=["longitude", "latitude"])
    centroids = provinces.copy()
    if not {"longitude", "latitude"}.issubset(centroids.columns):
        centroids = pd.DataFrame(columns=["longitude", "latitude", "province_name_en"])
    else:
        centroids["longitude"] = numeric(centroids["longitude"])
        centroids["latitude"] = numeric(centroids["latitude"])
        centroids = centroids.dropna(subset=["longitude", "latitude"])
    fig, ax = plt.subplots(figsize=(8, 8))
    if not sensors.empty:
        ax.scatter(
            sensors["longitude"], sensors["latitude"], s=25, alpha=0.45,
            color="#4C72B0", label="OpenAQ location",
        )
    if not centroids.empty:
        ax.scatter(
            centroids["longitude"], centroids["latitude"], s=85,
            marker="*", color="#C44E52", label="Province centroid",
        )
        for row in centroids.itertuples(index=False):
            label = getattr(row, "province_name_en", getattr(row, "province_key", ""))
            ax.annotate(str(label), (row.longitude, row.latitude), xytext=(4, 4),
                        textcoords="offset points", fontsize=8)
    ax.set(
        xlabel="Longitude", ylabel="Latitude",
        title="Observed PM2.5 locations in the study provinces",
    )
    if not sensors.empty or not centroids.empty:
        ax.legend(frameon=False)
    ax.set_aspect("equal", adjustable="datalim")
    save_figure(fig, "figS10_sensor_location_map.png")


def figure_model_eligibility(env_full: pd.DataFrame) -> None:
    frame = env_full.copy()
    frame["eligible"] = to_bool(frame["at1_one_day_ahead_eligible"])
    plot = frame.groupby("data_split", observed=True)["eligible"].agg(
        total_rows="size", eligible_rows="sum", eligible_pct="mean"
    ).reset_index()
    plot["eligible_pct"] *= 100
    order = [split for split in ["TRAIN", "VALIDATION", "TEST"] if split in set(plot["data_split"])]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.barplot(
        data=plot, x="data_split", y="eligible_pct", order=order,
        color="#55A868", ax=ax,
    )
    for patch in ax.patches:
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() + 1,
            f"{patch.get_height():.1f}%",
            ha="center", fontsize=10,
        )
    ax.set(
        xlabel="Chronological split", ylabel="Eligible rows (%)", ylim=(0, 105),
        title="One-day-ahead model eligibility by data split",
    )
    save_figure(fig, "figS06_model_eligibility_by_split.png")


def figure_pm25_observation_coverage(daily: pd.DataFrame) -> None:
    frame = daily.copy()
    frame["target_available"] = frame[TARGET].notna()
    coverage = frame.groupby(
        ["province_name_en", "year_ce"], observed=True
    )["target_available"].mean().mul(100).reset_index(name="coverage_pct")
    pivot = coverage.pivot(
        index="province_name_en", columns="year_ce", values="coverage_pct"
    )
    fig, ax = plt.subplots(figsize=(9, 5.8))
    sns.heatmap(
        pivot, cmap="Blues", annot=True, fmt=".1f", vmin=0, vmax=100,
        linewidths=0.35, cbar_kws={"label": "Target availability (%)"}, ax=ax,
    )
    ax.set(
        xlabel="Year", ylabel="Province",
        title="Observed PM2.5 target availability by province and year",
    )
    save_figure(fig, "figS12_pm25_observation_coverage.png")


def figure_pm25_coverage_and_patterns(data: dict[str, pd.DataFrame]) -> None:
    """Main Figure 1: coverage, monitoring locations, and temporal pattern."""
    daily = data["daily"].copy()
    daily["target_available"] = daily[TARGET].notna()
    province_order = (
        daily[["province_key", "province_name_en"]].drop_duplicates()
        .sort_values("province_key")["province_name_en"].tolist()
    )
    coverage = daily.pivot_table(
        index="province_name_en", columns="year_ce", values="target_available",
        aggfunc="mean",
    ).reindex(province_order).mul(100)

    observed = daily.dropna(subset=[TARGET]).copy()
    observed["year_month"] = observed["analysis_date"].dt.to_period("M").dt.to_timestamp()
    trend = observed.groupby(
        ["year_month", "province_name_en"], observed=True
    )[TARGET].median().reset_index(name="monthly_median_pm25")

    locations = data["locations"].copy()
    if {"longitude", "latitude"}.issubset(locations.columns):
        locations["longitude"] = numeric(locations["longitude"])
        locations["latitude"] = numeric(locations["latitude"])
        locations = locations.dropna(subset=["longitude", "latitude"])
    else:
        locations = pd.DataFrame(columns=["longitude", "latitude"])
    provinces = data["provinces"].copy()
    if {"longitude", "latitude"}.issubset(provinces.columns):
        provinces["longitude"] = numeric(provinces["longitude"])
        provinces["latitude"] = numeric(provinces["latitude"])
        provinces = provinces.dropna(subset=["longitude", "latitude"])
    else:
        provinces = pd.DataFrame(columns=["longitude", "latitude", "province_name_en"])

    fig = plt.figure(figsize=(16, 12))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.34, wspace=0.30)
    ax_coverage = fig.add_subplot(grid[0, 0])
    ax_map = fig.add_subplot(grid[0, 1])
    ax_trend = fig.add_subplot(grid[1, :])

    sns.heatmap(
        coverage, cmap="Blues", annot=True, fmt=".1f", vmin=0, vmax=100,
        linewidths=0.35, cbar_kws={"label": "Available province-days (%)"},
        ax=ax_coverage,
    )
    ax_coverage.set(title="A. PM2.5 observation coverage", xlabel="Year", ylabel="Province")
    ax_coverage.tick_params(axis="y", rotation=0)

    if not locations.empty:
        ax_map.scatter(
            locations["longitude"], locations["latitude"], s=28, alpha=0.5,
            color="#4C72B0", label="Monitoring location",
        )
    if not provinces.empty:
        ax_map.scatter(
            provinces["longitude"], provinces["latitude"], s=85, marker="*",
            color="#C44E52", label="Province centroid",
        )
        for row in provinces.itertuples(index=False):
            label = getattr(row, "province_name_en", getattr(row, "province_key", ""))
            ax_map.annotate(
                str(label), (row.longitude, row.latitude), xytext=(4, 4),
                textcoords="offset points", fontsize=8,
            )
    ax_map.set(
        title="B. PM2.5 monitoring locations", xlabel="Longitude", ylabel="Latitude",
    )
    if not locations.empty or not provinces.empty:
        ax_map.legend(frameon=False)
    ax_map.set_aspect("equal", adjustable="datalim")

    sns.lineplot(
        data=trend, x="year_month", y="monthly_median_pm25",
        hue="province_name_en", linewidth=1.5, ax=ax_trend,
    )
    ax_trend.axhline(37.5, color="#C44E52", linestyle="--", linewidth=1.2,
                     label="37.5 µg/m³")
    ax_trend.set(
        title="C. Monthly median observed PM2.5 by province",
        xlabel="Month", ylabel="Monthly median PM2.5 (µg/m³)",
    )
    ax_trend.legend(title="Province", bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.suptitle(
        "Completeness, locations, and temporal pattern of observed PM2.5",
        fontweight="bold", y=0.99,
    )
    save_figure(fig, "fig01_pm25_coverage_and_patterns.png")


def figure_pm25_exceedance_overview(
    daily: pd.DataFrame, exceed: pd.DataFrame
) -> None:
    """Main Figure 2: fixed-threshold exceedance by month and year."""
    observed = daily.dropna(subset=[TARGET]).copy()
    observed["above_37_5"] = numeric(observed[TARGET]).gt(37.5)
    province_order = (
        observed[["province_key", "province_name_en"]].drop_duplicates()
        .sort_values("province_key")["province_name_en"].tolist()
    )
    month = observed.pivot_table(
        index="province_name_en", columns="month", values="above_37_5", aggfunc="mean"
    ).reindex(province_order).reindex(columns=range(1, 13)).mul(100)
    month.columns = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    year = exceed.pivot(
        index="province_name_en", columns="year_ce", values="fixed_37_5_exceedance_pct"
    ).reindex(province_order)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.4), gridspec_kw={"width_ratios": [1.6, 1]})
    sns.heatmap(
        month, cmap="OrRd", annot=True, fmt=".0f", vmin=0, vmax=100,
        linewidths=0.35, cbar_kws={"label": "Observed days above 37.5 µg/m³ (%)"},
        ax=axes[0],
    )
    sns.heatmap(
        year, cmap="OrRd", annot=True, fmt=".1f", vmin=0, vmax=100,
        linewidths=0.35, cbar_kws={"label": "Observed days above 37.5 µg/m³ (%)"},
        ax=axes[1],
    )
    axes[0].set(title="A. Exceedance by calendar month", xlabel="Month", ylabel="Province")
    axes[1].set(title="B. Exceedance by year", xlabel="Year", ylabel="")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)
    fig.suptitle(
        "Frequency of observed PM2.5 above 37.5 µg/m³",
        fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, "fig02_pm25_exceedance.png")


def figure_environmental_factors(
    daily: pd.DataFrame, env: pd.DataFrame, correlation: pd.DataFrame
) -> None:
    """Main Figure 3: descriptive hotspot/weather associations and correlations."""
    hotspot = add_thai_hotspot_total(daily).dropna(
        subset=["thai_hotspots_total", TARGET]
    ).copy()
    hotspot["log1p_thai_hotspots"] = np.log1p(hotspot["thai_hotspots_total"])
    hotspot_sample = hotspot.sample(min(len(hotspot), 5000), random_state=20260902)
    train = env.loc[env["data_split"].astype("string").str.upper().eq("TRAIN")].copy()
    train_sample = train.sample(min(len(train), 5000), random_state=20260902)
    variables = [
        ("temperature_mean_c_lag1", "Temperature (°C)"),
        ("relative_humidity_mean_pct_lag1", "Relative humidity (%)"),
        ("precipitation_mm_lag1", "Precipitation (mm)"),
        ("wind_speed_mean_kmh_lag1", "Wind speed (km/h)"),
    ]

    fig = plt.figure(figsize=(18, 14))
    grid = fig.add_gridspec(3, 4, hspace=0.42, wspace=0.78)
    ax_hotspot = fig.add_subplot(grid[0, :2])
    weather_axes = [
        fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[2, 0]), fig.add_subplot(grid[2, 1]),
    ]
    ax_corr = fig.add_subplot(grid[:, 2:])

    sns.scatterplot(
        data=hotspot_sample, x="log1p_thai_hotspots", y=TARGET,
        alpha=0.18, s=18, color="#4C72B0", edgecolor=None, ax=ax_hotspot,
    )
    if hotspot["log1p_thai_hotspots"].nunique() >= 4:
        hotspot["hotspot_bin"] = pd.qcut(
            hotspot["log1p_thai_hotspots"],
            q=min(10, hotspot["log1p_thai_hotspots"].nunique()), duplicates="drop",
        )
        trend = hotspot.groupby("hotspot_bin", observed=True).agg(
            x=("log1p_thai_hotspots", "median"), y=(TARGET, "median")
        ).reset_index(drop=True)
        ax_hotspot.plot(
            trend["x"], trend["y"], color="#C44E52", marker="o",
            linewidth=2.2, label="Binned median",
        )
        ax_hotspot.legend(frameon=False)
    ax_hotspot.set(
        title="A. Same-day Thailand hotspots and observed PM2.5",
        xlabel="log(1 + Thailand hotspot count)", ylabel="Observed PM2.5 (µg/m³)",
    )

    for panel, ax, (column, label) in zip("BCDE", weather_axes, variables):
        sns.scatterplot(
            data=train_sample, x=column, y=TARGET, alpha=0.15, s=13,
            color="#4C72B0", edgecolor=None, ax=ax,
        )
        ax.set(
            title=f"{panel}. Prior-day {label.split(' (')[0].lower()}",
            xlabel=label, ylabel="Next-day PM2.5 (µg/m³)",
        )

    corr = correlation.sort_values("spearman_rho").copy()
    for index, row in enumerate(corr.itertuples(index=False)):
        ax_corr.scatter(
            row.pearson_r, index, marker="o", s=42,
            facecolors="#4C72B0" if bool(row.pearson_fdr_significant_0_05) else "white",
            edgecolors="#4C72B0", linewidths=1.2,
        )
        ax_corr.scatter(
            row.spearman_rho, index, marker="D", s=36,
            facecolors="#DD8452" if bool(row.spearman_fdr_significant_0_05) else "white",
            edgecolors="#DD8452", linewidths=1.2,
        )
    ax_corr.axvline(0, color="black", linewidth=0.8)
    ax_corr.set_yticks(
        np.arange(len(corr)), labels=[pretty_feature_label(value) for value in corr["feature"]]
    )
    # Put the long predictor labels outside the composite figure's scatter-plot
    # area so they cannot overlap Panels B-E.
    ax_corr.yaxis.tick_right()
    ax_corr.tick_params(axis="y", labelright=True, labelleft=False, labelsize=8, pad=4)
    ax_corr.set(
        title="F. Predictor correlations with next-day PM2.5 (TRAIN only)",
        xlabel="Correlation coefficient", ylabel="", xlim=(-1.05, 1.05),
    )
    ax_corr.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#4C72B0",
               markeredgecolor="#4C72B0", label="Pearson r", markersize=7),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#DD8452",
               markeredgecolor="#DD8452", label="Spearman rho", markersize=6),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#555555",
               markeredgecolor="#555555", label="FDR q < 0.05", markersize=7),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor="#555555", label="FDR q ≥ 0.05", markersize=7),
    ], frameon=False, loc="lower right")
    fig.suptitle(
        "Descriptive associations of environmental factors with observed PM2.5",
        fontweight="bold", y=0.995,
    )
    save_figure(fig, "fig03_environmental_factors.png")


def build_figures(
    data: dict[str, pd.DataFrame],
    tables: dict[str, pd.DataFrame],
) -> None:
    setup_plot_theme()
    figure_pm25_coverage_and_patterns(data)
    figure_pm25_exceedance_overview(data["daily"], tables["exceed_pm25.csv"])
    figure_environmental_factors(
        data["daily"], data["env_notnull"], tables["correlation.csv"]
    )
    figure_pm25_descriptive_overview(data["daily"])
    figure_data_coverage(data)
    figure_pm25_distribution(data["daily"])
    figure_observed_modeled(data["daily"])
    figure_hospital_capacity(tables["hospital_capacity.csv"])
    figure_model_eligibility(data["env_full"])


# =============================================================================
# 6. MANIFEST AND DISCUSSION DECISIONS
# =============================================================================

def write_decision_notes() -> None:
    text = """# Analysis decisions for Methods and Discussion

1. Predictor inclusion is pre-specified. Correlation significance is not used
   to select features for modeling.
2. The predictor correlation screen uses TRAIN rows only. All 24 numeric
   pre-specified predictors are reported. Spearman is primary, Pearson is
   supplementary, and Benjamini-Hochberg FDR q-values are supplied.
3. Province is a categorical feature; it is not treated as a numeric code.
4. The persistence baseline uses the previous observed province-day PM2.5.
5. The modeled PM2.5 variable is a benchmark and is not a main model predictor.
6. Observed PM2.5 is not imputed. AT1 model-ready rows are complete cases under
   the eligibility rules created by prepare_data.py.
7. analysis.py prepares the QC-passing AT2 model-ready rows but does not fit the
   respiratory association models. modeling.py owns the crude Spearman,
   univariable OLS-HC3, and adjusted OLS-HC3 analyses.
8. AT2 remains exploratory, non-causal, and low-sample. Diagnosis-code records
   are not unique patients or admissions. Opened beds are snapshot capacity
   context, not historical occupancy or demonstrated demand.
9. Chronological TRAIN, VALIDATION, and TEST labels are preserved. Validation
   and test rows are not used in the exploratory correlation screen.
10. analysis.py creates report Figures 1-3 and Supplementary Figures S1-S6.
    modeling.py owns report Figures 4-6, Supplementary Figures S7-S8, and the ninth report table,
    pm25_respi.csv; analysis.py does not fabricate post-model results.
"""
    (ANALYSIS_OUTPUT_DIR / "discussion_decisions.md").write_text(text, encoding="utf-8")


def write_manifest_and_summary(
    at1: pd.DataFrame,
    at2: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
) -> None:
    rows = [
        {
            "output_group": "MODEL_READY", "report_role": "MODELING_INPUT",
            "filename": "at1_model_ready.csv",
            "relative_path": "data/processed/model_ready/at1_model_ready.csv",
            "row_count": len(at1),
        },
        {
            "output_group": "MODEL_READY", "report_role": "MODELING_INPUT",
            "filename": "at2_model_ready.csv",
            "relative_path": "data/processed/model_ready/at2_model_ready.csv",
            "row_count": len(at2),
        },
    ]
    for filename, role in REPORT_TABLES.items():
        rows.append({
            "output_group": "TABLE", "report_role": role,
            "filename": filename,
            "relative_path": f"outputs/analysis/tables/{filename}",
            "row_count": len(tables[filename]),
        })
    for filename, role in FIGURES.items():
        rows.append({
            "output_group": "FIGURE", "report_role": role,
            "filename": filename,
            "relative_path": f"outputs/analysis/figures/{filename}",
            "row_count": np.nan,
        })
    manifest = pd.DataFrame(rows)
    save_csv(manifest, ANALYSIS_OUTPUT_DIR / "analysis_manifest.csv")

    summary = {
        "project_directory": str(PROJECT_DIR),
        "input_directory": str(ANALYSIS_READY_DIR),
        "model_ready_directory": str(MODEL_READY_DIR),
        "table_directory": str(TABLE_DIR),
        "figure_directory": str(FIGURE_DIR),
        "at1_model_ready_rows": len(at1),
        "at2_model_ready_rows": len(at2),
        "report_table_count": len(REPORT_TABLES),
        "main_table_count": sum(role.startswith("MAIN_TABLE") for role in REPORT_TABLES.values()),
        "supplement_table_count": sum(role.startswith("SUPPLEMENT_TABLE") for role in REPORT_TABLES.values()),
        "figure_count": len(FIGURES),
        "main_figure_count": sum(role.startswith("MAIN_FIGURE") for role in FIGURES.values()),
        "supplement_figure_count": sum(role.startswith("SUPPLEMENT_FIGURE") for role in FIGURES.values()),
        "correlation_data_split": "TRAIN",
        "correlation_feature_count": len(AT1_NUMERIC_FEATURES),
        "correlation_feature_selection": "NONE; ALL PRE-SPECIFIED FEATURES REPORTED",
    }
    (ANALYSIS_OUTPUT_DIR / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# =============================================================================
# 7. ENTRY POINT
# =============================================================================

def main() -> None:
    ensure_directories()
    remove_legacy_figures()
    print("Loading and validating analysis-ready inputs")
    data = load_inputs()
    validate_inputs(data)

    print("Creating model-ready datasets")
    at1_model, at2_model = build_model_ready(data)

    print("Creating eight pre-model report tables")
    tables = build_tables(data)

    print("Creating report Figures 1-3 and Supplementary Figures S1-S6")
    build_figures(data, tables)

    write_decision_notes()
    write_manifest_and_summary(at1_model, at2_model, tables)

    print("Analysis completed successfully.")
    print(f"AT1 model-ready rows: {len(at1_model):,}")
    print(f"AT2 model-ready rows: {len(at2_model):,}")
    print(f"Model-ready data: {MODEL_READY_DIR}")
    print(f"Pre-model tables (8): {TABLE_DIR}")
    print(f"Figures 1-3 and S1-S6: {FIGURE_DIR}")
    print(f"Manifest: {ANALYSIS_OUTPUT_DIR / 'analysis_manifest.csv'}")


if __name__ == "__main__":
    main()
