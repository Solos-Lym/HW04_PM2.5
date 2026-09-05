"""
Merge checked source tables into analysis-ready datasets.

REQUIRED RUN SCRIPTS:

    python src/fetch_data.py
    python src/prepare_tables.py

``prepare_tables.py`` owns raw parsing and source-level quality checks. 

This script reads only ``data/processed/source_tables`` and writes analytical data to ``data/processed/analysis_ready``. 
Data dictionaries are written separately to ``outputs/metadata/data_dictionary``. 
The script performs analytical merges, merge-retention checks, coverage assessment, leakage-safe lagging, 
and model-eligibility checks. It never edits raw or source-aligned tables.
"""

# =============================================================================
# 0. Package settings
# =============================================================================

from pathlib import Path
import json
import os
import re

import numpy as np
import pandas as pd

import prepare_tables as pt

# =============================================================================
# 1. PROJECT SETTINGS AND MODEL CONTRACT
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_project_dir():
    """Find the project root without depending on its directory name or cwd."""
    explicit = os.environ.get("PROJECT_ROOT", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    if SCRIPT_DIR.name.casefold() == "src":
        return SCRIPT_DIR.parent
    return SCRIPT_DIR


PROJECT_DIR = resolve_project_dir()
SOURCE_TABLE_DIR = PROJECT_DIR / "data" / "processed" / "source_tables"
ANALYSIS_READY_DIR = PROJECT_DIR / "data" / "processed" / "analysis_ready"
METADATA_DIR = PROJECT_DIR / "outputs" / "metadata"
DATA_DICTIONARY_DIR = METADATA_DIR / "data_dictionary"
QUALITY_DIR = PROJECT_DIR / "outputs" / "quality"

TEXT_ENCODING = "utf-8"
CSV_ENCODING = "utf-8-sig"

START_DATE = pd.Timestamp("2021-04-30")
END_DATE = pd.Timestamp("2025-12-31")
AT2_START = pd.Timestamp("2023-01-01")
AT2_END = pd.Timestamp("2025-12-31")
POLICY_DATE = pd.Timestamp("2025-04-28")

PROVINCES = pt.PROVINCES.copy()
DISTANCE_BANDS = pt.DISTANCE_BANDS

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
    for band, _lower, _upper in DISTANCE_BANDS
    for lag in [1, 2, 3]
]

CALENDAR_FEATURES = [
    "target_day_of_year_sin",
    "target_day_of_year_cos",
]

MAIN_FEATURES = (
    ["province_key"]
    + PM_LAG_FEATURES
    + WEATHER_FEATURES
    + HOTSPOT_FEATURES
    + CALENDAR_FEATURES
)

REQUIRED_SOURCE_TABLES = [
    "provinces.csv",
    "location.csv",
    "observepm25_sensorday.csv",
    "observepm25_provinceday.csv",
    "weather_provinceday.csv",
    "modelpm25_provinceday.csv",
    "hotspot.csv",
    "facility.csv",
    "population.csv",
    "hospital.csv",
    "diagnosis.csv",
]


# =============================================================================
# 2. INPUT, OUTPUT, AND MERGE HELPERS
# =============================================================================

def read_source(filename):
    """Read one required source table and preserve identifier text."""
    path = SOURCE_TABLE_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run python src/prepare_tables.py first."
        )
    frame = pd.read_csv(path, encoding=CSV_ENCODING, low_memory=False)
    frame.columns = [str(column).replace("\ufeff", "").strip() for column in frame]
    if "province_key" in frame:
        frame["province_key"] = pt.clean_key(frame["province_key"])
    if "facility_key" in frame:
        frame["facility_key"] = pt.clean_key(frame["facility_key"], width=5)
    if "analysis_date" in frame:
        frame["analysis_date"] = pd.to_datetime(
            frame["analysis_date"], format="%Y-%m-%d", errors="raise"
        )
    return frame


def verify_source_tables():
    """Require a complete, key-valid prepare_tables.py result."""
    missing = [
        filename for filename in REQUIRED_SOURCE_TABLES
        if not (SOURCE_TABLE_DIR / filename).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing source tables: " + ", ".join(missing)
            + ". Run python src/prepare_tables.py first."
        )
    quality_path = QUALITY_DIR / "table_quality_report.csv"
    if not quality_path.is_file():
        raise FileNotFoundError(
            f"Missing {quality_path}. Run python src/prepare_tables.py first."
        )
    quality = pd.read_csv(quality_path, encoding=CSV_ENCODING)
    failed = quality.loc[quality["status"].astype(str).str.upper().eq("FAIL"), "table"]
    if not failed.empty:
        raise RuntimeError(
            "Source-table QA contains failures: " + ", ".join(failed.astype(str))
        )


def save_csv(frame, path, date_columns=()):
    """Atomically write one derived CSV without modifying its input frame."""
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    for column in date_columns:
        if column in output:
            output[column] = pd.to_datetime(
                output[column], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
    temporary = path.with_suffix(path.suffix + ".part")
    output.to_csv(temporary, index=False, encoding=CSV_ENCODING)
    temporary.replace(path)


def add_qc_reason(current, condition, reason):
    """Append one semicolon-separated reason where a condition is true."""
    prefix = current.where(current.eq(""), current + ";")
    return current.where(~condition, prefix + reason)


def assert_unique(frame, keys, table_name):
    """Stop a many-to-many merge before it changes the analytical grain."""
    if frame[keys].isna().any(axis=1).any():
        raise ValueError(f"{table_name} contains missing merge keys: {keys}")
    if frame.duplicated(keys).any():
        raise ValueError(f"{table_name} is not unique by {keys}")


def merge_with_audit(left, right, keys, source_name, audit_rows, required=False):
    """Perform a validated left merge and record row retention and matching."""
    assert_unique(right, keys, source_name)
    before_rows = len(left)
    relationship = (
        "one_to_one" if not left.duplicated(keys).any() else "many_to_one"
    )
    marker = f"__matched_{len(audit_rows)}"
    right_copy = right.copy()
    right_copy[marker] = True
    merged = left.merge(
        right_copy,
        on=keys,
        how="left",
        validate=relationship,
    )
    matched = merged[marker].fillna(False).astype(bool)
    audit_rows.append({
        "analytical_table": "AT1" if "analysis_date" in keys else "AT2",
        "source_table": source_name,
        "merge_keys": ";".join(keys),
        "validated_relationship": relationship,
        "left_rows_before": before_rows,
        "rows_after": len(merged),
        "matched_rows": int(matched.sum()),
        "unmatched_rows": int((~matched).sum()),
        "matched_pct": 100 * matched.mean() if len(matched) else np.nan,
        "required_for_primary_model": required,
        "row_count_preserved": len(merged) == before_rows,
    })
    if len(merged) != before_rows:
        raise RuntimeError(f"{source_name} changed the left-table row count")
    merged[f"{source_name}_matched"] = matched
    return merged.drop(columns=marker)


def build_daily_grid():
    """Create the fixed AT1 grain: eight provinces by every study date."""
    dates = pd.DataFrame({
        "analysis_date": pd.date_range(START_DATE, END_DATE, freq="D")
    })
    grid = PROVINCES.drop(columns=["latitude", "longitude"]).assign(join_key=1).merge(
        dates.assign(join_key=1), on="join_key"
    ).drop(columns="join_key")
    return grid.sort_values(["province_key", "analysis_date"]).reset_index(drop=True)


# =============================================================================
# 3. AT1 PROVINCE-DAY TABLE
# =============================================================================

def build_at1(merge_audit):
    """Merge the environmental source tables on province and Bangkok date."""
    observed = read_source("observepm25_provinceday.csv")
    weather = read_source("weather_provinceday.csv")
    modeled = read_source("modelpm25_provinceday.csv")
    hotspots = read_source("hotspot.csv")

    keys = ["province_key", "analysis_date"]
    observed_columns = keys + [column for column in observed if column.startswith("observed_")]
    weather_columns = keys + [
        "temperature_mean_c",
        "relative_humidity_mean_pct",
        "precipitation_mm",
        "wind_speed_mean_kmh",
        "wind_direction_dominant_deg",
        "surface_pressure_mean_hpa",
    ]
    modeled_columns = keys + [
        "modeled_pm25_mean_ugm3",
        "modeled_pm25_median_ugm3",
        "modeled_pm25_hour_count",
    ]
    hotspot_columns = keys + [
        column for column in hotspots if column.startswith("hotspot_count_")
    ]

    table = build_daily_grid()
    table = merge_with_audit(
        table, observed[observed_columns], keys,
        "observed_pm25", merge_audit, required=True,
    )
    table = merge_with_audit(
        table, weather[weather_columns], keys,
        "weather", merge_audit, required=True,
    )
    table = merge_with_audit(
        table, modeled[modeled_columns], keys,
        "modeled_pm25", merge_audit, required=False,
    )
    table = merge_with_audit(
        table, hotspots[hotspot_columns], keys,
        "hotspots", merge_audit, required=True,
    )

    table["date_key"] = table["analysis_date"].dt.strftime("%Y%m%d").astype(int)
    table["year_ce"] = table["analysis_date"].dt.year
    table["month"] = table["analysis_date"].dt.month
    table["day_of_year"] = table["analysis_date"].dt.dayofyear
    table["day_of_week"] = table["analysis_date"].dt.dayofweek + 1
    table["is_weekend"] = table["day_of_week"].isin([6, 7])

    table["observed_pm25_available"] = table[
        "observed_pm25_median_ugm3"
    ].notna()
    weather_columns_no_key = weather_columns[2:]
    table["weather_source_available"] = (
        table["weather_matched"] & table[weather_columns_no_key].notna().all(axis=1)
    )
    table["hotspot_source_available"] = table["hotspots_matched"]

    table["thai_24h_standard_ugm3"] = np.where(
        table["analysis_date"] <= pd.Timestamp("2023-05-31"), 50.0, 37.5
    )
    table["pm25_exceeds_applicable_standard"] = (
        table["observed_pm25_median_ugm3"] > table["thai_24h_standard_ugm3"]
    )
    table["pm25_above_37_5"] = table["observed_pm25_median_ugm3"] > 37.5

    lag0_columns = [
        column for column in table
        if column.startswith("hotspot_count_") and column.endswith("_lag0")
    ]
    for column in lag0_columns:
        table[column] = pd.to_numeric(table[column], errors="coerce")

    groups = table.groupby("province_key", sort=False)
    for column in lag0_columns:
        stem = column[:-5]
        for lag in [1, 2, 3]:
            table[f"{stem}_lag{lag}"] = groups[column].shift(lag)

    table["hotspot_zero_confirmed"] = (
        table["hotspot_source_available"]
        & table[lag0_columns].fillna(0).eq(0).all(axis=1)
    )
    table["hotspot_missing_reason"] = np.where(
        table["hotspot_source_available"], pd.NA, "HOTSPOT_SOURCE_DAY_UNAVAILABLE"
    )

    current_day_complete = (
        table["observed_pm25_available"]
        & table["weather_source_available"]
        & table["hotspot_source_available"]
    )
    table["at1_source_complete"] = current_day_complete
    reasons = pd.Series("", index=table.index, dtype="object")
    reasons = add_qc_reason(
        reasons, ~table["observed_pm25_available"], "OBSERVED_PM25_MISSING"
    )
    reasons = add_qc_reason(
        reasons, ~table["weather_source_available"], "WEATHER_MISSING"
    )
    reasons = add_qc_reason(
        reasons, ~table["hotspot_source_available"], "HOTSPOT_SOURCE_MISSING"
    )
    table["at1_qc_status"] = np.where(current_day_complete, "PASS", "REVIEW")
    table["at1_qc_reason"] = reasons.replace("", pd.NA)

    assert_unique(table, keys, "AT_PM25_PROVINCE_DAY")
    return table.sort_values(keys).reset_index(drop=True)


def make_model_frame(at1):
    """Create one-day-ahead predictors known on day t-1 or earlier."""
    data = at1.copy().sort_values([
        "province_key", "analysis_date"
    ]).reset_index(drop=True)
    assert_unique(data, ["province_key", "analysis_date"], "AT1")

    gap = data.groupby("province_key")["analysis_date"].diff().dt.days.dropna()
    if not gap.eq(1).all():
        raise ValueError("AT1 has date gaps; row shifts would not be exact daily lags")

    groups = data.groupby("province_key", sort=False)
    for lag in [1, 2, 3]:
        data[f"observed_pm25_median_lag{lag}"] = groups[
            "observed_pm25_median_ugm3"
        ].shift(lag)

    data["modeled_pm25_mean_lag1"] = groups[
        "modeled_pm25_mean_ugm3"
    ].shift(1)

    weather_sources = [
        "temperature_mean_c",
        "relative_humidity_mean_pct",
        "precipitation_mm",
        "wind_speed_mean_kmh",
        "wind_direction_dominant_deg",
        "surface_pressure_mean_hpa",
    ]
    for column in weather_sources:
        data[f"{column}_lag1"] = groups[column].shift(1)

    radians = np.deg2rad(data["wind_direction_dominant_deg_lag1"])
    data["wind_direction_sin_lag1"] = np.sin(radians)
    data["wind_direction_cos_lag1"] = np.cos(radians)
    day_of_year = data["analysis_date"].dt.dayofyear
    data["target_day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    data["target_day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    missing_features = [column for column in MAIN_FEATURES if column not in data]
    if missing_features:
        raise ValueError(f"AT1 is missing model features: {missing_features}")

    target_missing = data["observed_pm25_median_ugm3"].isna()
    pm_missing = data[PM_LAG_FEATURES].isna().any(axis=1)
    weather_missing = data[WEATHER_FEATURES].isna().any(axis=1)
    hotspot_missing = data[HOTSPOT_FEATURES].isna().any(axis=1)
    data["at1_one_day_ahead_eligible"] = ~(
        target_missing | pm_missing | weather_missing | hotspot_missing
    )

    reasons = pd.Series("", index=data.index, dtype="object")
    reasons = add_qc_reason(reasons, target_missing, "TARGET_MISSING")
    reasons = add_qc_reason(reasons, pm_missing, "OBSERVED_PM25_LAG_MISSING")
    reasons = add_qc_reason(reasons, weather_missing, "WEATHER_LAG_MISSING")
    reasons = add_qc_reason(reasons, hotspot_missing, "HOTSPOT_LAG_MISSING")
    data["at1_one_day_ahead_qc_reason"] = reasons.replace("", pd.NA)
    data["at1_one_day_ahead_qc_status"] = np.where(
        data["at1_one_day_ahead_eligible"], "PASS", "REVIEW"
    )

    year = data["analysis_date"].dt.year
    data["data_split"] = np.select(
        [year.between(2021, 2023), year.eq(2024), year.eq(2025)],
        ["TRAIN", "VALIDATION", "TEST"],
        default="OUTSIDE_SCOPE",
    )
    if data["data_split"].eq("OUTSIDE_SCOPE").any():
        raise ValueError("AT1 contains dates outside the documented split")
    data["persistence_baseline_prediction"] = data[
        "observed_pm25_median_lag1"
    ]

    output_columns = [
        "province_key",
        "province_name_th",
        "province_name_en",
        "analysis_date",
        "data_split",
        "observed_pm25_median_ugm3",
        "persistence_baseline_prediction",
    ] + MAIN_FEATURES + [
        "modeled_pm25_mean_lag1",
        "at1_source_complete",
        "at1_one_day_ahead_eligible",
        "at1_one_day_ahead_qc_status",
        "at1_one_day_ahead_qc_reason",
        "hotspot_zero_confirmed",
        "hotspot_missing_reason",
    ]
    output_columns = list(dict.fromkeys(output_columns))
    return data[output_columns]


CORE_COLUMN_DESCRIPTIONS = {
    "province_key": "Two-digit official province code used as the province key.",
    "province_name_th": "Province name in Thai.",
    "province_name_en": "Province name in English.",
    "analysis_date": "Bangkok calendar date represented by the row.",
    "date_key": "Calendar date encoded as YYYYMMDD.",
    "year_ce": "Calendar year in the Common Era system.",
    "year_month": "Calendar month encoded as YYYY-MM.",
    "month": "Calendar month number from 1 to 12.",
    "month_start": "First calendar date of the month.",
    "month_end": "Last calendar date of the month.",
    "days_in_month": "Number of calendar days in the month.",
    "day_of_year": "Ordinal calendar day within the year.",
    "day_of_week": "ISO-style weekday number: Monday 1 through Sunday 7.",
    "is_weekend": "Whether the date is Saturday or Sunday.",
    "observed_pm25_median_ugm3": "Median observed PM2.5 across eligible OpenAQ sensors in the province-day.",
    "observed_pm25_mean_ugm3": "Mean observed PM2.5 across eligible OpenAQ sensors in the province-day.",
    "observed_pm25_min_ugm3": "Minimum observed PM2.5 across eligible OpenAQ sensors in the province-day.",
    "observed_pm25_max_ugm3": "Maximum observed PM2.5 across eligible OpenAQ sensors in the province-day.",
    "observed_pm25_sd_across_sensors_ugm3": "Standard deviation of observed PM2.5 across eligible sensors in the province-day.",
    "observed_primary_row_count": "Number of eligible OpenAQ sensor-day rows contributing to the province-day.",
    "observed_sensor_count": "Number of distinct eligible OpenAQ sensors contributing to the province-day.",
    "observed_location_count": "Number of distinct OpenAQ locations contributing to the province-day.",
    "temperature_mean_c": "Daily mean 2-metre air temperature.",
    "relative_humidity_mean_pct": "Daily mean 2-metre relative humidity.",
    "precipitation_mm": "Daily accumulated precipitation.",
    "wind_speed_mean_kmh": "Daily mean 10-metre wind speed.",
    "wind_direction_dominant_deg": "Daily dominant 10-metre wind direction.",
    "surface_pressure_mean_hpa": "Daily mean surface pressure.",
    "modeled_pm25_mean_ugm3": "Daily mean modeled PM2.5 benchmark.",
    "modeled_pm25_median_ugm3": "Daily median modeled PM2.5 benchmark.",
    "modeled_pm25_hour_count": "Number of modeled hourly PM2.5 values contributing to the day.",
    "thai_24h_standard_ugm3": "Thai 24-hour PM2.5 standard applicable on the date.",
    "pm25_exceeds_applicable_standard": "Whether observed PM2.5 exceeds the standard applicable on that date.",
    "pm25_above_37_5": "Whether observed PM2.5 is greater than 37.5 micrograms per cubic metre.",
    "observed_pm25_available": "Whether the observed province-day PM2.5 target is available.",
    "weather_source_available": "Whether every required current-day weather value is available.",
    "hotspot_source_available": "Whether the expected hotspot source row is available.",
    "hotspot_zero_confirmed": "Whether an available hotspot source row confirms zero events in every band.",
    "hotspot_missing_reason": "Reason hotspot information is unavailable; blank when available.",
    "at1_source_complete": "Whether observed PM2.5, weather, and hotspot sources are all complete for the day.",
    "at1_qc_status": "Current-day source completeness status: PASS or REVIEW.",
    "at1_qc_reason": "Semicolon-separated reasons for current-day source review.",
    "data_split": "Temporal model split: TRAIN, VALIDATION, or TEST.",
    "persistence_baseline_prediction": "Baseline prediction equal to observed PM2.5 on the previous day.",
    "modeled_pm25_mean_lag1": "Daily mean modeled PM2.5 from one day before the target date.",
    "target_day_of_year_sin": "Sine transformation of target day-of-year for annual seasonality.",
    "target_day_of_year_cos": "Cosine transformation of target day-of-year for annual seasonality.",
    "at1_one_day_ahead_eligible": "Whether the target and every main one-day-ahead predictor are complete.",
    "at1_one_day_ahead_qc_status": "One-day-ahead model eligibility status: PASS or REVIEW.",
    "at1_one_day_ahead_qc_reason": "Semicolon-separated reasons the row is not model eligible.",
    "all_diagnosis_record_count": "Count of valid DDC diagnosis-code records in the province-month.",
    "respiratory_record_count_any_position": "Count of J00-J99 records with diagnosis type 1, 2, or 3.",
    "respiratory_record_count_principal": "Count of principal J00-J99 diagnosis records with diagnosis type 1.",
    "active_reporting_facility_count": "Number of distinct facilities contributing valid diagnosis records in the month.",
    "pm25_monthly_mean_ugm3": "Mean of available observed province-day median PM2.5 values in the month.",
    "pm25_monthly_median_ugm3": "Median of available observed province-day median PM2.5 values in the month.",
    "pm25_days_observed": "Number of days with observed province-day PM2.5 in the month.",
    "pm25_day_coverage_pct": "Percentage of calendar days with observed province-day PM2.5.",
    "male_population": "Registered male population for the province-year.",
    "female_population": "Registered female population for the province-year.",
    "total_population": "Registered total population for the province-year.",
    "hospital_count_with_bed_data": "Number of hospitals contributing opened-bed information.",
    "opened_beds_total": "Sum of opened beds in the undated hospital-capacity snapshot.",
    "opened_beds_per_1000_population": "Opened beds per 1,000 registered population.",
    "respiratory_records_per_active_facility": "Respiratory diagnosis-code records divided by active reporting facilities.",
    "active_reporting_facility_count_prev_month": "Active reporting facility count in the preceding province-month.",
    "active_reporting_facility_change_pct": "Month-to-month percentage change in active reporting facilities.",
    "reporting_change_flag": "Whether active reporting facility count changed by more than 20 percent.",
    "diagnosis_data_available": "Whether the province-month contains at least one diagnosis record.",
    "population_available": "Whether a positive registered population value is available.",
    "diagtype_policy_review_flag": "Whether the month falls in the diagnosis-type policy review period.",
    "at2_model_eligible": "Whether the province-month meets the documented exploratory-model criteria.",
    "at2_qc_status": "Respiratory-capacity analytical quality status: PASS or REVIEW.",
    "at2_qc_reason": "Semicolon-separated reasons for respiratory-capacity review.",
}


def dictionary_data_type(series, column):
    """Return a compact, human-readable type for one analytical column."""
    if column == "province_key":
        return "categorical"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "string"


def dictionary_unit(column):
    """Infer the documented measurement unit from a standardized name."""
    if (
        "ugm3" in column
        or column.startswith("observed_pm25_median_lag")
        or column == "persistence_baseline_prediction"
    ):
        return "micrograms per cubic metre"
    if column.endswith("_c") or "_c_lag" in column:
        return "degrees Celsius"
    if column.endswith("_pct") or "_pct_lag" in column:
        return "percent"
    if column.endswith("_mm") or "_mm_lag" in column:
        return "millimetres"
    if column.endswith("_kmh") or "_kmh_lag" in column:
        return "kilometres per hour"
    if column.endswith("_hpa") or "_hpa_lag" in column:
        return "hectopascals"
    if column.endswith("_deg") or "_deg_lag" in column:
        return "degrees"
    if column == "opened_beds_per_1000_population":
        return "beds per 1,000 population"
    if column.endswith("_count") or "_count_" in column:
        return "count"
    if column.endswith("_date") or column in {"month_start", "month_end"}:
        return "date"
    return "unitless"


def dictionary_description(column):
    """Return a definition, including systematic lag and hotspot definitions."""
    if column in CORE_COLUMN_DESCRIPTIONS:
        return CORE_COLUMN_DESCRIPTIONS[column]

    pm_lag = re.fullmatch(r"observed_pm25_median_lag([123])", column)
    if pm_lag:
        return (
            "Observed province-day median PM2.5 from "
            f"{pm_lag.group(1)} day(s) before the target date."
        )

    weather_lag = re.fullmatch(
        r"(temperature_mean_c|relative_humidity_mean_pct|precipitation_mm|"
        r"wind_speed_mean_kmh|surface_pressure_mean_hpa)_lag1",
        column,
    )
    if weather_lag:
        source = CORE_COLUMN_DESCRIPTIONS[weather_lag.group(1)].rstrip(".")
        return f"{source}, shifted one day before the target date."

    if column == "wind_direction_sin_lag1":
        return "Sine transformation of dominant wind direction one day before the target date."
    if column == "wind_direction_cos_lag1":
        return "Cosine transformation of dominant wind direction one day before the target date."

    hotspot = re.fullmatch(
        r"hotspot_count_(thailand|laos|myanmar|china)_"
        r"(\d{3})_(\d{3})km_lag([0-3])",
        column,
    )
    if hotspot:
        country, lower, upper, lag = hotspot.groups()
        timing = "on the represented date" if lag == "0" else (
            f"{lag} day(s) before the represented target date"
        )
        return (
            f"Filtered VIIRS S-NPP hotspot count in {country.title()} within "
            f"{int(lower)}-{int(upper)} km of the nearest eligible monitoring "
            f"location, {timing}."
        )

    if column.endswith("_matched"):
        source = column.removesuffix("_matched").replace("_", " ")
        return f"Whether the {source} source table matched the analytical row."
    return column.replace("_", " ").capitalize() + "."


def dictionary_source(column):
    """Identify the principal source or derivation for one column."""
    if column.startswith("observed_") or column.startswith("persistence_"):
        return "OpenAQ or lag derived from OpenAQ"
    if column.startswith(("temperature_", "relative_humidity_", "precipitation_", "wind_", "surface_pressure_")):
        return "Open-Meteo historical weather or derived lag"
    if column.startswith("modeled_pm25"):
        return "Open-Meteo air-quality model benchmark or derived lag"
    if column.startswith("hotspot_"):
        return "NASA FIRMS VIIRS S-NPP or derived lag/QC"
    if column.startswith("respiratory_") or column.startswith("all_diagnosis") or column.startswith("active_reporting") or column.startswith("diagnosis_"):
        return "Thailand DDC diagnosis records or derived QC"
    if "population" in column:
        return "Thailand DOPA registered population or derived rate/QC"
    if "bed" in column or column.startswith("hospital_count"):
        return "Hospital Accreditation Thailand capacity snapshot or derived rate"
    if column.startswith(("at1_", "at2_", "pm25_day_", "reporting_", "diagtype_")):
        return "Derived analytical or quality-control field"
    if column in {"province_key", "province_name_th", "province_name_en"}:
        return "Study province reference"
    return "Derived calendar or analytical field"


def dictionary_role(column, dataset_name):
    """Assign the analytical role used in downstream scripts."""
    key_columns = {
        "data_per_day": {"province_key", "analysis_date"},
        "env_data": {"province_key", "analysis_date"},
        "respi_disease_data": {"province_key", "year_month"},
    }
    if dataset_name == "env_data" and column == "province_key":
        return "PRIMARY_KEY_AND_MODEL_FEATURE"
    if column in key_columns[dataset_name]:
        return "PRIMARY_KEY"
    if dataset_name == "env_data" and column == "observed_pm25_median_ugm3":
        return "TARGET"
    if column == "persistence_baseline_prediction":
        return "BASELINE_PREDICTION"
    if dataset_name == "env_data" and column in MAIN_FEATURES:
        return "MAIN_MODEL_FEATURE"
    if column.startswith("respiratory_record_count"):
        return "EXPLORATORY_OUTCOME"
    if column.endswith("_matched") or "qc_" in column or column.endswith("_available") or column.endswith("_eligible") or column.endswith("_flag"):
        return "QUALITY_CONTROL"
    return "DESCRIPTIVE_OR_CONTEXT"


def build_data_dictionary(frame, dataset_name, used_by_files):
    """Create one complete dictionary row for every output-table column."""
    rows = []
    for position, column in enumerate(frame.columns, start=1):
        prediction_time = pd.NA
        if dataset_name == "env_data":
            if column == "observed_pm25_median_ugm3":
                prediction_time = False
            elif column in MAIN_FEATURES or column in {
                "province_name_th", "province_name_en", "analysis_date",
                "data_split", "persistence_baseline_prediction",
            }:
                prediction_time = True
        rows.append({
            "dataset_name": dataset_name,
            "used_by_files": ";".join(used_by_files),
            "column_order": position,
            "column_name": column,
            "data_type": dictionary_data_type(frame[column], column),
            "role": dictionary_role(column, dataset_name),
            "is_main_model_feature": bool(
                dataset_name == "env_data" and column in MAIN_FEATURES
            ),
            "unit": dictionary_unit(column),
            "nullable_in_full_dataset": bool(frame[column].isna().any()),
            "available_at_prediction_time": prediction_time,
            "source_or_derivation": dictionary_source(column),
            "description": dictionary_description(column),
        })
    dictionary = pd.DataFrame(rows)
    if dictionary["column_name"].tolist() != frame.columns.tolist():
        raise RuntimeError(f"Data dictionary does not match {dataset_name}")
    return dictionary


# =============================================================================
# 4. AT2 PROVINCE-MONTH EXPLORATORY TABLE
# =============================================================================

def build_at2(at1, merge_audit):
    """Merge respiratory, PM2.5, population, and capacity at province-month grain."""
    respiratory = read_source("diagnosis.csv")
    population = read_source("population.csv")
    capacity_rows = read_source("hospital.csv")

    capacity = capacity_rows.groupby("province_key", observed=True).agg(
        hospital_count_with_bed_data=("facility_key", "nunique"),
        opened_beds_total=("opened_beds", lambda values: values.sum(min_count=1)),
    ).reset_index()

    daily = at1.loc[at1["analysis_date"].between(AT2_START, AT2_END)].copy()
    daily["year_month"] = daily["analysis_date"].dt.to_period("M").astype("string")
    pm = daily.groupby(["province_key", "year_month"], observed=True)[
        "observed_pm25_median_ugm3"
    ].agg(
        pm25_monthly_mean_ugm3="mean",
        pm25_monthly_median_ugm3="median",
        pm25_days_observed="count",
    ).reset_index()

    months = pd.DataFrame({
        "month_start": pd.date_range(AT2_START, AT2_END, freq="MS")
    })
    months["year_month"] = months["month_start"].dt.to_period("M").astype("string")
    months["month_end"] = months["month_start"] + pd.offsets.MonthEnd(0)
    months["year_ce"] = months["month_start"].dt.year
    months["month"] = months["month_start"].dt.month
    months["days_in_month"] = months["month_start"].dt.days_in_month
    table = PROVINCES.drop(columns=["latitude", "longitude"]).assign(join_key=1).merge(
        months.assign(join_key=1), on="join_key"
    ).drop(columns="join_key")

    table = merge_with_audit(
        table, respiratory, ["province_key", "year_month"],
        "respiratory", merge_audit, required=False,
    )
    table = merge_with_audit(
        table, pm, ["province_key", "year_month"],
        "observed_pm25_monthly", merge_audit, required=False,
    )
    table = merge_with_audit(
        table, population, ["province_key", "year_ce"],
        "population", merge_audit, required=False,
    )
    table = merge_with_audit(
        table, capacity, ["province_key"],
        "hospital_capacity", merge_audit, required=False,
    )

    table["diagnosis_data_available"] = table[
        "all_diagnosis_record_count"
    ].gt(0)
    table["pm25_days_observed"] = table["pm25_days_observed"].fillna(0)
    table["pm25_day_coverage_pct"] = (
        100 * table["pm25_days_observed"] / table["days_in_month"]
    )
    table["population_available"] = table["total_population"].gt(0)
    table["opened_beds_per_1000_population"] = (
        1000 * table["opened_beds_total"]
        / table["total_population"].replace(0, np.nan)
    )
    table["respiratory_records_per_active_facility"] = (
        table["respiratory_record_count_any_position"]
        / table["active_reporting_facility_count"].replace(0, np.nan)
    )
    table["active_reporting_facility_count_prev_month"] = table.groupby(
        "province_key"
    )["active_reporting_facility_count"].shift(1)
    table["active_reporting_facility_change_pct"] = 100 * (
        table["active_reporting_facility_count"]
        - table["active_reporting_facility_count_prev_month"]
    ) / table["active_reporting_facility_count_prev_month"].replace(0, np.nan)
    table["reporting_change_flag"] = (
        table["active_reporting_facility_change_pct"].abs() > 20
    )
    table["diagtype_policy_review_flag"] = table["month_end"] >= POLICY_DATE
    table["at2_model_eligible"] = (
        table["diagnosis_data_available"]
        & table["pm25_day_coverage_pct"].ge(75)
        & table["population_available"]
        & ~table["diagtype_policy_review_flag"]
    )
    table["at2_qc_status"] = np.where(
        table["at2_model_eligible"]
        & ~table["reporting_change_flag"].fillna(False),
        "PASS", "REVIEW",
    )
    reasons = pd.Series("", index=table.index, dtype="object")
    reasons = add_qc_reason(
        reasons, ~table["diagnosis_data_available"], "DIAGNOSIS_DATA_MISSING"
    )
    reasons = add_qc_reason(
        reasons, table["pm25_day_coverage_pct"].lt(75), "PM25_COVERAGE_BELOW_75_PCT"
    )
    reasons = add_qc_reason(
        reasons, ~table["population_available"], "POPULATION_MISSING"
    )
    reasons = add_qc_reason(
        reasons, table["diagtype_policy_review_flag"], "DIAGTYPE_POLICY_REVIEW_PERIOD"
    )
    reasons = add_qc_reason(
        reasons, table["reporting_change_flag"].fillna(False),
        "REPORTING_FACILITY_COUNT_CHANGED_OVER_20_PCT",
    )
    table["at2_qc_reason"] = reasons.replace("", pd.NA)
    assert_unique(table, ["province_key", "year_month"], "AT2")
    return table.sort_values(["province_key", "year_month"]).reset_index(drop=True)


# =============================================================================
# 5. POST-MERGE COVERAGE AND ELIGIBILITY REPORTS
# =============================================================================

def coverage_report(at1, model_frame, at2):
    """Summarize availability after merge without hiding unavailable rows."""
    rows = []

    def add(scope, component, available):
        available = pd.Series(available).fillna(False).astype(bool)
        rows.append({
            "analytical_table": scope,
            "component": component,
            "total_rows": len(available),
            "available_rows": int(available.sum()),
            "unavailable_rows": int((~available).sum()),
            "coverage_pct": 100 * available.mean() if len(available) else np.nan,
        })

    add("AT1", "observed_pm25_target", at1["observed_pm25_available"])
    add("AT1", "weather_current_day", at1["weather_source_available"])
    add("AT1", "hotspot_current_day", at1["hotspot_source_available"])
    add("AT1", "modeled_pm25_benchmark", at1["modeled_pm25_matched"])
    add("AT1", "all_current_day_primary_sources", at1["at1_source_complete"])
    add("AT1_MODEL", "one_day_ahead_model_eligible", model_frame[
        "at1_one_day_ahead_eligible"
    ])
    add("AT2", "diagnosis", at2["diagnosis_data_available"])
    add("AT2", "pm25_at_least_75_pct_days", at2["pm25_day_coverage_pct"].ge(75))
    add("AT2", "population", at2["population_available"])
    add("AT2", "capacity", at2["opened_beds_total"].notna())
    add("AT2", "exploratory_model_eligible", at2["at2_model_eligible"])
    return pd.DataFrame(rows)


def model_eligibility_reports(model_frame):
    """Report eligible rows by temporal split and every exclusion reason."""
    split = model_frame.groupby("data_split", observed=True).agg(
        total_rows=("analysis_date", "size"),
        eligible_rows=("at1_one_day_ahead_eligible", "sum"),
        start_date=("analysis_date", "min"),
        end_date=("analysis_date", "max"),
    ).reset_index()
    split["ineligible_rows"] = split["total_rows"] - split["eligible_rows"]
    split["eligible_pct"] = 100 * split["eligible_rows"] / split["total_rows"]

    exploded = (
        model_frame.loc[
            ~model_frame["at1_one_day_ahead_eligible"],
            ["data_split", "at1_one_day_ahead_qc_reason"],
        ]
        .assign(reason=lambda data: data["at1_one_day_ahead_qc_reason"].str.split(";"))
        .explode("reason")
    )
    reason = exploded.groupby(
        ["data_split", "reason"], observed=True
    ).size().rename("row_count").reset_index()
    return split, reason


def analytical_decisions_table():
    """Document decisions made specifically during analytical construction."""
    rows = [
        ("AT1 grain", "One row per study province and Bangkok calendar date", "Creates a complete grid before left merges so missing source data remain visible"),
        ("AT1 target", "Observed province-day median PM2.5 on date t", "The primary question predicts the next day's observed PM2.5"),
        ("PM predictors", "Observed PM2.5 from t-1, t-2, and t-3", "Known before the target day and captures persistence"),
        ("Weather predictors", "Weather from t-1", "Avoid same-day information that would not be known for a strict one-day-ahead prediction"),
        ("Hotspot predictors", "Thailand hotspot counts from t-1, t-2, and t-3", "Avoid target-day leakage; cross-border counts remain available for descriptive analysis"),
        ("Calendar predictors", "Sine and cosine of target day-of-year", "Target calendar date is known at prediction time and represents seasonality"),
        ("Data split", "2021-2023 train, 2024 validation, 2025 test", "Preserve temporal order and avoid an ordinary random split"),
        ("Baseline", "Previous-day observed PM2.5", "A persistence baseline is appropriate for daily air-pollution forecasting"),
        ("Model-ready subset", "Keep only rows with target and every main predictor", "No imputation is performed; all exclusions are reported"),
        ("AT2 grain", "One row per study province and month, 2023-2025", "Align monthly respiratory counts with monthly PM2.5 and annual/contextual covariates"),
        ("AT2 interpretation", "Exploratory association and capacity context only", "DDC reporting and undated bed context do not establish historical admissions or causal effects"),
    ]
    return pd.DataFrame(rows, columns=["decision", "implementation", "reason"])


# =============================================================================
# 6. MAIN WORKFLOW
# =============================================================================

def main():
    """Build analytical datasets, dictionaries, and post-merge QA reports."""
    verify_source_tables()
    ANALYSIS_READY_DIR.mkdir(parents=True, exist_ok=True)
    QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DICTIONARY_DIR.mkdir(parents=True, exist_ok=True)

    merge_audit = []
    print("Building AT1 from checked source tables")
    at1 = build_at1(merge_audit)
    model_frame = make_model_frame(at1)
    ready = model_frame.loc[
        model_frame["at1_one_day_ahead_eligible"]
    ].copy()

    required_ready = MAIN_FEATURES + [
        "observed_pm25_median_ugm3",
        "persistence_baseline_prediction",
    ]
    if ready[required_ready].isna().any().any():
        raise ValueError("AT1 model-ready rows still contain required missing values")

    print("Building exploratory AT2 from checked source tables")
    at2 = build_at2(at1, merge_audit)

    save_csv(
        at1,
        ANALYSIS_READY_DIR / "data_per_day.csv",
        date_columns=["analysis_date"],
    )
    save_csv(
        model_frame,
        ANALYSIS_READY_DIR / "env_data_full.csv",
        date_columns=["analysis_date"],
    )
    save_csv(
        ready,
        ANALYSIS_READY_DIR / "env_data_notnull.csv",
        date_columns=["analysis_date"],
    )
    save_csv(
        at2,
        ANALYSIS_READY_DIR / "respi_disease_data.csv",
        date_columns=["month_start", "month_end"],
    )
    save_csv(
        build_data_dictionary(
            at1, "data_per_day", ["data_per_day.csv"]
        ),
        DATA_DICTIONARY_DIR / "data_per_day_dictionary.csv",
    )
    save_csv(
        build_data_dictionary(
            model_frame,
            "env_data",
            ["env_data_full.csv", "env_data_notnull.csv"],
        ),
        DATA_DICTIONARY_DIR / "env_data_dictionary.csv",
    )
    save_csv(
        build_data_dictionary(
            at2, "respi_disease_data", ["respi_disease_data.csv"]
        ),
        DATA_DICTIONARY_DIR / "respi_disease_data_dictionary.csv",
    )

    merge_report = pd.DataFrame(merge_audit)
    if not merge_report["row_count_preserved"].all():
        raise RuntimeError("At least one analytical merge changed the intended grain")
    save_csv(merge_report, QUALITY_DIR / "merge_retention_report.csv")
    save_csv(
        coverage_report(at1, model_frame, at2),
        QUALITY_DIR / "coverage_report.csv",
    )
    split_report, exclusion_report = model_eligibility_reports(model_frame)
    save_csv(
        split_report,
        QUALITY_DIR / "model_eligibility_report.csv",
        date_columns=["start_date", "end_date"],
    )
    save_csv(
        exclusion_report,
        QUALITY_DIR / "model_exclusion_reason_report.csv",
    )
    save_csv(
        analytical_decisions_table(),
        QUALITY_DIR / "analytical_decisions.csv",
    )

    summary = {
        "at1_province_day_rows": len(at1),
        "at1_full_rows": len(model_frame),
        "at1_model_ready_rows": len(ready),
        "at2_rows": len(at2),
        "target": "observed_pm25_median_ugm3",
        "same_day_environmental_predictors_used": False,
        "ordinary_random_split_used": False,
        "main_feature_count": len(MAIN_FEATURES),
        "analysis_ready_directory": str(ANALYSIS_READY_DIR),
        "data_dictionary_directory": str(DATA_DICTIONARY_DIR),
        "analysis_ready_files": [
            "data_per_day.csv",
            "env_data_full.csv",
            "env_data_notnull.csv",
            "respi_disease_data.csv",
        ],
    }
    with open(
        METADATA_DIR / "prepare_data_summary.json", "w", encoding=TEXT_ENCODING
    ) as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, sort_keys=True)

    print("Analysis-ready data preparation completed successfully.")
    print(f"Environmental full rows: {len(model_frame):,}")
    print(f"Environmental non-null rows: {len(ready):,}")
    print(split_report.to_string(index=False))
    print(f"Respiratory disease rows: {len(at2):,}")
    print(f"Analysis-ready data: {ANALYSIS_READY_DIR}")
    print(f"Data dictionaries: {DATA_DICTIONARY_DIR}")
    print(f"Merge and coverage reports: {QUALITY_DIR}")


if __name__ == "__main__":
    main()
