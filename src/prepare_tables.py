"""
Tables making from raw data 
Run this script after ``fetch_data.py``
"""
# =============================================================================
# 0. Package settings
# =============================================================================

from pathlib import Path
import importlib.util
import json
import os
import re
import tempfile

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

# =============================================================================
# 1. PROJECT SETTINGS
# =============================================================================

# Setting directory 
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
RAW_DIR = PROJECT_DIR / "data" / "raw"
SOURCE_TABLE_DIR = PROJECT_DIR / "data" / "processed" / "source_tables"
METADATA_DIR = PROJECT_DIR / "outputs" / "metadata"
QUALITY_DIR = PROJECT_DIR / "outputs" / "quality"
FETCH_STATUS_FILE = METADATA_DIR / "fetch_source_status.csv"

TEXT_ENCODING = "utf-8"
CSV_ENCODING = "utf-8-sig"
CHUNK_SIZE = 100_000
EARTH_RADIUS_KM = 6371.0088

AT1_START = pd.Timestamp("2021-04-30")
AT1_END = pd.Timestamp("2025-12-31")
HOTSPOT_START = AT1_START - pd.Timedelta(days=3)
AT2_START = pd.Timestamp("2023-01-01")
AT2_END = pd.Timestamp("2025-12-31")

# Setting Key columns 
PROVINCES = pd.DataFrame([
    ["50", "เชียงใหม่", "Chiang Mai", 18.7883, 98.9853],
    ["51", "ลำพูน", "Lamphun", 18.5745, 99.0087],
    ["52", "ลำปาง", "Lampang", 18.2888, 99.4909],
    ["54", "แพร่", "Phrae", 18.1446, 100.1403],
    ["55", "น่าน", "Nan", 18.7756, 100.7730],
    ["56", "พะเยา", "Phayao", 19.1665, 99.9019],
    ["57", "เชียงราย", "Chiang Rai", 19.9105, 99.8406],
    ["58", "แม่ฮ่องสอน", "Mae Hong Son", 19.3013, 97.9685],
], columns=[
    "province_key",
    "province_name_th",
    "province_name_en",
    "latitude",
    "longitude",
])

COUNTRIES = {
    "THA": "thailand",
    "LAO": "laos",
    "MMR": "myanmar",
    "CHN": "china",
}

DISTANCE_BANDS = [
    ("000_050km", 0, 50),
    ("050_100km", 50, 100),
    ("100_300km", 100, 300),
    ("300_500km", 300, 500),
]

# =============================================================================
# 2. GENERAL HELPERS
# =============================================================================

def read_csv(path, **options):
    """Read UTF-8 or common Thai-encoded CSV files."""
    last_error = None
    for encoding in [CSV_ENCODING, TEXT_ENCODING, "cp874", "tis-620"]:
        try:
            frame = pd.read_csv(path, encoding=encoding, **options)
            frame.columns = [
                str(column).replace("\ufeff", "").strip()
                for column in frame.columns
            ]
            return frame
        except UnicodeDecodeError as error:
            last_error = error
    raise last_error


def csv_encoding(path):
    """Detect an encoding supported by the Thai CSV inputs."""
    with open(path, "rb") as file:
        sample = file.read(256_000)
    for encoding in [CSV_ENCODING, TEXT_ENCODING, "cp874", "tis-620"]:
        try:
            sample.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path}")


def normalize_headers(frame):
    """Remove BOM characters and repeated spaces from column names."""
    result = frame.copy()
    result.columns = [
        re.sub(r"\s+", " ", str(column).replace("\ufeff", "").strip())
        for column in result.columns
    ]
    return result


def clean_text(series):
    """Standardize blank text without changing meaningful source values."""
    return (
        series.astype("string")
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .replace({
            "": pd.NA,
            "nan": pd.NA,
            "NaN": pd.NA,
            "None": pd.NA,
            "<NA>": pd.NA,
            "null": pd.NA,
            "NULL": pd.NA,
        })
    )


def numeric(series):
    """Convert comma-formatted source numbers to numeric values."""
    return pd.to_numeric(
        clean_text(series).str.replace(",", "", regex=False),
        errors="coerce",
    )


def clean_key(series, width=2):
    """Standardize codes such as 50, TH50, 50.0, or =\"00050\"."""
    text = clean_text(series).str.replace(r"\.0$", "", regex=True)
    text = text.str.extract(r"(\d+)", expand=False)
    return text.str.zfill(width).astype("string")


def parse_iso_date(series):
    """Parse an ISO-like timestamp by its YYYY-MM-DD calendar component."""
    text = clean_text(series).str.slice(0, 10)
    return pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")


def parse_thai_calendar_date(series):
    """Parse supported CE/BE dates without guessing month/day order."""
    text = clean_text(series).str.replace(r"\.0$", "", regex=True)
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    iso = text.str.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}(?:[ T].*)?", na=False)
    if iso.any():
        pieces = text.loc[iso].str.extract(r"^(\d{4})-(\d{1,2})-(\d{1,2})")
        year = pd.to_numeric(pieces[0], errors="coerce")
        year = year.where(~year.between(2400, 2699), year - 543)
        standardized = (
            year.astype("Int64").astype("string")
            + "-" + pieces[1].str.zfill(2)
            + "-" + pieces[2].str.zfill(2)
        )
        result.loc[iso] = pd.to_datetime(
            standardized, format="%Y-%m-%d", errors="coerce"
        )

    dmy = text.str.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}(?:[ T].*)?", na=False)
    if dmy.any():
        pieces = text.loc[dmy].str.extract(r"^(\d{1,2})/(\d{1,2})/(\d{4})")
        year = pd.to_numeric(pieces[2], errors="coerce")
        year = year.where(~year.between(2400, 2699), year - 543)
        standardized = (
            year.astype("Int64").astype("string")
            + "-" + pieces[1].str.zfill(2)
            + "-" + pieces[0].str.zfill(2)
        )
        result.loc[dmy] = pd.to_datetime(
            standardized, format="%Y-%m-%d", errors="coerce"
        )

    compact = text.str.fullmatch(r"\d{8}", na=False)
    if compact.any():
        pieces = text.loc[compact].str.extract(r"^(\d{4})(\d{2})(\d{2})$")
        year = pd.to_numeric(pieces[0], errors="coerce")
        year = year.where(~year.between(2400, 2699), year - 543)
        standardized = (
            year.astype("Int64").astype("string")
            + "-" + pieces[1] + "-" + pieces[2]
        )
        result.loc[compact] = pd.to_datetime(
            standardized, format="%Y-%m-%d", errors="coerce"
        )

    return result.dt.normalize()


def first_column(frame, choices, table_name):
    """Return the first available column from a documented list."""
    for column in choices:
        if column in frame.columns:
            return column
    raise ValueError(f"{table_name} needs one of these columns: {choices}")


def optional_column(frame, choices):
    """Return an available optional column, otherwise None."""
    for column in choices:
        if column in frame.columns:
            return column
    return None


def to_boolean(series):
    """Convert common true/false representations to pandas Boolean values."""
    text = series.astype("string").str.strip().str.lower()
    result = pd.Series(pd.NA, index=series.index, dtype="boolean")
    result.loc[text.isin(["true", "1", "yes", "y", "pass", "matched"])] = True
    result.loc[text.isin(["false", "0", "no", "n", "fail", "unmatched"])] = False
    return result


def map_province_key(frame):
    """Add the two-digit province key from a code or province name."""
    result = frame.copy()
    key_column = optional_column(result, [
        "province_key", "province_code", "changwat_code", "prov_code",
        "chwpart", "รหัสจังหวัด",
    ])
    if key_column:
        result["province_key"] = clean_key(result[key_column])
        return result

    name_column = optional_column(result, [
        "province", "province_name", "province_name_th", "province_name_en",
        "changwat", "จังหวัด", "ชื่อจังหวัด",
    ])
    if not name_column:
        raise ValueError("A province key or province name column is required")

    name_map = {}
    for row in PROVINCES.itertuples(index=False):
        name_map[row.province_name_th.casefold()] = row.province_key
        name_map[row.province_name_en.casefold()] = row.province_key
    cleaned = clean_text(result[name_column]).str.casefold()
    cleaned = cleaned.str.replace(r"^จังหวัด", "", regex=True).str.strip()
    result["province_key"] = cleaned.map(name_map).astype("string")
    return result


def save_table(frame, filename, date_columns=()):
    """Atomically write one reproducible source table."""
    SOURCE_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = SOURCE_TABLE_DIR / filename
    temporary = path.with_suffix(path.suffix + ".part")
    output = frame.copy()
    for column in date_columns:
        if column in output.columns:
            output[column] = pd.to_datetime(
                output[column], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
    output.to_csv(temporary, index=False, encoding=CSV_ENCODING)
    temporary.replace(path)
    return path


def load_fetch_module():
    """Load the neighboring acquisition script without a package install."""
    candidates = [SCRIPT_DIR / "fetch_data.py", PROJECT_DIR / "src" / "fetch_data.py"]
    fetch_path = next((path for path in candidates if path.is_file()), None)
    if fetch_path is None:
        raise FileNotFoundError("fetch_data.py must be beside this script in src/")
    spec = importlib.util.spec_from_file_location("project_fetch_data", fetch_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {fetch_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_fetch_completed():
    """Stop before preparation when fetch_data reported an incomplete source."""
    if not RAW_DIR.is_dir():
        raise FileNotFoundError(
            f"Missing {RAW_DIR}. Run python src/fetch_data.py first."
        )
    if not FETCH_STATUS_FILE.is_file():
        raise FileNotFoundError(
            f"Missing {FETCH_STATUS_FILE}. Run python src/fetch_data.py first."
        )
    status = read_csv(FETCH_STATUS_FILE, low_memory=False)
    if "complete" not in status.columns:
        raise ValueError("fetch_source_status.csv has no complete column")
    complete = to_boolean(status["complete"]).fillna(False)
    if not complete.all():
        missing = status.loc[~complete, "dataset"].astype(str).tolist()
        raise RuntimeError("Raw acquisition is incomplete: " + "; ".join(missing))


def build_api_adapters(adapter_dir):
    """Parse existing API responses without making live requests."""
    acquisition = load_fetch_module()
    acquisition.PROJECT_DIR = PROJECT_DIR
    acquisition.RAW_DIR = RAW_DIR
    acquisition.ADAPTER_DIR = Path(adapter_dir)
    acquisition.FETCH_LIVE = False
    acquisition.REQUEST_RECORDS = {}

    def block_live_request(*_args, **_kwargs):
        raise FileNotFoundError(
            "A raw API response needed by prepare_tables.py is missing. "
            "Run python src/fetch_data.py again before preparing tables."
        )

    # The acquisition functions are reused only as parsers here. Blocking the
    # request helper guarantees that this stage cannot silently fetch or alter
    # the raw snapshot.
    acquisition.request_bytes = block_live_request

    acquisition.fetch_weather()
    acquisition.fetch_modeled_pm25()
    acquisition.fetch_openaq()
    return Path(adapter_dir), acquisition


# =============================================================================
# 3. ENVIRONMENTAL SOURCE TABLES
# =============================================================================

def prepare_observed_tables(observed_path, location_path):
    """Create sensor-day and province-day OpenAQ PM2.5 tables."""
    observed = map_province_key(read_csv(observed_path, low_memory=False))
    observed["analysis_date"] = parse_iso_date(observed["analysis_date_bkk"])
    observed["pm25_ugm3"] = numeric(observed["pm25"])
    observed["percent_coverage"] = numeric(observed["percent_coverage"])
    observed["primary_analysis_eligible"] = to_boolean(
        observed["primary_analysis_eligible"]
    ).fillna(False)
    keep = [
        "province_key", "province_name_th", "province_name_en",
        "location_id", "location_name", "sensor_id", "sensor_name",
        "latitude", "longitude", "analysis_date", "pm25_ugm3",
        "percent_coverage", "primary_analysis_eligible", "source",
    ]
    sensor_day = observed[[column for column in keep if column in observed]].copy()
    sensor_day = sensor_day.sort_values(["sensor_id", "analysis_date"])
    if sensor_day.duplicated(["sensor_id", "analysis_date"]).any():
        raise ValueError("OpenAQ contains duplicate sensor-day keys")

    eligible = sensor_day.loc[
        sensor_day["primary_analysis_eligible"]
        & sensor_day["pm25_ugm3"].between(0, 500)
        & sensor_day["province_key"].isin(PROVINCES["province_key"])
    ].copy()
    if eligible.empty:
        raise ValueError("No eligible OpenAQ observations were found")

    grouped = eligible.groupby(["province_key", "analysis_date"], observed=True)
    province_day = grouped["pm25_ugm3"].agg(
        observed_pm25_median_ugm3="median",
        observed_pm25_mean_ugm3="mean",
        observed_pm25_min_ugm3="min",
        observed_pm25_max_ugm3="max",
        observed_pm25_sd_across_sensors_ugm3="std",
        observed_primary_row_count="size",
    ).reset_index()
    province_day = province_day.merge(
        grouped["sensor_id"].nunique().rename("observed_sensor_count").reset_index(),
        on=["province_key", "analysis_date"], how="left",
    )
    province_day = province_day.merge(
        grouped["location_id"].nunique().rename("observed_location_count").reset_index(),
        on=["province_key", "analysis_date"], how="left",
    )

    locations = map_province_key(read_csv(location_path, low_memory=False))
    locations["latitude"] = numeric(locations["latitude"])
    locations["longitude"] = numeric(locations["longitude"])
    locations = locations.loc[
        locations["province_key"].isin(PROVINCES["province_key"])
        & locations["latitude"].between(-90, 90)
        & locations["longitude"].between(-180, 180)
    ].drop_duplicates("location_id").sort_values("location_id")

    used_location_ids = set(eligible["location_id"].dropna().astype(str))
    sensor_locations = locations.loc[
        locations["location_id"].astype(str).isin(used_location_ids),
        ["province_key", "latitude", "longitude"],
    ].drop_duplicates()
    missing_provinces = set(PROVINCES["province_key"]) - set(
        sensor_locations["province_key"]
    )
    if missing_provinces:
        sensor_locations = pd.concat([
            sensor_locations,
            PROVINCES.loc[
                PROVINCES["province_key"].isin(missing_provinces),
                ["province_key", "latitude", "longitude"],
            ],
        ], ignore_index=True)

    stats = {
        "openaq_sensor_day_rows": len(sensor_day),
        "openaq_eligible_sensor_day_rows": len(eligible),
        "openaq_province_day_rows": len(province_day),
        "openaq_locations": len(locations),
        "openaq_centroid_fallback_provinces": len(missing_provinces),
    }
    return sensor_day, province_day, locations, sensor_locations, stats


def prepare_weather_table(path):
    """Standardize the Open-Meteo daily weather table."""
    frame = map_province_key(read_csv(path, low_memory=False))
    frame["analysis_date"] = parse_iso_date(frame["analysis_date_bkk"])
    mapping = {
        "temperature_mean_c": "temperature_2m_mean",
        "relative_humidity_mean_pct": "relative_humidity_2m_mean",
        "precipitation_mm": "precipitation_sum",
        "wind_speed_mean_kmh": "wind_speed_10m_mean",
        "wind_direction_dominant_deg": "wind_direction_10m_dominant",
        "surface_pressure_mean_hpa": "surface_pressure_mean",
    }
    for new_name, source_name in mapping.items():
        frame[new_name] = numeric(frame[source_name])
    keep = ["province_key", "analysis_date"] + list(mapping)
    result = frame.loc[
        frame["province_key"].isin(PROVINCES["province_key"]), keep
    ].sort_values(["province_key", "analysis_date"])
    if result.duplicated(["province_key", "analysis_date"]).any():
        raise ValueError("Weather contains duplicate province-day keys")
    return result


def prepare_modeled_pm25_table(path):
    """Aggregate the optional hourly CAMS benchmark to province-day."""
    frame = map_province_key(read_csv(path, low_memory=False))
    frame["analysis_date"] = parse_iso_date(frame["analysis_date_bkk"])
    frame["modeled_pm25_ugm3"] = numeric(frame["pm25_modeled"])
    frame = frame.loc[
        frame["province_key"].isin(PROVINCES["province_key"])
        & frame["modeled_pm25_ugm3"].ge(0)
    ]
    return frame.groupby(
        ["province_key", "analysis_date"], observed=True
    )["modeled_pm25_ugm3"].agg(
        modeled_pm25_mean_ugm3="mean",
        modeled_pm25_median_ugm3="median",
        modeled_pm25_hour_count="size",
    ).reset_index()


def normalize_confidence(series):
    """Standardize FIRMS low, nominal, and high confidence labels."""
    text = series.astype("string").str.strip().str.upper()
    return pd.Series(np.select(
        [
            text.isin(["H", "HIGH"]),
            text.isin(["N", "NOMINAL", "MEDIUM"]),
            text.isin(["L", "LOW"]),
        ],
        ["high", "nominal", "low"],
        default="unknown",
    ), index=series.index)


def prepare_hotspot_table(acquisition, sensor_locations):
    """Clip FIRMS area responses and aggregate eligible events by distance band."""
    files = sorted((RAW_DIR / "api" / "nasa_firms").glob(
        "*/VIIRS_SNPP_SP_AREA_*.csv"
    ))
    if not files:
        raise FileNotFoundError("No raw NASA FIRMS Area API files were found")

    geometries = {
        code: acquisition.load_country_geometry(code) for code in COUNTRIES
    }
    trees = {}
    for province_key, locations in sensor_locations.groupby("province_key"):
        coordinates = np.radians(
            locations[["latitude", "longitude"]].to_numpy(float)
        )
        trees[province_key] = BallTree(coordinates, metric="haversine")

    parts = []
    raw_rows = 0
    inside_rows = 0
    eligible_rows = 0
    deduplicated_rows = 0

    for position, path in enumerate(files, start=1):
        country_code = path.parent.name.upper()
        if country_code not in COUNTRIES:
            raise ValueError(f"Unknown FIRMS country folder: {path.parent.name}")
        frame = read_csv(path, low_memory=False)
        raw_rows += len(frame)
        required = {"acq_date", "acq_time", "latitude", "longitude"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} is missing FIRMS columns: {sorted(missing)}")

        frame["latitude"] = numeric(frame["latitude"])
        frame["longitude"] = numeric(frame["longitude"])
        valid_coordinates = (
            frame["latitude"].between(-90, 90)
            & frame["longitude"].between(-180, 180)
        )
        inside = np.zeros(len(frame), dtype=bool)
        if valid_coordinates.any():
            valid_positions = np.flatnonzero(valid_coordinates.to_numpy())
            inside[valid_positions] = acquisition.points_inside_country(
                frame["longitude"].iloc[valid_positions].to_numpy(),
                frame["latitude"].iloc[valid_positions].to_numpy(),
                geometries[country_code],
            )
        frame = frame.loc[inside].copy()
        inside_rows += len(frame)
        if frame.empty:
            continue

        frame["analysis_date"] = parse_iso_date(frame["acq_date"])
        primary = pd.Series(True, index=frame.index)
        if "confidence" in frame.columns:
            primary &= normalize_confidence(frame["confidence"]).isin([
                "nominal", "high"
            ])
        if "satellite" in frame.columns:
            satellite = frame["satellite"].astype("string").str.upper().str.replace(
                r"[^A-Z0-9]", "", regex=True
            )
            primary &= satellite.isin(["N", "SNPP", "SUOMINPP"])
        if "instrument" in frame.columns:
            instrument = frame["instrument"].astype("string").str.upper().str.replace(
                r"[^A-Z0-9]", "", regex=True
            )
            primary &= instrument.eq("VIIRS")

        frame = frame.loc[
            primary & frame["analysis_date"].between(HOTSPOT_START, AT1_END)
        ].copy()
        eligible_rows += len(frame)
        key_columns = [
            column for column in [
                "acq_date", "acq_time", "latitude", "longitude",
                "satellite", "instrument",
            ] if column in frame.columns
        ]
        before = len(frame)
        frame = frame.drop_duplicates(key_columns)
        deduplicated_rows += before - len(frame)
        if frame.empty:
            continue
        frame["country"] = COUNTRIES[country_code]
        event_coordinates = np.radians(
            frame[["latitude", "longitude"]].to_numpy(float)
        )

        for province_key, tree in trees.items():
            distance, _ = tree.query(event_coordinates, k=1)
            distance_km = distance[:, 0] * EARTH_RADIUS_KM
            selected = distance_km <= 500
            if not selected.any():
                continue
            near = frame.loc[selected, ["analysis_date", "country"]].copy()
            near["province_key"] = province_key
            near_distance = distance_km[selected]
            near["band"] = np.select(
                [
                    near_distance <= 50,
                    (near_distance > 50) & (near_distance <= 100),
                    (near_distance > 100) & (near_distance <= 300),
                    (near_distance > 300) & (near_distance <= 500),
                ],
                [band for band, _lower, _upper in DISTANCE_BANDS],
                default="outside",
            )
            parts.append(
                near.groupby(
                    ["province_key", "analysis_date", "country", "band"],
                    observed=True,
                ).size().rename("count").reset_index()
            )

        if position % 100 == 0 or position == len(files):
            print(f"    FIRMS files processed: {position:,}/{len(files):,}")

    if not parts:
        raise ValueError("No eligible VIIRS S-NPP events were found")
    long = pd.concat(parts, ignore_index=True).groupby(
        ["province_key", "analysis_date", "country", "band"],
        observed=True, as_index=False,
    )["count"].sum()
    long["column_name"] = (
        "hotspot_count_" + long["country"] + "_" + long["band"] + "_lag0"
    )
    wide = long.pivot_table(
        index=["province_key", "analysis_date"],
        columns="column_name",
        values="count",
        fill_value=0,
        aggfunc="sum",
    ).reset_index()
    wide.columns.name = None

    grid = PROVINCES[["province_key"]].assign(join_key=1).merge(
        pd.DataFrame({
            "analysis_date": pd.date_range(HOTSPOT_START, AT1_END, freq="D")
        }).assign(join_key=1),
        on="join_key",
    ).drop(columns="join_key")
    result = grid.merge(wide, on=["province_key", "analysis_date"], how="left")
    expected_columns = [
        f"hotspot_count_{country}_{band}_lag0"
        for country in COUNTRIES.values()
        for band, _lower, _upper in DISTANCE_BANDS
    ]
    for column in expected_columns:
        if column not in result.columns:
            result[column] = 0
        result[column] = numeric(result[column]).fillna(0).astype("int64")
    result = result[["province_key", "analysis_date"] + expected_columns]

    stats = {
        "firms_raw_area_rows": raw_rows,
        "firms_rows_inside_requested_country": inside_rows,
        "firms_rows_after_product_confidence_date_filters": eligible_rows,
        "firms_exact_duplicate_rows_removed": deduplicated_rows,
        "firms_source_table_rows": len(result),
    }
    return result, stats


# =============================================================================
# 4. HEALTH AND CAPACITY SOURCE TABLES
# =============================================================================

def read_population_html_xls(path):
    """Find the DOPA population table inside an HTML-based .xls file.

    Some DOPA yearly exports contain layout or nested HTML tables in addition
    to the actual data table. Therefore, the parser searches every table and
    accepts the one containing the four documented population headers instead
    of assuming that exactly one HTML table exists.
    """
    try:
        tables = pd.read_html(str(path), encoding=TEXT_ENCODING)
    except ImportError as error:
        raise RuntimeError("Reading stat_c*.xls requires lxml") from error

    required = [
        "รหัสจังหวัด",
        "จำนวนประชากรชาย",
        "จำนวนประชากรหญิง",
        "จำนวนประชากรทั้งหมด",
    ]

    def label(value):
        """Convert a normal or MultiIndex header cell to clean text."""
        values = value if isinstance(value, tuple) else [value]
        parts = [
            re.sub(r"\s+", " ", str(item).replace("\ufeff", "").strip())
            for item in values
            if pd.notna(item)
        ]
        documented = next((item for item in parts if item in required), None)
        return documented or (parts[-1] if parts else "")

    candidates = []
    for table_number, source in enumerate(tables):
        if source.empty:
            continue

        possible_headers = [(None, [label(column) for column in source.columns])]
        for row_position in range(min(30, len(source))):
            possible_headers.append((
                row_position,
                [label(value) for value in source.iloc[row_position].tolist()],
            ))

        for header_position, headers in possible_headers:
            if not set(required).issubset(headers):
                continue
            data = source.copy() if header_position is None else source.iloc[
                header_position + 1:
            ].copy()
            selected = pd.DataFrame({
                column: data.iloc[:, headers.index(column)]
                for column in required
            }).dropna(how="all")
            province_keys = clean_key(selected["รหัสจังหวัด"])
            score = int(province_keys.str.fullmatch(r"\d{2}", na=False).sum())
            candidates.append((score, len(selected), -table_number, selected))

    if not candidates:
        raise ValueError(
            f"Population headers were not found in any HTML table in {path.name}"
        )

    # Prefer the candidate with the largest number of province-code rows. The
    # row count and original table order are deterministic tie-breakers.
    candidates.sort(key=lambda item: item[:3], reverse=True)
    return normalize_headers(candidates[0][3].reset_index(drop=True))


def prepare_facility_table(path):
    """Create the five-digit facility-to-province dimension."""
    frame = map_province_key(normalize_headers(
        read_csv(path, dtype="string", low_memory=False)
    ))
    key_column = first_column(
        frame, ["facility_key", "hcode", "รหัส 5 หลัก"], path.name
    )
    frame["facility_key"] = clean_key(frame[key_column], width=5)
    result = frame.loc[
        frame["province_key"].isin(PROVINCES["province_key"])
        & frame["facility_key"].notna(),
        ["facility_key", "province_key"],
    ].drop_duplicates()
    conflicts = result.groupby("facility_key")["province_key"].nunique()
    if conflicts.gt(1).any():
        raise ValueError("A facility maps to more than one study province")
    result = result.drop_duplicates("facility_key").sort_values("facility_key")
    if result.empty:
        raise ValueError("No study-province facilities were found")
    return result


def prepare_population_table(paths):
    """Parse and validate the four official DOPA population files."""
    parts = []
    for path in paths:
        match = re.fullmatch(r"stat_c(\d{2})\.xls", path.name, flags=re.I)
        if not match:
            raise ValueError(f"Unexpected population filename: {path.name}")
        year_be = 2500 + int(match.group(1))
        frame = map_province_key(read_population_html_xls(path))
        frame["year_ce"] = year_be - 543
        frame["male_population"] = numeric(frame["จำนวนประชากรชาย"])
        frame["female_population"] = numeric(frame["จำนวนประชากรหญิง"])
        frame["total_population"] = numeric(frame["จำนวนประชากรทั้งหมด"])
        frame = frame.loc[
            frame["province_key"].isin(PROVINCES["province_key"]),
            [
                "province_key", "year_ce", "male_population",
                "female_population", "total_population",
            ],
        ]
        mismatch = (
            frame["male_population"] + frame["female_population"]
            != frame["total_population"]
        )
        if mismatch.any():
            raise ValueError(f"Population components do not sum in {path.name}")
        parts.append(frame)
    result = pd.concat(parts, ignore_index=True).sort_values([
        "province_key", "year_ce"
    ])
    if result.duplicated(["province_key", "year_ce"]).any():
        raise ValueError("Population contains duplicate province-year keys")
    if result["total_population"].isna().any() or result["total_population"].le(0).any():
        raise ValueError("Population totals must be present and positive")
    required_years = {2022, 2023, 2024, 2025}
    if not required_years.issubset(set(result["year_ce"].astype(int))):
        raise ValueError("Population files do not cover 2022-2025")
    return result


def prepare_capacity_table(path):
    """Normalize the undated HAI opened-bed context table."""
    frame = map_province_key(normalize_headers(read_csv(path, low_memory=False)))
    key_column = first_column(
        frame, ["facility_key", "hcode", "รหัส H Code"], path.name
    )
    beds_column = first_column(
        frame, ["จำนวนเตียงเปิดจริง", "opened_beds", "licensed_beds"], path.name
    )
    frame["facility_key"] = clean_key(frame[key_column], width=5)
    frame["opened_beds"] = numeric(frame[beds_column])
    result = frame.loc[
        frame["province_key"].isin(PROVINCES["province_key"])
        & frame["facility_key"].notna(),
        ["facility_key", "province_key", "opened_beds"],
    ].copy()
    if result.empty:
        raise ValueError("No study-province HAI capacity rows were found")
    if result.duplicated("facility_key").any():
        raise ValueError("HAI capacity contains duplicate facility keys")
    if result["opened_beds"].lt(0).fillna(False).any():
        raise ValueError("HAI capacity contains a negative opened-bed value")
    return result.sort_values("facility_key")


def canonical_diagnosis_chunk(chunk):
    """Return the canonical fields needed for DDC respiratory aggregation."""
    chunk = normalize_headers(chunk)
    required = ["hospcode", "date_serv", "diagtype", "diagcode"]
    missing = [column for column in required if column not in chunk]
    if missing:
        raise ValueError(f"Raw DDC file is missing columns: {missing}")
    diagnosis_code = (
        clean_text(chunk["diagcode"])
        .str.upper()
        .str.replace(".", "", regex=False)
    )
    valid_icd = diagnosis_code.str.fullmatch(
        r"[A-Z][0-9]{2}[0-9A-Z]{0,4}", na=False
    )
    return pd.DataFrame({
        "facility_key": clean_key(chunk["hospcode"], width=5),
        "service_date": parse_thai_calendar_date(chunk["date_serv"]),
        "valid_icd": valid_icd,
        "is_respiratory": valid_icd & diagnosis_code.str.match(
            r"^J(?:[0-8][0-9]|9[0-9])", na=False
        ),
        "diagtype": numeric(chunk["diagtype"]),
    })


def prepare_respiratory_table(paths, facility):
    """Stream DDC records into transparent province-month counts."""
    facility_map = dict(facility[["facility_key", "province_key"]].itertuples(
        index=False, name=None
    ))
    monthly_parts = []
    reporter_parts = []
    scanned = 0
    valid_icd_rows = 0
    unmapped_facility_rows = 0

    for path in paths:
        for source_chunk in pd.read_csv(
            path,
            encoding=csv_encoding(path),
            dtype="string",
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ):
            scanned += len(source_chunk)
            chunk = canonical_diagnosis_chunk(source_chunk)
            chunk["province_key"] = chunk["facility_key"].map(facility_map)
            chunk["year_month"] = chunk["service_date"].dt.to_period("M").astype("string")
            in_period = chunk["service_date"].between(AT2_START, AT2_END)
            valid_icd_rows += int((in_period & chunk["valid_icd"]).sum())
            unmapped_facility_rows += int(
                (in_period & chunk["valid_icd"] & chunk["province_key"].isna()).sum()
            )
            chunk = chunk.loc[
                in_period
                & chunk["valid_icd"]
                & chunk["province_key"].isin(PROVINCES["province_key"])
            ].copy()
            if chunk.empty:
                continue
            respiratory = chunk["is_respiratory"] & chunk["diagtype"].isin([1, 2, 3])
            chunk["all_diagnosis_record_count"] = 1
            chunk["respiratory_record_count_any_position"] = respiratory.astype(int)
            chunk["respiratory_record_count_principal"] = (
                respiratory & chunk["diagtype"].eq(1)
            ).astype(int)
            monthly_parts.append(chunk.groupby(
                ["province_key", "year_month"], observed=True
            )[[
                "all_diagnosis_record_count",
                "respiratory_record_count_any_position",
                "respiratory_record_count_principal",
            ]].sum().reset_index())
            reporter_parts.append(chunk[[
                "province_key", "year_month", "facility_key"
            ]].drop_duplicates())

    if not monthly_parts:
        raise ValueError("No valid study-province DDC rows were found")
    monthly = pd.concat(monthly_parts, ignore_index=True).groupby(
        ["province_key", "year_month"], observed=True, as_index=False
    ).sum()
    reporters = pd.concat(reporter_parts, ignore_index=True).drop_duplicates()
    reporters = reporters.groupby(
        ["province_key", "year_month"], observed=True
    )["facility_key"].nunique().rename(
        "active_reporting_facility_count"
    ).reset_index()
    result = monthly.merge(reporters, on=["province_key", "year_month"], how="left")
    stats = {
        "ddc_rows_scanned": scanned,
        "ddc_valid_icd_rows_in_at2_period": valid_icd_rows,
        "ddc_valid_icd_rows_with_unmapped_facility": unmapped_facility_rows,
        "ddc_province_month_rows": len(result),
    }
    return result.sort_values(["province_key", "year_month"]), stats


# =============================================================================
# 5. QUALITY AND COVERAGE REPORTS
# =============================================================================

TABLE_KEYS = {
    "provinces.csv": ["province_key"],
    "location.csv": ["location_id"],
    "observepm25_sensorday.csv": ["sensor_id", "analysis_date"],
    "observepm25_provinceday.csv": ["province_key", "analysis_date"],
    "weather_provinceday.csv": ["province_key", "analysis_date"],
    "modelpm25_provinceday.csv": ["province_key", "analysis_date"],
    "hotspot.csv": ["province_key", "analysis_date"],
    "facility.csv": ["facility_key"],
    "population.csv": ["province_key", "year_ce"],
    "hospital.csv": ["facility_key"],
    "diagnosis.csv": ["province_key", "year_month"],
}


def audit_tables(tables):
    """Create schema, missingness, duplicate, and foreign-key reports."""
    quality_rows = []
    missing_rows = []
    dtype_rows = []
    duplicate_rows = []
    duplicate_examples = []
    foreign_key_rows = []
    known_provinces = set(PROVINCES["province_key"])

    for filename, frame in tables.items():
        keys = TABLE_KEYS[filename]
        missing_keys = int(frame[keys].isna().any(axis=1).sum())
        exact_duplicates = int(frame.duplicated().sum())
        duplicate_keys = int(frame.duplicated(keys, keep=False).sum())
        invalid_province_rows = 0
        if "province_key" in frame.columns:
            invalid = frame["province_key"].notna() & ~frame["province_key"].isin(
                known_provinces
            )
            invalid_province_rows = int(invalid.sum())
            foreign_key_rows.append({
                "table": filename,
                "foreign_key": "province_key",
                "invalid_row_count": invalid_province_rows,
                "status": "PASS" if invalid_province_rows == 0 else "FAIL",
            })

        status = "PASS"
        if missing_keys or duplicate_keys or invalid_province_rows:
            status = "FAIL"
        elif frame.isna().any().any():
            status = "REVIEW"
        quality_rows.append({
            "table": filename,
            "row_count": len(frame),
            "column_count": len(frame.columns),
            "primary_key": ";".join(keys),
            "missing_primary_key_rows": missing_keys,
            "exact_duplicate_rows": exact_duplicates,
            "duplicate_primary_key_rows": duplicate_keys,
            "missing_cell_count": int(frame.isna().sum().sum()),
            "invalid_province_key_rows": invalid_province_rows,
            "status": status,
        })
        duplicate_rows.append({
            "table": filename,
            "key": ";".join(keys),
            "exact_duplicate_rows": exact_duplicates,
            "duplicate_primary_key_rows": duplicate_keys,
        })
        if duplicate_keys:
            examples = frame.loc[frame.duplicated(keys, keep=False), keys].head(20)
            for row in examples.astype("string").to_dict("records"):
                duplicate_examples.append({"table": filename, **row})
        for column in frame.columns:
            missing_count = int(frame[column].isna().sum())
            missing_rows.append({
                "table": filename,
                "column": column,
                "missing_count": missing_count,
                "missing_pct": 100 * missing_count / len(frame) if len(frame) else np.nan,
            })
            dtype_rows.append({
                "table": filename,
                "column": column,
                "dtype": str(frame[column].dtype),
            })

    reports = {
        "table_quality_report.csv": pd.DataFrame(quality_rows),
        "missing_value_report.csv": pd.DataFrame(missing_rows),
        "dtype_report.csv": pd.DataFrame(dtype_rows),
        "duplicate_report.csv": pd.DataFrame(duplicate_rows),
        "duplicate_key_examples.csv": pd.DataFrame(duplicate_examples),
        "foreign_key_report.csv": pd.DataFrame(foreign_key_rows),
    }
    failed = [row["table"] for row in quality_rows if row["status"] == "FAIL"]
    return reports, failed


def source_coverage_report(tables):
    """Summarize date/province coverage before any analytical merge."""
    rows = []
    expected_at1 = len(PROVINCES) * len(pd.date_range(AT1_START, AT1_END))
    expected_at2 = len(PROVINCES) * len(pd.date_range(AT2_START, AT2_END, freq="MS"))
    for filename, frame in tables.items():
        date_column = next((column for column in [
            "analysis_date", "year_month", "year_ce"
        ] if column in frame), None)
        minimum = maximum = ""
        if date_column is not None and not frame.empty:
            minimum = str(frame[date_column].dropna().min())
            maximum = str(frame[date_column].dropna().max())
        expected = np.nan
        if filename in {
            "observepm25_provinceday.csv",
            "weather_provinceday.csv",
        }:
            in_scope = frame["analysis_date"].between(AT1_START, AT1_END)
            available = int(frame.loc[in_scope, ["province_key", "analysis_date"]].drop_duplicates().shape[0])
            expected = expected_at1
        elif filename == "hotspot.csv":
            in_scope = frame["analysis_date"].between(AT1_START, AT1_END)
            available = int(frame.loc[in_scope, ["province_key", "analysis_date"]].drop_duplicates().shape[0])
            expected = expected_at1
        elif filename == "diagnosis.csv":
            available = len(frame)
            expected = expected_at2
        else:
            available = len(frame)
        rows.append({
            "table": filename,
            "row_count": len(frame),
            "available_key_rows_in_analysis_scope": available,
            "expected_key_rows_in_analysis_scope": expected,
            "coverage_pct": (
                100 * available / expected if pd.notna(expected) and expected else np.nan
            ),
            "province_count": (
                frame["province_key"].nunique() if "province_key" in frame else np.nan
            ),
            "minimum_time_key": minimum,
            "maximum_time_key": maximum,
        })
    return pd.DataFrame(rows)


def processing_decisions_table():
    """Document decisions that materially affect the prepared tables."""
    rows = [
        ("OpenAQ eligibility", "Keep daily sensor values from 0 to 500 µg/m³ with at least 75% coverage", "Exclude implausible or insufficiently covered daily summaries"),
        ("Observed PM2.5 aggregation", "Use the median across eligible sensors for the primary province-day value; retain mean, range, SD, and counts", "Median reduces sensitivity to disagreement between sensors without hiding coverage"),
        ("FIRMS product", "Use VIIRS S-NPP Standard Processing only", "Maintain one consistent satellite product"),
        ("FIRMS spatial filter", "Clip every Area API response to its GADM 4.1 country boundary", "The Area API bounding box contains points outside the requested country"),
        ("FIRMS confidence", "Keep nominal and high confidence detections", "Low-confidence detections are excluded from primary features"),
        ("FIRMS distance", "Count events by nearest eligible OpenAQ location within 0-50, 50-100, 100-300, and 300-500 km", "Preserve distance information while keeping a province-day grain"),
        ("FIRMS zero", "A complete downloaded day with no eligible event is stored as zero", "Distinguish confirmed zero from an unavailable source"),
        ("Weather", "Use Open-Meteo ERA5 daily values in Asia/Bangkok time", "Match the province-day analytical grain"),
        ("Modeled PM2.5", "Aggregate CAMS hourly values to daily mean and median", "Use as a descriptive benchmark, not the observed target"),
        ("Respiratory definition", "Valid ICD-10 J00-J99 records with diagtype 1, 2, or 3", "Retain principal and any-position respiratory counts separately"),
        ("Capacity", "Use opened-bed values from the undated HAI AOD snapshot as context", "This is capacity context, not historical occupancy"),
        ("Missing data", "Do not impute source-table missing values", "Missingness remains visible for coverage and eligibility checks"),
    ]
    return pd.DataFrame(rows, columns=["decision", "implementation", "reason"])


# =============================================================================
# 6. MAIN WORKFLOW
# =============================================================================

def main():
    """Build all source tables, run QA, and stop on key-integrity failures."""
    verify_fetch_completed()
    SOURCE_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    QUALITY_DIR.mkdir(parents=True, exist_ok=True)

    print("Preparing source-aligned tables from the verified raw snapshot")
    processing_stats = {}

    # Validate the smaller static health sources before starting the expensive
    # FIRMS calculation. A publisher-format problem therefore fails fast.
    facility = prepare_facility_table(RAW_DIR / "PROVIDER" / "health_office.csv")
    population = prepare_population_table(sorted(
        (RAW_DIR / "POPULATIONS").glob("stat_c*.xls")
    ))
    capacity = prepare_capacity_table(RAW_DIR / "PROVIDER" / "ha_aod_001-2.csv")
    respiratory, ddc_stats = prepare_respiratory_table(
        sorted((RAW_DIR / "DDC").glob("smog_odpc1_*.csv")), facility
    )
    processing_stats.update(ddc_stats)
    print("  Static health, population, and capacity tables prepared")

    with tempfile.TemporaryDirectory(prefix="source_adapters_") as temporary:
        adapter_dir, acquisition = build_api_adapters(temporary)

        sensor_day, observed_day, locations, sensor_locations, openaq_stats = (
            prepare_observed_tables(
                adapter_dir / "OBSERVED_PM25_F.csv",
                adapter_dir / "OPENAQ_LOCATION_D.csv",
            )
        )
        processing_stats.update(openaq_stats)
        print(f"  OpenAQ province-day rows: {len(observed_day):,}")

        weather = prepare_weather_table(adapter_dir / "DAILY_WEATHER_F.csv")
        print(f"  Weather province-day rows: {len(weather):,}")

        modeled = prepare_modeled_pm25_table(adapter_dir / "MODELED_PM25_F.csv")
        print(f"  Modeled PM2.5 province-day rows: {len(modeled):,}")

        hotspots, hotspot_stats = prepare_hotspot_table(acquisition, sensor_locations)
        processing_stats.update(hotspot_stats)
        print(f"  Hotspot feature province-day rows: {len(hotspots):,}")

    tables = {
        "provinces.csv": PROVINCES.copy(),
        "location.csv": locations,
        "observepm25_sensorday.csv": sensor_day,
        "observepm25_provinceday.csv": observed_day,
        "weather_provinceday.csv": weather,
        "modelpm25_provinceday.csv": modeled,
        "hotspot.csv": hotspots,
        "facility.csv": facility,
        "population.csv": population,
        "hospital.csv": capacity,
        "diagnosis.csv": respiratory,
    }

    reports, failed = audit_tables(tables)
    reports["source_coverage_report.csv"] = source_coverage_report(tables)
    reports["processing_decisions.csv"] = processing_decisions_table()
    reports["source_processing_counts.csv"] = pd.DataFrame([
        {"metric": key, "value": value}
        for key, value in processing_stats.items()
    ])

    for filename, frame in tables.items():
        date_columns = [column for column in ["analysis_date"] if column in frame]
        save_table(frame, filename, date_columns=date_columns)
    for filename, report in reports.items():
        report.to_csv(QUALITY_DIR / filename, index=False, encoding=CSV_ENCODING)

    summary = {
        "source_table_directory": str(SOURCE_TABLE_DIR),
        "table_count": len(tables),
        "quality_report_directory": str(QUALITY_DIR),
        "failed_key_integrity_tables": failed,
    }
    with open(METADATA_DIR / "prepare_tables_summary.json", "w", encoding=TEXT_ENCODING) as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, sort_keys=True)

    if failed:
        raise RuntimeError(
            "Source-table key integrity failed for: " + ", ".join(failed)
            + f". See {QUALITY_DIR / 'table_quality_report.csv'}"
        )

    print("Source-table preparation completed successfully.")
    print(f"Source tables: {SOURCE_TABLE_DIR}")
    print(f"Quality reports: {QUALITY_DIR}")


if __name__ == "__main__":
    main()
