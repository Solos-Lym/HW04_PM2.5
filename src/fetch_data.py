"""Fetch or verify the raw sources used by the final HW04 AT1/AT2 workflow.

Normal reproducible run:

    FETCH_LIVE=0 python src/fetch_data.py
    
REQUIREMENTS API KEYS
Keys are read from environment variables and are never written to files
Live OpenAQ access uses ``OPENAQ_API_KEY`` >>  You can get at https://docs.openaq.org/using-the-api/api-key
Live NASA FIRMS access uses ``NASA_FIRMS_MAP_KEY`` >>  You can get at https://firms.modaps.eosdis.nasa.gov/api/map_key/

Public direct URLs for DDC, population, accreditation and boundary files are built in  
``DOWNLOAD_DIR`` remains available as a fallback when a publisher temporarily blocks an automated download:
    DOWNLOAD_DIR="/path/to/downloaded/files" FETCH_LIVE=1 \
    python src/fetch_data.py

Three optional acquisition overrides are available:

1. ``DOWNLOAD_DIR``: local folder searched recursively for downloaded source files.

2. ``RAW_BUNDLE_URL``: URL of one ZIP whose member paths match the paths in ``STATIC_SOURCES`` 
    below (it may optionally start with ``data/raw/``).
    The same ZIP may also contain the optional ``data/raw/frozen`` 
    analytical snapshots listed in ``OPTIONAL_FROZEN_SOURCES`` for exact-number reruns.
    
3. ``SOURCE_URLS_JSON``: JSON text, or a path to a JSON file, mapping each relative raw path to its direct download URL.

=============================================================================
FUNCTION GUIDE
=============================================================================

File and audit helpers
----------------------
utc_now                  -> timestamp a newly acquired file in UTC.
file_hash                -> calculate one file's SHA256 checksum.
file_facts               -> calculate SHA256 and count CSV/TSV data rows.
record_request           -> remember safe acquisition metadata for manifest.
redact_url               -> remove query values/credentials from saved URLs.
list_raw_files           -> list completed raw files, excluding temp files.
count_raw_matches        -> count raw files matching expected path patterns.
write_manifest           -> write hashes, sizes, row counts and provenance.
build_source_status      -> check whether every required source group exists.

Download and frozen-file helpers
--------------------------------
request_bytes            -> download a small API response with retry logic.
save_raw_response        -> save exact API bytes once without overwriting.
download_file            -> stream a large CSV/XLS/ZIP safely to disk.
is_population_html_xls   -> recognize DOPA's HTML table with a .xls suffix.
validate_static_file     -> reject empty/error pages or wrong file formats.
build_download_file_index-> index a fallback download folder by filename.
import_downloaded_source -> copy a known fallback export into data/raw.
load_source_url_mapping  -> read user-supplied per-file URL overrides.
normalized_bundle_member -> validate a ZIP member path before extraction.
extract_static_bundle    -> extract expected files from a raw-source ZIP.
fetch_static_sources     -> acquire DDC, HCODE, HAI, population and boundary.
read_json                -> load one frozen JSON response for parsing.

Geographic helpers
------------------
haversine_distance       -> calculate distance between two coordinates.
nearest_province         -> attach an OpenAQ location to a study province.

API acquisition functions
-------------------------
fetch_weather            -> fetch/parse Open-Meteo daily weather responses.
fetch_modeled_pm25       -> fetch/parse optional modeled PM2.5 responses.
openaq_pages             -> follow all pages returned by one OpenAQ request.
discover_openaq_sensors  -> find PM2.5 sensors near the eight provinces.
parse_openaq_day         -> turn one OpenAQ daily result into one table row.
fetch_openaq             -> fetch/parse observed PM2.5 and sensor metadata.
five_day_periods         -> split the FIRMS date range into safe API windows.
fetch_firms              -> fetch/parse VIIRS S-NPP hotspot detections.

Workflow controller
-------------------
main                     -> run live or frozen mode, then validate and report.

-*- coding: utf-8 -*-
"""

# =============================================================================
# 0. Package settings
# =============================================================================

# Standard-library modules for API and webscraping
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
import hashlib
import http.cookiejar
import http.client
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

# DataFrame + Visualization packages
import numpy as np
import pandas as pd
from matplotlib.path import Path as GeometryPath


# =============================================================================
# 1. SETTINGS
# =============================================================================

# Setting path for input and output 
SCRIPT_DIR = Path(__file__).resolve().parent

def environment_setting(name, default=""):
    """Read one optional project setting from the environment."""
    return os.environ.get(name, default)


def resolve_project_dir():
    """Find project root without depending on its directory name or cwd."""
    explicit_root = environment_setting("PROJECT_ROOT").strip()
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()

    # Normal layout: <any-project-name>/src/fetch_data.py
    if SCRIPT_DIR.name.casefold() == "src":
        return SCRIPT_DIR.parent

    # Also support placing fetch_data.py directly in the project root.
    return SCRIPT_DIR


PROJECT_DIR = resolve_project_dir()
RAW_DIR = PROJECT_DIR / "data" / "raw"
METADATA_DIR = PROJECT_DIR / "outputs" / "metadata"
MANIFEST_FILE = METADATA_DIR / "fetch_manifest.csv"
SOURCE_STATUS_FILE = METADATA_DIR / "fetch_source_status.csv"

# Normally acquisition writes raw responses and audit metadata only.
# prepare_data.py temporarily assigns ADAPTER_DIR while rebuilding analytical
# tables directly from raw API responses.
ADAPTER_DIR = None

# Python/JSON text uses UTF-8.  CSV uses UTF-8 with BOM so Thai text also opens correctly in Microsoft Excel.
TEXT_ENCODING = "utf-8"
CSV_ENCODING = "utf-8-sig"

# Live acquisition is the normal behaviour. Set FETCH_LIVE=0 only when intentionally verifying a previously saved raw snapshot.
FETCH_LIVE = environment_setting("FETCH_LIVE", "1") == "1"
OPENAQ_KEY = os.environ.get("OPENAQ_API_KEY", "").strip()
FIRMS_KEY = os.environ.get("NASA_FIRMS_MAP_KEY", "").strip()
RAW_BUNDLE_URL = environment_setting("RAW_BUNDLE_URL").strip()
SOURCE_URLS_SETTING = environment_setting("SOURCE_URLS_JSON").strip()
DOWNLOAD_DIR_SETTING = environment_setting("DOWNLOAD_DIR").strip()

OPENAQ_BASE = "https://api.openaq.org/v3"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
MODELED_PM25_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api"
GADM_BASE = "https://geodata.ucdavis.edu/gadm/gadm4.1/json"

START_YEAR = 2020
END_YEAR = 2025
FIRMS_START = pd.Timestamp("2021-04-27")
FIRMS_END = pd.Timestamp("2025-12-31")

# Setting variables from raw sources 
# Cross-border fire files were present in the original source inventory.
FIRMS_COUNTRIES = {
    "THA": "Thailand",
    "LAO": "Laos",
    "MMR": "Myanmar",
    "CHN": "China",
}

# NASA FIRMS currently provides the Area API while the Country API may be
# unavailable. Country-level GADM geometries are used to obtain each bounding
# box and to remove detections outside the requested country after download.
GADM_COUNTRY_FILES = {
    country_code: f"gadm41_{country_code}_0.json"
    for country_code in FIRMS_COUNTRIES
}

WEATHER_VARIABLES = [
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_mean",
    "wind_direction_10m_dominant",
    "surface_pressure_mean",
]

PROVINCES = [
    {"province_key": "50", "province_name_th": "เชียงใหม่", "province_name_en": "Chiang Mai", "latitude": 18.7883, "longitude": 98.9853},
    {"province_key": "51", "province_name_th": "ลำพูน", "province_name_en": "Lamphun", "latitude": 18.5745, "longitude": 99.0087},
    {"province_key": "52", "province_name_th": "ลำปาง", "province_name_en": "Lampang", "latitude": 18.2888, "longitude": 99.4909},
    {"province_key": "54", "province_name_th": "แพร่", "province_name_en": "Phrae", "latitude": 18.1446, "longitude": 100.1403},
    {"province_key": "55", "province_name_th": "น่าน", "province_name_en": "Nan", "latitude": 18.7756, "longitude": 100.7730},
    {"province_key": "56", "province_name_th": "พะเยา", "province_name_en": "Phayao", "latitude": 19.1665, "longitude": 99.9019},
    {"province_key": "57", "province_name_th": "เชียงราย", "province_name_en": "Chiang Rai", "latitude": 19.9105, "longitude": 99.8406},
    {"province_key": "58", "province_name_th": "แม่ฮ่องสอน", "province_name_en": "Mae Hong Son", "latitude": 19.3013, "longitude": 97.9685},
]

# These are the non-API files needed by AT2 plus the boundary used for spatialoutputs.  
# The relative paths deliberately match the project's raw inventory.
STATIC_SOURCES = []

for year_be in [2567]:
    STATIC_SOURCES.append({
        "relative_path": f"DDC/smog_odpc1_{year_be}.csv",
        "source": "Department of Disease Control",
        "dataset": "DDC respiratory disease records",
    })

for month in range(1, 13):
    STATIC_SOURCES.append({
        "relative_path": f"DDC/smog_odpc1_2568-{month:02d}.csv",
        "source": "Department of Disease Control",
        "dataset": "DDC respiratory disease records",
    })

STATIC_SOURCES.extend([
    {
        "relative_path": "DDC/datadict.csv",
        "source": "Department of Disease Control",
        "dataset": "DDC data dictionary",
    },
    {
        "relative_path": "PROVIDER/health_office.csv",
        "source": "Ministry of Public Health",
        "dataset": "Healthcare provider master",
    },
    {
        "relative_path": "PROVIDER/ha_aod_001-2.csv",
        "source": "Provider export; source organization to verify",
        "dataset": "Hospital accreditation and capacity",
    },
])

for year_suffix in [65, 66, 67, 68]:
    STATIC_SOURCES.append({
        "relative_path": f"POPULATIONS/stat_c{year_suffix}.xls",
        "source": "Department of Provincial Administration",
        "dataset": "Registered population",
    })

STATIC_SOURCES.append({
    "relative_path": "BOUNDARY/tha_admbnda_adm1_rtsd_20190221.zip",
    "source": "Open Development Mekong distribution; source data attributed to RTSD",
    "dataset": "Thailand administrative level 1 boundary",
})

# These tables are not original external sources and are therefore optional.
# When a verified project snapshot includes them, prepare_data.py can preserve
# the exact locked model input bytes instead of rebuilding floating-point text.
OPTIONAL_FROZEN_SOURCES = [
    {
        "relative_path": "frozen/AT1_ONE_DAY_AHEAD_FULL.csv",
        "source": "Verified analytical snapshot",
        "dataset": "Locked full one-day-ahead AT1 frame",
    },
    {
        "relative_path": "frozen/AT1_ONE_DAY_AHEAD_MODEL_READY.csv",
        "source": "Verified analytical snapshot",
        "dataset": "Locked model-ready AT1 frame",
    },
    {
        "relative_path": "frozen/AT_PM25_PROVINCE_DAY.csv",
        "source": "Verified analytical snapshot",
        "dataset": "Locked province-day analytical table",
    },
    {
        "relative_path": "frozen/AT_RESPIRATORY_CAPACITY_PROVINCE_MONTH.csv",
        "source": "Verified analytical snapshot",
        "dataset": "Locked exploratory AT2 table",
    },
]

# Stable public downloads verified from the publishers' catalogue/download pages.  
# A fallback folder or per-file URL override can still be used if a publisher temporarily changes one of these links.
DDC_DATASET_ID = "543d7d8c-47f1-4b88-906a-f56f7033ebac"
DDC_RESOURCE_IDS = {
    "DDC/smog_odpc1_2567.csv": "2d4a0ef3-ae2e-4af6-8af3-04e3b2ad152e",
    "DDC/smog_odpc1_2568-01.csv": "c47d11c8-1ee4-4bee-afda-62839a158097",
    "DDC/smog_odpc1_2568-02.csv": "badff9c7-8850-468f-8b8a-b3a1e15d48fa",
    "DDC/smog_odpc1_2568-03.csv": "d30b6a6d-e6d1-4dfd-a305-71ae79245091",
    "DDC/smog_odpc1_2568-04.csv": "96d648e2-4356-423d-9b48-6f735891794d",
    "DDC/smog_odpc1_2568-05.csv": "37ad257a-c056-4366-bb02-9a0d4743c414",
    "DDC/smog_odpc1_2568-06.csv": "8d953a0f-673f-45b9-a9f6-5e80dce2eed9",
    "DDC/smog_odpc1_2568-07.csv": "b1d1acda-0448-4c69-9388-7394f2640a77",
    "DDC/smog_odpc1_2568-08.csv": "bed9356f-4dd1-4c68-b28a-6c25173e67e8",
    "DDC/smog_odpc1_2568-09.csv": "6f2de1a1-f13d-4a84-a69d-f446d138b4fb",
    "DDC/smog_odpc1_2568-10.csv": "cc508005-8e18-4971-8253-9f91637c6d31",
    "DDC/smog_odpc1_2568-11.csv": "816232eb-efed-42d4-836f-458ec19498f7",
    "DDC/smog_odpc1_2568-12.csv": "dfd175a4-6cde-4202-b948-d23f1daba6cd",
}

STATIC_DIRECT_URLS = {
    relative_path: (
        "https://opendata.ddc.moph.go.th/dataset/"
        f"{DDC_DATASET_ID}/resource/{resource_id}/download/"
        f"{PurePosixPath(relative_path).name}"
    )
    for relative_path, resource_id in DDC_RESOURCE_IDS.items()
}
STATIC_DIRECT_URLS.update({
    "DDC/datadict.csv": (
        "https://opendata.ddc.moph.go.th/dataset/"
        f"{DDC_DATASET_ID}/resource/7aa55374-d58b-4f17-8b9b-06143fec44c4/"
        "download/datadict.csv"
    ),
    "PROVIDER/ha_aod_001-2.csv": (
        "https://data.ha.or.th/dataset/"
        "5e44de52-ca41-4d1d-bd6a-0a0fd56d7b75/resource/"
        "4e20e752-25f8-468c-b155-33be7aecc0d4/download/ha_aod_001-2.csv"
    ),
    "PROVIDER/health_office.csv": (
        "https://hcode.moph.go.th/static/hcode/csv/health_office.csv"
    ),
    "POPULATIONS/stat_c65.xls": "https://stat.bora.dopa.go.th/new_stat/file/65/stat_c65.xls",
    "POPULATIONS/stat_c66.xls": "https://stat.bora.dopa.go.th/new_stat/file/66/stat_c66.xls",
    "POPULATIONS/stat_c67.xls": "https://stat.bora.dopa.go.th/new_stat/file/67/stat_c67.xls",
    "POPULATIONS/stat_c68.xls": "https://stat.bora.dopa.go.th/new_stat/file/68/stat_c68.xls",
    "BOUNDARY/tha_admbnda_adm1_rtsd_20190221.zip": (
        "https://data.opendevelopmentmekong.net/th/dataset/"
        "8f3fa1b8-cb5c-48c8-9fd7-d3c213ea23db/resource/"
        "1559cee4-fedc-4330-be9c-d8cf3dd75015/download/"
        "tha_admbnda_adm1_rtsd_20190221.zip"
    ),
})

STATIC_DOWNLOAD_PAGES = {
    "POPULATIONS/stat_c66.xls": "https://stat.bora.dopa.go.th/new_stat/webPage/statByYear.php",
}

# Some publisher download links work only after the catalogue page has first established a browser session.  
# The preflight URL supplies the cookies and Referer that a normal click in the browser would carry to the file request.
STATIC_PREFLIGHT_URLS = {
    "POPULATIONS/stat_c66.xls": (
        "https://stat.bora.dopa.go.th/new_stat/webPage/statByYear.php"
    ),
}
# Additional filename aliases can be added here if a publisher changes the filename while keeping the same dataset.  
# No aliases are currently needed.
STATIC_FILENAME_ALIASES = {}

for item in STATIC_SOURCES:
    relative_path = item["relative_path"]
    item["download_url"] = STATIC_DIRECT_URLS.get(relative_path, "")
    item["download_page"] = STATIC_DOWNLOAD_PAGES.get(relative_path, "")
    item["preflight_url"] = STATIC_PREFLIGHT_URLS.get(relative_path, "")
    item["filename_aliases"] = STATIC_FILENAME_ALIASES.get(relative_path, [])

# Correct the source attribution now that the official catalogue was located.
for item in STATIC_SOURCES:
    if item["relative_path"] == "PROVIDER/ha_aod_001-2.csv":
        item["source"] = "Healthcare Accreditation Institute (Public Organization)"

MANIFEST_COLUMNS = [
    "relative_path",
    "source",
    "request_url",
    "request_parameters",
    "row_count",
    "size_bytes",
    "sha256",
    "retrieved_at_utc",
    "status",
]

# Request metadata are kept in memory and written to the raw-file manifest.
# API keys are deliberately excluded.
REQUEST_RECORDS = {}


# =============================================================================
# 2. SIMPLE HELPER FUNCTIONS
# =============================================================================

def utc_now():
    """Return the current UTC time for a newly fetched file."""
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    """Return SHA256 for one file without loading it all into memory."""
    hasher = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            block = file.read(1024 * 1024)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def file_facts(path):
    """Return SHA256 and CSV row count in one pass through a file."""
    hasher = hashlib.sha256()
    newline_count = 0
    last_byte = b""
    count_rows = path.suffix.lower() in [".csv", ".tsv"]

    with open(path, "rb") as file:
        while True:
            block = file.read(1024 * 1024)
            if not block:
                break
            hasher.update(block)
            if count_rows:
                newline_count += block.count(b"\n")
                last_byte = block[-1:]

    row_count = ""
    if count_rows:
        physical_lines = newline_count + (1 if last_byte and last_byte != b"\n" else 0)
        row_count = max(0, physical_lines - 1)

    return hasher.hexdigest(), row_count


def record_request(path, source, request_url, parameters, row_count):
    """Remember non-secret request details for the final manifest."""
    relative_path = str(path.relative_to(PROJECT_DIR))
    safe_url = redact_url(request_url)
    if FIRMS_KEY:
        safe_url = safe_url.replace(FIRMS_KEY, "{NASA_FIRMS_MAP_KEY}")
    REQUEST_RECORDS[relative_path] = {
        "source": source,
        "request_url": safe_url,
        "request_parameters": json.dumps(
            parameters, ensure_ascii=False, sort_keys=True
        ),
        "row_count": row_count,
    }


def redact_url(url):
    """Remove credentials and query values before a URL enters the manifest."""
    if not url:
        return ""
    parsed = urllib.parse.urlsplit(str(url))
    host = parsed.hostname or ""
    if parsed.port:
        host += f":{parsed.port}"
    query_names = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    safe_query = urllib.parse.urlencode([
        (name, "{REDACTED}") for name, _ in query_names
    ])
    return urllib.parse.urlunsplit(
        (parsed.scheme, host, parsed.path, safe_query, "")
    )


def request_bytes(url, parameters=None, headers=None, attempts=6):
    """Download bytes with retry for rate limits and temporary server errors."""
    if not FETCH_LIVE:
        raise RuntimeError(
            "A required raw API response is missing. Run fetch_data.py with "
            "FETCH_LIVE=1 to create a new snapshot."
        )
    parameters = parameters or {}
    headers = headers or {}
    full_url = url
    if parameters:
        full_url += "?" + urllib.parse.urlencode(parameters)

    last_error = None
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(full_url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read(), full_url
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code == 429 or error.code >= 500:
                wait = error.headers.get("Retry-After")
                wait_seconds = float(wait) if wait else min(60, 2 ** attempt)
                time.sleep(wait_seconds)
                continue
            try:
                response_detail = error.read().decode(
                    TEXT_ENCODING, errors="replace"
                ).strip()
            except Exception:
                response_detail = ""
            safe_url = redact_url(full_url)
            if FIRMS_KEY:
                safe_url = safe_url.replace(
                    FIRMS_KEY, "{NASA_FIRMS_MAP_KEY}"
                )
            detail_suffix = (
                f" Response: {response_detail[:500]}"
                if response_detail else ""
            )
            raise RuntimeError(
                f"HTTP {error.code} from {safe_url}.{detail_suffix}"
            ) from error
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionResetError,
            BrokenPipeError,
            http.client.IncompleteRead,
            http.client.RemoteDisconnected,
            http.client.BadStatusLine,
        ) as error:
            last_error = error
            if attempt < attempts:
                time.sleep(min(60, 2 ** attempt))
                continue
            raise

    raise RuntimeError(f"Request failed: {last_error}")


def save_raw_response(path, content):
    """Save the exact response bytes once; never overwrite a frozen response."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return False

    temporary = path.with_suffix(path.suffix + ".part")
    with open(temporary, "wb") as file:
        file.write(content)
    temporary.replace(path)
    return True


def download_file(url, path, headers=None, attempts=6, preflight_url=""):
    """Stream one potentially large file to disk with retries."""
    if path.exists():
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    headers = headers or {
        "User-Agent": "Mozilla/5.0 (compatible; ProjectDataFetcher/1.0)",
        "Accept": "*/*",
    }
    last_error = None

    for attempt in range(1, attempts + 1):
        try:
            opener = urllib.request.build_opener(
                urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
            )
            request_headers = dict(headers)
            if preflight_url:
                preflight_request = urllib.request.Request(
                    preflight_url,
                    headers=request_headers,
                )
                with opener.open(preflight_request, timeout=180) as response:
                    response.read(1)
                request_headers["Referer"] = preflight_url

            request = urllib.request.Request(url, headers=request_headers)
            with opener.open(request, timeout=180) as response:
                with open(temporary, "wb") as file:
                    shutil.copyfileobj(response, file, length=1024 * 1024)
            if temporary.stat().st_size == 0:
                raise RuntimeError("Downloaded file is empty")
            temporary.replace(path)
            return True
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code != 429 and error.code < 500:
                break
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = error

        if attempt < attempts:
            time.sleep(min(60, 2 ** attempt))

    if temporary.exists():
        temporary.unlink()
    raise RuntimeError(f"Download failed for {redact_url(url)}: {last_error}")


def is_population_html_xls(path):
    """Recognize DOPA's real population table distributed with a .xls suffix."""
    if not re.fullmatch(r"stat_c\d{2}\.xls", path.name, flags=re.I):
        return False
    try:
        text = path.read_text(encoding=TEXT_ENCODING).lstrip().lower()
    except UnicodeDecodeError:
        return False
    return (
        text.startswith("<table")
        and "</table>" in text
        and "รหัสจังหวัด" in text
        and "จำนวนประชากรทั้งหมด" in text
    )


def validate_static_file(path):
    """Reject empty/error payloads while accepting official source formats."""
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty source file: {path}")

    with open(path, "rb") as file:
        prefix = file.read(4096)
    lowered = prefix.lstrip().lower()
    suffix = path.suffix.lower()
    population_html_xls = suffix == ".xls" and is_population_html_xls(path)
    if (
        lowered.startswith((b"<!doctype html", b"<html", b"<head", b"<body"))
        and not population_html_xls
    ):
        raise ValueError(f"Expected data but received an HTML page: {path.name}")

    if suffix in [".xlsx", ".zip"] and not zipfile.is_zipfile(path):
        raise ValueError(f"Expected a ZIP-based {suffix} file: {path.name}")
    if (
        suffix == ".xls"
        and not prefix.startswith(bytes.fromhex("D0CF11E0A1B11AE1"))
        and not population_html_xls
    ):
        raise ValueError(
            f"Expected a binary Excel file or the official DOPA HTML table: {path.name}"
        )


def build_download_file_index(download_dir):
    """Index local downloads by case-insensitive filename."""
    if not download_dir:
        return {}

    root = Path(download_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"DOWNLOAD_DIR is not a directory: {root}")

    index = {}
    for path in root.rglob("*"):
        if path.is_file():
            index.setdefault(path.name.casefold(), []).append(path)
    return index


def import_downloaded_source(item, target, download_index):
    """Copy one previously downloaded export into its deterministic raw path."""
    names = [PurePosixPath(item["relative_path"]).name]
    names.extend(item.get("filename_aliases", []))
    matches = []
    for name in names:
        matches.extend(download_index.get(name.casefold(), []))
    matches = list(dict.fromkeys(matches))
    if not matches:
        return False
    if len(matches) > 1:
        choices = "\n  - " + "\n  - ".join(str(path) for path in matches)
        raise RuntimeError(
            f"More than one local file matches {item['relative_path']}:" + choices
        )

    source = matches[0]
    validate_static_file(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    shutil.copy2(source, temporary)
    validate_static_file(temporary)
    temporary.replace(target)
    record_request(
        target,
        item["source"],
        f"local-download:{source.name}",
        {
            "dataset": item["dataset"],
            "download_page": item.get("download_page", ""),
            "original_filename": source.name,
        },
        "",
    )
    return True


def load_source_url_mapping():
    """Read optional per-file URLs from JSON text or a JSON file path."""
    if not SOURCE_URLS_SETTING:
        return {}

    possible_file = None
    if not SOURCE_URLS_SETTING.lstrip().startswith("{"):
        possible_file = Path(SOURCE_URLS_SETTING)
    if possible_file is not None and possible_file.is_file():
        with open(possible_file, "r", encoding=TEXT_ENCODING) as file:
            mapping = json.load(file)
    else:
        mapping = json.loads(SOURCE_URLS_SETTING)

    if not isinstance(mapping, dict):
        raise ValueError("SOURCE_URLS_JSON must contain a JSON object")
    return {
        str(key).replace("\\", "/").lstrip("/"): str(value).strip()
        for key, value in mapping.items()
        if str(value).strip()
    }


def normalized_bundle_member(name):
    """Return a safe raw-relative path for one ZIP member."""
    text = str(name).replace("\\", "/").lstrip("/")
    for prefix in ["data/raw/", "raw/"]:
        if text.startswith(prefix):
            text = text[len(prefix):]
    path = PurePosixPath(text)
    if not text or path.is_absolute() or ".." in path.parts:
        return None
    return path.as_posix()


def extract_static_bundle(bundle_path, bundle_url):
    """Extract expected static raw files and optional frozen model snapshots."""
    if not zipfile.is_zipfile(bundle_path):
        raise ValueError("RAW_BUNDLE_URL did not return a valid ZIP file")

    expected = {
        item["relative_path"]: item
        for item in STATIC_SOURCES + OPTIONAL_FROZEN_SOURCES
    }
    extracted = 0

    with zipfile.ZipFile(bundle_path) as archive:
        for member in archive.infolist():
            relative_path = normalized_bundle_member(member.filename)
            if member.is_dir() or relative_path not in expected:
                continue

            target = RAW_DIR / relative_path
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(target.suffix + ".part")
                with archive.open(member) as source, open(temporary, "wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                if temporary.stat().st_size == 0:
                    temporary.unlink()
                    raise RuntimeError(f"Empty bundle member: {relative_path}")
                temporary.replace(target)
                extracted += 1

            item = expected[relative_path]
            record_request(
                target,
                item["source"],
                bundle_url,
                {"bundle_member": relative_path, "dataset": item["dataset"]},
                "",
            )

    return extracted


def fetch_static_sources():
    """Acquire non-API files from local exports, public URLs or overrides."""
    url_mapping = load_source_url_mapping()
    download_index = build_download_file_index(DOWNLOAD_DIR_SETTING)

    if RAW_BUNDLE_URL:
        bundle_file = RAW_DIR / "_bundle" / "project_static_raw_sources.zip"
        was_downloaded = download_file(RAW_BUNDLE_URL, bundle_file)
        record_request(
            bundle_file,
            "Frozen raw source bundle",
            RAW_BUNDLE_URL,
            {"purpose": "reproducible acquisition of static source exports"},
            "",
        )
        extracted = extract_static_bundle(bundle_file, RAW_BUNDLE_URL)
        action = "Downloaded" if was_downloaded else "Reused"
        print(f"  {action} raw bundle; extracted {extracted:,} missing files")

    downloaded = 0
    failures = []
    for item in STATIC_SOURCES:
        relative_path = item["relative_path"]
        target = RAW_DIR / relative_path
        if target.exists():
            validate_static_file(target)
            continue

        if import_downloaded_source(item, target, download_index):
            downloaded += 1
            print(f"  Imported downloaded source: {relative_path}")
            continue

        override_url = (
            url_mapping.get(relative_path)
            or url_mapping.get(f"data/raw/{relative_path}")
        )
        url = override_url or item.get("download_url", "")
        if not url:
            continue

        try:
            download_file(
                url,
                target,
                preflight_url=item.get("preflight_url", ""),
            )
            validate_static_file(target)
        except Exception as error:
            if target.exists():
                target.unlink()
            failures.append((relative_path, str(error)))
            print(f"  Could not fetch {relative_path}: {error}")
            continue
        record_request(
            target,
            item["source"],
            url,
            {
                "dataset": item["dataset"],
                "url_kind": "override" if override_url else "publisher_direct",
            },
            "",
        )
        downloaded += 1
        print(f"  Fetched static source: {relative_path}")

    missing = [
        item["relative_path"]
        for item in STATIC_SOURCES
        if not (RAW_DIR / item["relative_path"]).is_file()
    ]
    if missing:
        items_by_path = {item["relative_path"]: item for item in STATIC_SOURCES}
        detail_rows = []
        for relative_path in missing:
            page = items_by_path[relative_path].get("download_page", "")
            suffix = f" (export page: {page})" if page else ""
            detail_rows.append(relative_path + suffix)
        details = "\n  - " + "\n  - ".join(detail_rows)
        failure_text = ""
        if failures:
            failure_text = "\nDownload errors:" + "".join(
                f"\n  - {path}: {message}" for path, message in failures
            )
        raise RuntimeError(
            "Static raw acquisition is incomplete. Put existing exports in "
            "DOWNLOAD_DIR, or supply RAW_BUNDLE_URL / "
            "SOURCE_URLS_JSON for these files:" + details + failure_text
        )
    return len(STATIC_SOURCES), downloaded


def read_json(path):
    """Read one raw JSON response."""
    with open(path, "r", encoding=TEXT_ENCODING) as file:
        return json.load(file)


def haversine_distance(lat1, lon1, lat2, lon2):
    """Return distance in kilometres between two latitude/longitude points."""
    lat1 = np.radians(float(lat1))
    lon1 = np.radians(float(lon1))
    lat2 = np.radians(float(lat2))
    lon2 = np.radians(float(lon2))
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    value = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
    )
    return 6371.0088 * 2 * np.arctan2(np.sqrt(value), np.sqrt(1 - value))


def nearest_province(latitude, longitude):
    """Assign a discovered OpenAQ location to the nearest study centroid."""
    distances = [
        haversine_distance(
            latitude,
            longitude,
            province["latitude"],
            province["longitude"],
        )
        for province in PROVINCES
    ]
    return PROVINCES[int(np.argmin(distances))]


def list_raw_files():
    """List data files while excluding the manifest and unfinished downloads."""
    files = []
    if RAW_DIR.exists():
        for path in RAW_DIR.rglob("*"):
            if not path.is_file():
                continue
            if path.name == "fetch_manifest.csv" or path.name.endswith(".part"):
                continue
            files.append(path)
    return sorted(files)


def write_adapter(frame, filename):
    """Write a temporary parsed table only when prepare_data.py requests it."""
    if ADAPTER_DIR is None:
        return
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        ADAPTER_DIR / filename,
        index=False,
        encoding=CSV_ENCODING,
    )


# =============================================================================
# 3. OPEN-METEO HISTORICAL WEATHER
# =============================================================================

def fetch_weather():
    """Fetch exact Open-Meteo JSON responses and build a daily source table."""
    raw_folder = RAW_DIR / "api" / "open_meteo_weather"
    rows = []

    for province in PROVINCES:
        for year in range(START_YEAR, END_YEAR + 1):
            raw_file = raw_folder / f"TH{province['province_key']}_{year}.json"

            if not raw_file.exists():
                parameters = {
                    "latitude": province["latitude"],
                    "longitude": province["longitude"],
                    "start_date": f"{year}-01-01",
                    "end_date": f"{year}-12-31",
                    "daily": ",".join(WEATHER_VARIABLES),
                    "timezone": "Asia/Bangkok",
                    "models": "era5",
                }
                content, full_url = request_bytes(WEATHER_URL, parameters)
                save_raw_response(raw_file, content)
                print(f"  Fetched weather: TH{province['province_key']} {year}")
                time.sleep(0.2)

            payload = read_json(raw_file)
            daily = payload.get("daily", {})
            dates = daily.get("time", [])
            parameters = {
                "latitude": province["latitude"],
                "longitude": province["longitude"],
                "start_date": f"{year}-01-01",
                "end_date": f"{year}-12-31",
                "daily": ",".join(WEATHER_VARIABLES),
                "timezone": "Asia/Bangkok",
                "models": "era5",
            }
            record_request(
                raw_file,
                "Open-Meteo Historical Weather API",
                WEATHER_URL,
                parameters,
                len(dates),
            )

            for position, date in enumerate(dates):
                row = {
                    "province_key": province["province_key"],
                    "province_name_th": province["province_name_th"],
                    "province_name_en": province["province_name_en"],
                    "analysis_date_bkk": date,
                    "source": "Open-Meteo Historical Weather API / ERA5",
                    "raw_file": str(raw_file.relative_to(PROJECT_DIR)),
                }
                for variable in WEATHER_VARIABLES:
                    values = daily.get(variable, [])
                    row[variable] = values[position] if position < len(values) else None
                rows.append(row)

    table = pd.DataFrame(rows).sort_values(["province_key", "analysis_date_bkk"])
    write_adapter(table, "DAILY_WEATHER_F.csv")
    return len(table)


def fetch_modeled_pm25():
    """Fetch optional CAMS Global PM2.5 used only as a descriptive benchmark."""
    raw_folder = RAW_DIR / "api" / "open_meteo_modeled_pm25"
    rows = []

    for province in PROVINCES:
        for year in range(2022, END_YEAR + 1):
            start = max(pd.Timestamp("2022-08-01"), pd.Timestamp(f"{year}-01-01"))
            end = min(pd.Timestamp("2025-12-31"), pd.Timestamp(f"{year}-12-31"))
            if start > end:
                continue

            raw_file = raw_folder / f"TH{province['province_key']}_{year}.json"
            parameters = {
                "latitude": province["latitude"],
                "longitude": province["longitude"],
                "hourly": "pm2_5",
                "start_date": f"{start:%Y-%m-%d}",
                "end_date": f"{end:%Y-%m-%d}",
                "timezone": "Asia/Bangkok",
                "domains": "cams_global",
            }

            if not raw_file.exists():
                content, full_url = request_bytes(MODELED_PM25_URL, parameters)
                save_raw_response(raw_file, content)
                print(f"  Fetched modeled PM2.5: TH{province['province_key']} {year}")
                time.sleep(0.2)

            payload = read_json(raw_file)
            hourly = payload.get("hourly", {})
            times = hourly.get("time", [])
            values = hourly.get("pm2_5", [])
            record_request(
                raw_file,
                "Open-Meteo Air Quality API / CAMS Global",
                MODELED_PM25_URL,
                parameters,
                len(times),
            )

            for position, datetime_value in enumerate(times):
                rows.append({
                    "province_key": province["province_key"],
                    "province_name_th": province["province_name_th"],
                    "province_name_en": province["province_name_en"],
                    "analysis_date_bkk": str(datetime_value)[:10],
                    "datetime_bkk": datetime_value,
                    "pm25_modeled": values[position] if position < len(values) else None,
                    "source": "Open-Meteo Air Quality API / CAMS Global",
                    "raw_file": str(raw_file.relative_to(PROJECT_DIR)),
                })

    table = pd.DataFrame(rows).sort_values(["province_key", "datetime_bkk"])
    write_adapter(table, "MODELED_PM25_F.csv")
    return len(table)


# =============================================================================
# 4. OPENAQ LOCATIONS, PM2.5 SENSORS, AND DAILY VALUES
# =============================================================================

def openaq_pages(endpoint, parameters, output_stem):
    """Fetch every OpenAQ page and return all result objects."""
    all_results = []
    page = 1

    while True:
        page_parameters = dict(parameters)
        page_parameters["limit"] = 1000
        page_parameters["page"] = page
        raw_file = RAW_DIR / "api" / "openaq" / f"{output_stem}_page{page}.json"

        if not raw_file.exists():
            content, full_url = request_bytes(
                OPENAQ_BASE + endpoint,
                page_parameters,
                headers={"X-API-Key": OPENAQ_KEY},
            )
            save_raw_response(raw_file, content)
            time.sleep(1.05)

        payload = read_json(raw_file)
        results = payload.get("results", [])
        all_results.extend(results)
        record_request(
            raw_file,
            "OpenAQ API v3",
            OPENAQ_BASE + endpoint,
            page_parameters,
            len(results),
        )

        found = payload.get("meta", {}).get("found")
        try:
            found = int(found)
        except (TypeError, ValueError):
            found = None

        if not results or len(results) < 1000:
            break
        if found is not None and page * 1000 >= found:
            break
        page += 1

    return all_results


def discover_openaq_sensors():
    """Discover Thai PM2.5 sensors within 100 km of a study centroid.

    OpenAQ limits a point-radius query to 25 km. Querying all Thai locations
    once and applying the original 100 km study rule locally avoids that API
    limit without reducing the intended geographic coverage.
    """
    locations = {}
    sensors = {}

    results = openaq_pages(
        "/locations",
        {
            "iso": "TH",
            "order_by": "id",
            "sort_order": "asc",
        },
        "locations_TH_all",
    )

    for item in results:
        coordinates = item.get("coordinates") or {}
        latitude = coordinates.get("latitude")
        longitude = coordinates.get("longitude")
        if latitude is None or longitude is None:
            continue

        province_distances = [
            (
                haversine_distance(
                    latitude,
                    longitude,
                    province["latitude"],
                    province["longitude"],
                ),
                province,
            )
            for province in PROVINCES
        ]
        distance_km, assigned = min(province_distances, key=lambda item: item[0])
        if distance_km > 100:
            continue

        location_id = item.get("id")
        if location_id is None:
            continue
        locations[location_id] = {
            "location_id": location_id,
            "location_name": item.get("name"),
            "province_key": assigned["province_key"],
            "province_name_th": assigned["province_name_th"],
            "province_name_en": assigned["province_name_en"],
            "distance_to_province_centroid_km": round(distance_km, 3),
            "latitude": latitude,
            "longitude": longitude,
            "timezone": item.get("timezone"),
        }

        for sensor in item.get("sensors", []) or []:
            parameter = sensor.get("parameter") or {}
            name = str(parameter.get("name", "")).lower().replace(".", "")
            if name not in ["pm25", "pm2_5"]:
                continue
            sensor_id = sensor.get("id")
            if sensor_id is None:
                continue
            sensors[sensor_id] = {
                **locations[location_id],
                "sensor_id": sensor_id,
                "sensor_name": sensor.get("name"),
                "unit": parameter.get("units", "µg/m³"),
            }

    return pd.DataFrame(locations.values()), pd.DataFrame(sensors.values())


def parse_openaq_day(item, sensor):
    """Convert one OpenAQ day result to one transparent source-table row."""
    period = item.get("period") or {}
    datetime_from = period.get("datetimeFrom") or {}
    coverage = item.get("coverage") or {}
    summary = item.get("summary") or {}
    parameter = item.get("parameter") or {}

    datetime_utc = datetime_from.get("utc")
    datetime_local = datetime_from.get("local")
    value = item.get("value")
    if value is None:
        value = summary.get("avg")
    percent_coverage = coverage.get("percentCoverage")
    try:
        value_number = float(value)
        coverage_number = float(percent_coverage)
        primary_eligible = (
            0 <= value_number <= 500 and coverage_number >= 75
        )
    except (TypeError, ValueError):
        primary_eligible = False

    return {
        "province_key": sensor["province_key"],
        "province_name_th": sensor["province_name_th"],
        "province_name_en": sensor["province_name_en"],
        "location_id": sensor["location_id"],
        "location_name": sensor["location_name"],
        "sensor_id": sensor["sensor_id"],
        "sensor_name": sensor["sensor_name"],
        "latitude": sensor["latitude"],
        "longitude": sensor["longitude"],
        "datetime_utc": datetime_utc,
        "datetime_local": datetime_local,
        "analysis_date_bkk": str(datetime_local or datetime_utc)[:10],
        "pm25": value,
        "unit": parameter.get("units", sensor["unit"]),
        "percent_complete": coverage.get("percentComplete"),
        "percent_coverage": percent_coverage,
        "primary_analysis_eligible": primary_eligible,
        "source": f"OpenAQ API v3 / sensors/{sensor['sensor_id']}/days",
    }


def fetch_openaq():
    """Fetch daily observed PM2.5 from every discovered sensor."""
    locations, sensors = discover_openaq_sensors()
    if sensors.empty:
        raise RuntimeError("OpenAQ discovery returned no PM2.5 sensors")

    rows = []
    for sensor in sensors.sort_values("sensor_id").to_dict("records"):
        sensor_id = int(sensor["sensor_id"])
        for year in range(START_YEAR, END_YEAR + 1):
            results = openaq_pages(
                f"/sensors/{sensor_id}/days",
                {
                    "date_from": f"{year}-01-01",
                    "date_to": f"{year}-12-31",
                },
                f"sensor_{sensor_id}_{year}",
            )
            rows.extend(parse_openaq_day(item, sensor) for item in results)
            print(f"  OpenAQ sensor {sensor_id}, {year}: {len(results):,} days")

    write_adapter(
        locations.sort_values("location_id"),
        "OPENAQ_LOCATION_D.csv",
    )
    observed = pd.DataFrame(rows).sort_values(["sensor_id", "analysis_date_bkk"])
    write_adapter(observed, "OBSERVED_PM25_F.csv")
    return len(observed), len(sensors)


# =============================================================================
# 5. NASA FIRMS VIIRS S-NPP STANDARD PROCESSING
# =============================================================================

def five_day_periods(start_date, end_date):
    """Yield non-overlapping periods of at most five days."""
    current = start_date
    while current <= end_date:
        period_end = min(current + pd.Timedelta(days=4), end_date)
        day_count = int((period_end - current).days + 1)
        yield current, period_end, day_count
        current = period_end + pd.Timedelta(days=1)


def geojson_geometries(payload):
    """Return Polygon/MultiPolygon geometries from a GeoJSON object."""
    object_type = payload.get("type")
    if object_type == "FeatureCollection":
        geometries = [
            feature.get("geometry")
            for feature in payload.get("features", [])
            if feature.get("geometry")
        ]
    elif object_type == "Feature":
        geometries = [payload.get("geometry")]
    else:
        geometries = [payload]

    geometries = [item for item in geometries if item]
    if not geometries:
        raise ValueError("Country boundary GeoJSON contains no geometry")
    return geometries


def compile_country_geometry(payload):
    """Compile GeoJSON polygons for fast vectorized point-in-country tests."""
    compiled_polygons = []
    all_vertices = []

    for geometry in geojson_geometries(payload):
        geometry_type = geometry.get("type")
        coordinates = geometry.get("coordinates") or []
        if geometry_type == "Polygon":
            polygons = [coordinates]
        elif geometry_type == "MultiPolygon":
            polygons = coordinates
        else:
            raise ValueError(
                f"Unsupported country boundary geometry: {geometry_type}"
            )

        for polygon in polygons:
            if not polygon:
                continue
            rings = []
            for ring in polygon:
                vertices = np.asarray(ring, dtype=float)
                if vertices.ndim != 2 or vertices.shape[0] < 3:
                    continue
                vertices = vertices[:, :2]
                rings.append(GeometryPath(vertices, closed=True))
                all_vertices.append(vertices)
            if rings:
                compiled_polygons.append((rings[0], rings[1:]))

    if not compiled_polygons or not all_vertices:
        raise ValueError("Country boundary has no usable polygon rings")

    vertices = np.vstack(all_vertices)
    bounds = (
        float(vertices[:, 0].min()),
        float(vertices[:, 1].min()),
        float(vertices[:, 0].max()),
        float(vertices[:, 1].max()),
    )
    return {"polygons": compiled_polygons, "bounds": bounds}


def points_inside_country(longitudes, latitudes, compiled_geometry):
    """Return a Boolean mask for points inside a compiled country geometry."""
    points = np.column_stack([
        np.asarray(longitudes, dtype=float),
        np.asarray(latitudes, dtype=float),
    ])
    inside = np.zeros(len(points), dtype=bool)

    for outer_ring, hole_rings in compiled_geometry["polygons"]:
        polygon_inside = outer_ring.contains_points(points, radius=1e-10)
        for hole_ring in hole_rings:
            polygon_inside &= ~hole_ring.contains_points(points, radius=-1e-10)
        inside |= polygon_inside
    return inside


def load_country_geometry(country_code):
    """Fetch, store and compile one GADM ADM0 country boundary."""
    filename = GADM_COUNTRY_FILES[country_code]
    raw_file = RAW_DIR / "boundary_country" / filename
    url = f"{GADM_BASE}/{filename}"

    if not raw_file.exists():
        content, full_url = request_bytes(url)
        try:
            payload = json.loads(content.decode(TEXT_ENCODING))
            compile_country_geometry(payload)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise RuntimeError(
                f"Invalid GADM boundary response for {country_code}: {error}"
            ) from error
        save_raw_response(raw_file, content)
        print(f"  Fetched country boundary: {country_code}")
    else:
        payload = read_json(raw_file)

    compiled = compile_country_geometry(payload)
    record_request(
        raw_file,
        "GADM 4.1 ADM0 boundary",
        url,
        {"country": country_code, "administrative_level": 0},
        len(compiled["polygons"]),
    )
    return compiled


def validate_firms_csv(content, country_code, start, end):
    """Reject a FIRMS error message saved with an HTTP 200 response."""
    first_line = content.splitlines()[0].decode(
        TEXT_ENCODING, errors="replace"
    ).lstrip("\ufeff") if content else ""
    columns = {item.strip().lower() for item in first_line.split(",")}
    required = {"latitude", "longitude", "acq_date", "acq_time"}
    if not required.issubset(columns):
        preview = content[:500].decode(TEXT_ENCODING, errors="replace")
        raise RuntimeError(
            f"Unexpected NASA FIRMS response for {country_code}, "
            f"{start.date()} to {end.date()}: {preview}"
        )


def fetch_firms():
    """Fetch FIRMS Area API data and clip it to four country boundaries."""
    raw_folder = RAW_DIR / "api" / "nasa_firms"
    files = []
    total_request_rows = 0
    country_geometries = {
        country_code: load_country_geometry(country_code)
        for country_code in FIRMS_COUNTRIES
    }

    for country_code, country_name in FIRMS_COUNTRIES.items():
        geometry = country_geometries[country_code]
        west, south, east, north = geometry["bounds"]
        bounding_box = ",".join(
            f"{value:.5f}" for value in [west, south, east, north]
        )

        for start, end, day_count in five_day_periods(FIRMS_START, FIRMS_END):
            raw_file = raw_folder / country_code / (
                f"VIIRS_SNPP_SP_AREA_{country_code}_"
                f"{start:%Y%m%d}_{end:%Y%m%d}.csv"
            )
            if not raw_file.exists():
                url = (
                    f"{FIRMS_BASE}/area/csv/{FIRMS_KEY}/VIIRS_SNPP_SP/"
                    f"{bounding_box}/{day_count}/{start:%Y-%m-%d}"
                )
                content, full_url = request_bytes(url, attempts=12)
                validate_firms_csv(content, country_code, start, end)
                save_raw_response(raw_file, content)
                print(f"  Fetched FIRMS {country_code}: {start.date()} to {end.date()}")
                time.sleep(0.35)

            # Count exact CSV records without loading every column.
            try:
                with open(raw_file, "rb") as file:
                    request_rows = max(0, sum(1 for _ in file) - 1)
            except OSError:
                request_rows = ""
            if isinstance(request_rows, int):
                total_request_rows += request_rows
            record_request(
                raw_file,
                "NASA FIRMS Area API",
                (
                    f"{FIRMS_BASE}/area/csv/{{NASA_FIRMS_MAP_KEY}}/"
                    f"VIIRS_SNPP_SP/{bounding_box}/{day_count}/{start:%Y-%m-%d}"
                ),
                {
                    "source": "VIIRS_SNPP_SP",
                    "country": country_code,
                    "area_coordinates": bounding_box,
                    "day_count": day_count,
                    "start_date": f"{start:%Y-%m-%d}",
                    "spatial_filter": f"GADM 4.1 {country_code} ADM0",
                },
                request_rows,
            )
            files.append((country_code, country_name, raw_file))

    # In the normal acquisition run the exact country responses in data/raw
    # are the only persistent output. prepare_data.py sets ADAPTER_DIR when it
    # needs a temporary combined event table.
    if ADAPTER_DIR is None:
        return total_request_rows

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ADAPTER_DIR / "HOTSPOTS_F.csv"
    temporary = output_path.with_suffix(".csv.part")
    if temporary.exists():
        temporary.unlink()

    total_rows = 0
    wrote_header = False
    required_columns = {"acq_date", "acq_time", "latitude", "longitude"}
    for country_code, country_name, path in files:
        frame = pd.read_csv(path, encoding=CSV_ENCODING, low_memory=False)
        if frame.empty:
            continue
        missing = required_columns - set(frame.columns)
        if missing:
            raise ValueError(
                f"Unexpected FIRMS response in {path}: missing {sorted(missing)}"
            )

        latitude = pd.to_numeric(frame["latitude"], errors="coerce")
        longitude = pd.to_numeric(frame["longitude"], errors="coerce")
        valid_coordinates = latitude.notna() & longitude.notna()
        inside_country = np.zeros(len(frame), dtype=bool)
        if valid_coordinates.any():
            valid_positions = np.flatnonzero(valid_coordinates.to_numpy())
            inside_country[valid_positions] = points_inside_country(
                longitude.iloc[valid_positions].to_numpy(),
                latitude.iloc[valid_positions].to_numpy(),
                country_geometries[country_code],
            )
        frame = frame.loc[inside_country].copy()
        if frame.empty:
            continue

        frame["source_country_code"] = country_code
        frame["source_country"] = country_name
        frame["analysis_date_bkk"] = frame["acq_date"]
        key_columns = [
            column for column in [
            "acq_date", "acq_time", "latitude", "longitude",
            "satellite", "instrument",
            ]
            if column in frame.columns
        ]
        frame = frame.drop_duplicates(key_columns)
        frame.to_csv(
            temporary,
            mode="a",
            header=not wrote_header,
            index=False,
            encoding=CSV_ENCODING if not wrote_header else TEXT_ENCODING,
        )
        wrote_header = True
        total_rows += len(frame)

    if not wrote_header:
        raise RuntimeError("FIRMS returned no rows for the study period")
    temporary.replace(output_path)
    return total_rows


# =============================================================================
# 6. RAW MANIFEST
# =============================================================================

def count_raw_matches(patterns):
    """Count unique raw files matching one or more relative glob patterns."""
    matches = set()
    for pattern in patterns:
        matches.update(path for path in RAW_DIR.glob(pattern) if path.is_file())
    return len(matches)


def build_source_status():
    """Build an explicit completeness check for every project source group."""
    rows = []

    grouped_static = {}
    for item in STATIC_SOURCES:
        grouped_static.setdefault(
            (item["dataset"], item["source"]), []
        ).append(item["relative_path"])

    for (dataset, source), relative_paths in grouped_static.items():
        found = sum((RAW_DIR / path).is_file() for path in relative_paths)
        rows.append({
            "dataset": dataset,
            "source": source,
            "required_file_count": len(relative_paths),
            "found_file_count": found,
            "complete": found == len(relative_paths),
            "accepted_patterns": "; ".join(relative_paths),
        })

    api_groups = [
        {
            "dataset": "Historical daily weather",
            "source": "Open-Meteo Historical Weather API / ERA5",
            "required": len(PROVINCES) * (END_YEAR - START_YEAR + 1),
            "api_patterns": ["api/open_meteo_weather/*.json"],
            "api_requirements": [
                ("api/open_meteo_weather/*.json", len(PROVINCES) * (END_YEAR - START_YEAR + 1)),
            ],
            "legacy_patterns": ["OPEN-MATEO/WEATHER.csv"],
            "legacy_required": 1,
        },
        {
            "dataset": "Modeled PM2.5 benchmark",
            "source": "Open-Meteo Air Quality API / CAMS Global",
            "required": len(PROVINCES) * (END_YEAR - 2022 + 1),
            "api_patterns": ["api/open_meteo_modeled_pm25/*.json"],
            "api_requirements": [
                ("api/open_meteo_modeled_pm25/*.json", len(PROVINCES) * (END_YEAR - 2022 + 1)),
            ],
            "legacy_patterns": ["OPEN-MATEO/PM2.5.csv"],
            "legacy_required": 1,
        },
        {
            "dataset": "Observed PM2.5 and sensor locations",
            "source": "OpenAQ API v3",
            "required": 2,
            "api_patterns": ["api/openaq/*.json"],
            "api_requirements": [
                ("api/openaq/locations_TH_all_page*.json", 1),
                ("api/openaq/sensor_*_*.json", 1),
            ],
            "legacy_patterns": [
                "OPENAQ/STATION.csv",
                "OPENAQ/LOCATION.csv",
            ],
            "legacy_required": 2,
        },
    ]

    for group in api_groups:
        api_found = count_raw_matches(group["api_patterns"])
        legacy_found = count_raw_matches(group["legacy_patterns"])
        api_complete = all(
            count_raw_matches([pattern]) >= required
            for pattern, required in group["api_requirements"]
        )
        complete = api_complete
        if not FETCH_LIVE:
            complete = complete or legacy_found >= group["legacy_required"]
        rows.append({
            "dataset": group["dataset"],
            "source": group["source"],
            "required_file_count": group["required"],
            "found_file_count": api_found + legacy_found,
            "complete": complete,
            "accepted_patterns": "; ".join(
                group["api_patterns"] + group["legacy_patterns"]
            ),
        })

    boundary_count = count_raw_matches(["boundary_country/gadm41_*_0.json"])
    rows.append({
        "dataset": "Country boundaries for FIRMS clipping",
        "source": "GADM 4.1",
        "required_file_count": len(FIRMS_COUNTRIES),
        "found_file_count": boundary_count,
        "complete": boundary_count >= len(FIRMS_COUNTRIES),
        "accepted_patterns": "boundary_country/gadm41_*_0.json",
    })

    periods_per_country = sum(
        1 for _ in five_day_periods(FIRMS_START, FIRMS_END)
    )
    new_firms_count = count_raw_matches([
        "api/nasa_firms/*/VIIRS_SNPP_SP_AREA_*.csv"
    ])
    legacy_firms_count = count_raw_matches([
        "NASA_FIRMS/CSV_SUOMI-*_archive.csv",
        "NASA_FIRMS/CSV_SUOMI-*archive.csv",
    ])
    firms_complete = (
        new_firms_count >= len(FIRMS_COUNTRIES) * periods_per_country
    )
    if not FETCH_LIVE:
        firms_complete = (
            firms_complete
            or legacy_firms_count >= len(FIRMS_COUNTRIES)
        )
    rows.append({
        "dataset": "VIIRS S-NPP fire hotspots",
        "source": "NASA FIRMS",
        "required_file_count": len(FIRMS_COUNTRIES) * periods_per_country,
        "found_file_count": new_firms_count + legacy_firms_count,
        "complete": firms_complete,
        "accepted_patterns": (
            "api/nasa_firms/*/VIIRS_SNPP_SP_AREA_*.csv; "
            "NASA_FIRMS/CSV_SUOMI-*_archive.csv"
        ),
    })

    status = pd.DataFrame(rows)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    status.to_csv(
        SOURCE_STATUS_FILE,
        index=False,
        encoding=CSV_ENCODING,
    )
    return status

def write_manifest():
    """Record file paths, sizes, hashes, and stable retrieval metadata."""
    old_rows = {}
    legacy_manifest = RAW_DIR / "fetch_manifest.csv"
    existing_manifest = (
        MANIFEST_FILE if MANIFEST_FILE.exists() else legacy_manifest
    )
    if existing_manifest.exists():
        old = pd.read_csv(
            existing_manifest,
            dtype="string",
            encoding=CSV_ENCODING,
        )
        if "relative_path" in old.columns:
            old_rows = {
                row["relative_path"]: row
                for row in old.to_dict("records")
            }

    rows = []
    for path in list_raw_files():
        relative_path = str(path.relative_to(PROJECT_DIR))
        sha256, counted_rows = file_facts(path)
        old = old_rows.get(relative_path, {})
        request = REQUEST_RECORDS.get(relative_path, {})

        # Preserve the first retrieval time when an unchanged file is reused.
        old_sha256 = old.get("sha256")
        if isinstance(old_sha256, str) and old_sha256 == sha256:
            retrieved_at = old.get("retrieved_at_utc", "")
        else:
            retrieved_at = utc_now()

        rows.append({
            "relative_path": relative_path,
            "source": request.get("source", old.get("source", "frozen_project_snapshot")),
            "request_url": request.get("request_url", old.get("request_url", "")),
            "request_parameters": request.get(
                "request_parameters", old.get("request_parameters", "")
            ),
            "row_count": request.get(
                "row_count", old.get("row_count", counted_rows)
            ),
            "size_bytes": path.stat().st_size,
            "sha256": sha256,
            "retrieved_at_utc": retrieved_at,
            "status": "FROZEN_AND_VERIFIED",
        })

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    if not manifest.empty:
        manifest = manifest.sort_values("relative_path")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_FILE, index=False, encoding=CSV_ENCODING)
    return manifest


# =============================================================================
# 7. MAIN WORKFLOW
# =============================================================================

def main():
    """Run the complete acquisition check selected by ``FETCH_LIVE``.

    Frozen mode verifies existing files and preserves the original snapshot.
    Live mode downloads only missing files, prepares API source tables, and
    then performs the same manifest and completeness checks.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    existing_before = list_raw_files()

    if FETCH_LIVE:
        missing_keys = []
        if not OPENAQ_KEY:
            missing_keys.append("OPENAQ_API_KEY")
        if not FIRMS_KEY:
            missing_keys.append("NASA_FIRMS_MAP_KEY")
        if missing_keys:
            raise RuntimeError(
                "Live acquisition requires these environment variables: "
                + ", ".join(missing_keys)
            )

        print("LIVE FETCH MODE: creating or completing a new raw snapshot")

        static_count, static_downloaded = fetch_static_sources()
        print(
            f"Static source files ready: {static_count:,} "
            f"({static_downloaded:,} acquired this run)"
        )

        weather_rows = fetch_weather()
        print(f"Weather rows prepared: {weather_rows:,}")

        modeled_rows = fetch_modeled_pm25()
        print(f"Modeled PM2.5 rows prepared: {modeled_rows:,}")

        observed_rows, sensor_count = fetch_openaq()
        print(
            f"OpenAQ rows prepared: {observed_rows:,} "
            f"from {sensor_count:,} sensors"
        )

        hotspot_rows = fetch_firms()
        print(f"FIRMS raw area rows fetched/reused: {hotspot_rows:,}")
    else:
        print("FROZEN MODE: live APIs were not called.")
        if existing_before:
            print(f"Found {len(existing_before):,} frozen raw files.")
        else:
            print(
                "No frozen raw files were found. Commit the verified data/raw "
                "snapshot, or intentionally run once with FETCH_LIVE=1."
            )

    manifest = write_manifest()
    status = build_source_status()

    duplicate_paths = manifest["relative_path"].duplicated().sum()
    missing_hashes = manifest["sha256"].isna().sum()
    if duplicate_paths or missing_hashes:
        raise RuntimeError("Raw manifest validation failed")

    incomplete = status.loc[~status["complete"], "dataset"].tolist()
    if incomplete:
        print("Source acquisition is incomplete:")
        for dataset in incomplete:
            print(f"  - {dataset}")
        raise RuntimeError(
            "Not every required source is available. See "
            f"{SOURCE_STATUS_FILE}"
        )

    print("All source groups were fetched or verified successfully.")
    print(f"Raw files verified: {len(manifest):,}")
    print(f"Manifest: {MANIFEST_FILE}")
    print(f"Source status: {SOURCE_STATUS_FILE}")


if __name__ == "__main__":
    main()
