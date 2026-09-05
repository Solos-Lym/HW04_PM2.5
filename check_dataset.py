# -*- coding: utf-8 -*-
"""Check whether a processed dataset can support the proposed analysis.

This script performs a technical feasibility check. It does not decide whether
the research question is interesting or whether an association is causal.

Example for the HW04 PM2.5 project:

    python src/check_dataset.py \
        data/processed/analysis_ready/env_data_notnull.csv \
        --target observed_pm25_median_ugm3 \
        --time analysis_date \
        --fetch-script src/fetch_data.py \
        --dictionary outputs/metadata/data_dictionary/env_data_dictionary.csv

The default report is written to:

    outputs/quality/feasibility_report.md
"""

import argparse
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


CSV_ENCODINGS = ["utf-8-sig", "utf-8", "cp874", "tis-620"]

GROUP_CANDIDATES = [
    "province_key",
    "province_id",
    "province",
    "location_id",
    "station_id",
    "entity_id",
    "group_id",
    "patient_id",
    "subject_id",
]

METADATA_SUFFIXES = (
    "_status",
    "_reason",
    "_eligible",
    "_flag",
)


def parse_arguments():
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create a technical feasibility report for a CSV dataset."
    )
    parser.add_argument("dataset", help="Path to the processed CSV file")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--time", required=True, help="Time column name")
    parser.add_argument(
        "--fetch-script",
        required=True,
        help="Path to the reproducible data-acquisition script",
    )
    parser.add_argument(
        "--group",
        default=None,
        help="Optional panel/entity column, for example province_key",
    )
    parser.add_argument(
        "--dictionary",
        default=None,
        help=(
            "Optional data dictionary containing column_name and "
            "is_main_model_feature; when supplied, only those pre-specified "
            "features are evaluated"
        ),
    )
    parser.add_argument(
        "--output",
        default="outputs/quality/feasibility_report.md",
        help="Output Markdown report path",
    )
    return parser.parse_args()


def read_csv_with_fallback(path):
    """Read a CSV while supporting the encodings used in the project."""
    last_error = None
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False), encoding
        except UnicodeDecodeError as error:
            last_error = error

    raise UnicodeError(
        f"Could not decode {path} with: {', '.join(CSV_ENCODINGS)}"
    ) from last_error


def read_feature_dictionary(path, frame, checks):
    """Read the pre-specified model feature contract from a data dictionary."""
    if path is None:
        return None
    if not path.is_file():
        add_check(
            checks,
            "FAIL",
            "Feature dictionary",
            f"Not found: {path}",
            "Run prepare_data.py or correct the --dictionary path.",
        )
        return None

    dictionary, encoding = read_csv_with_fallback(path)
    required = {"column_name", "is_main_model_feature"}
    missing_columns = sorted(required - set(dictionary.columns))
    if missing_columns:
        add_check(
            checks,
            "FAIL",
            "Feature dictionary",
            "Missing columns: " + ", ".join(missing_columns),
            "The dictionary must identify the pre-specified main model features.",
        )
        return None

    selected = dictionary["is_main_model_feature"].astype(
        "string"
    ).str.strip().str.lower().isin({"true", "1", "yes", "y"})
    features = dictionary.loc[selected, "column_name"].astype(str).tolist()
    duplicate_features = sorted(
        pd.Series(features)[pd.Series(features).duplicated()].unique().tolist()
    )
    unavailable = [column for column in features if column not in frame.columns]

    if not features:
        status = "FAIL"
        interpretation = "No main model feature is marked in the dictionary."
    elif duplicate_features:
        status = "FAIL"
        interpretation = "Main model feature names must be unique."
    elif unavailable:
        status = "FAIL"
        interpretation = "Every pre-specified feature must exist in the dataset."
    else:
        status = "PASS"
        interpretation = (
            "Predictor checks will use only the pre-specified main model features."
        )

    evidence = f"{len(features):,} main features; loaded with {encoding}"
    if duplicate_features:
        evidence += "; duplicates: " + ", ".join(duplicate_features[:8])
    if unavailable:
        evidence += "; absent from dataset: " + ", ".join(unavailable[:8])
    add_check(checks, status, "Feature dictionary", evidence, interpretation)
    return features if status == "PASS" else None


def add_check(checks, status, check_name, evidence, interpretation):
    """Add one structured check to the report."""
    checks.append(
        {
            "status": status,
            "check": check_name,
            "evidence": str(evidence),
            "interpretation": interpretation,
        }
    )


def detect_group_column(frame, requested_group):
    """Use the requested group or detect a common panel identifier."""
    if requested_group:
        return requested_group if requested_group in frame.columns else None

    for column in GROUP_CANDIDATES:
        if column in frame.columns:
            return column
    return None


def is_metadata_column(column):
    """Identify audit columns that are not intended as model predictors."""
    name = column.lower()
    return (
        name == "data_split"
        or name.startswith("qc_")
        or name.endswith(METADATA_SUFFIXES)
        or name in {"province_name_th", "province_name_en"}
    )


def markdown_text(value):
    """Keep table cells on one line and escape Markdown pipes."""
    return str(value).replace("|", "\\|").replace("\n", " ")


def check_fetch_script(fetch_path, checks):
    """Check that data acquisition is represented by a readable Python script."""
    if not fetch_path.exists() or not fetch_path.is_file():
        add_check(
            checks,
            "FAIL",
            "Fetch script",
            f"Not found: {fetch_path}",
            "A reproducible acquisition script is required.",
        )
        return

    try:
        script_text = fetch_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        script_text = fetch_path.read_text(encoding="utf-8-sig")

    acquisition_terms = [
        "requests.get",
        "requests.post",
        "urlopen",
        "httpx",
        "api",
        "fetch",
        "download",
    ]
    raw_output_terms = ["data/raw", "raw_dir", "to_csv", "write_text", "json.dump"]

    has_acquisition_code = any(term in script_text.lower() for term in acquisition_terms)
    has_output_code = any(term in script_text.lower() for term in raw_output_terms)

    if fetch_path.suffix.lower() != ".py" or not script_text.strip():
        status = "FAIL"
        interpretation = "The supplied file is not a non-empty Python script."
    elif has_acquisition_code and has_output_code:
        status = "PASS"
        interpretation = (
            "The script contains acquisition/verification logic and code that records data."
        )
    else:
        status = "WARNING"
        interpretation = (
            "The script exists, but its acquisition and raw-output logic should be reviewed manually."
        )

    add_check(
        checks,
        status,
        "Fetch script",
        f"{fetch_path} ({len(script_text.splitlines()):,} lines)",
        interpretation,
    )


def check_dataset(
    frame,
    target,
    time_column,
    requested_group,
    checks,
    specified_features=None,
):
    """Run checks on dataset size, target, time, panel keys, and predictors."""
    row_count, column_count = frame.shape

    if row_count >= 500 and column_count >= 3:
        size_status = "PASS"
        size_note = "The dataset has enough rows for a basic modeling workflow."
    elif row_count >= 100 and column_count >= 3:
        size_status = "WARNING"
        size_note = "The dataset may support a simple model, but complexity must remain limited."
    else:
        size_status = "FAIL"
        size_note = "The dataset is too small or has too few columns for the proposed workflow."

    add_check(
        checks,
        size_status,
        "Dataset size",
        f"{row_count:,} rows and {column_count:,} columns",
        size_note,
    )

    if target not in frame.columns:
        add_check(
            checks,
            "FAIL",
            "Target column",
            f"Missing column: {target}",
            "The requested target must be present before modeling.",
        )
        numeric_target = None
    else:
        original_target = frame[target]
        numeric_target = pd.to_numeric(original_target, errors="coerce")
        invalid_target = int((original_target.notna() & numeric_target.isna()).sum())
        missing_target = int(numeric_target.isna().sum())
        missing_pct = 100 * missing_target / max(row_count, 1)

        if invalid_target > 0:
            target_status = "FAIL"
            target_note = "Some non-missing target values cannot be converted to numbers."
        elif missing_pct > 20:
            target_status = "FAIL"
            target_note = "Too many target values are missing for a dependable supervised model."
        elif missing_target > 0:
            target_status = "WARNING"
            target_note = "Rows with a missing target must be excluded before training."
        else:
            target_status = "PASS"
            target_note = "The target is numeric and complete."

        add_check(
            checks,
            target_status,
            "Target validity",
            (
                f"{missing_target:,} missing ({missing_pct:.2f}%); "
                f"{invalid_target:,} invalid numeric values"
            ),
            target_note,
        )

        valid_target = numeric_target.dropna()
        unique_target = int(valid_target.nunique())
        target_std = valid_target.std()

        if unique_target < 2 or pd.isna(target_std) or target_std == 0:
            variation_status = "FAIL"
            variation_note = "The target does not vary, so a predictive model cannot learn it."
        elif unique_target < 20:
            variation_status = "WARNING"
            variation_note = (
                "The target has few unique values; confirm whether this should be regression "
                "or classification."
            )
        else:
            variation_status = "PASS"
            variation_note = "The continuous target has usable variation for regression."

        if valid_target.empty:
            target_range = "No valid target values"
        else:
            target_range = (
                f"{unique_target:,} unique values; min={valid_target.min():.3f}; "
                f"median={valid_target.median():.3f}; max={valid_target.max():.3f}; "
                f"SD={target_std:.3f}"
            )

        add_check(
            checks,
            variation_status,
            "Target variation",
            target_range,
            variation_note,
        )

    if time_column not in frame.columns:
        add_check(
            checks,
            "FAIL",
            "Time column",
            f"Missing column: {time_column}",
            "A valid time column is required for temporal validation.",
        )
        parsed_time = None
    else:
        parsed_time = pd.to_datetime(frame[time_column], errors="coerce")
        invalid_time = int(parsed_time.isna().sum())
        invalid_pct = 100 * invalid_time / max(row_count, 1)

        if invalid_pct > 5:
            time_status = "FAIL"
            time_note = "Too many time values are missing or invalid."
        elif invalid_time > 0:
            time_status = "WARNING"
            time_note = "Invalid time rows must be corrected or excluded."
        else:
            time_status = "PASS"
            time_note = "All time values can be parsed."

        valid_time = parsed_time.dropna()
        if valid_time.empty:
            time_evidence = "No valid time values"
        else:
            time_evidence = (
                f"{invalid_time:,} invalid ({invalid_pct:.2f}%); "
                f"{valid_time.nunique():,} unique times; "
                f"{valid_time.min().date()} to {valid_time.max().date()}"
            )

        add_check(
            checks,
            time_status,
            "Time validity",
            time_evidence,
            time_note,
        )

        unique_times = int(valid_time.nunique())
        if unique_times >= 100:
            coverage_status = "PASS"
            coverage_note = "The time axis is long enough for a chronological split."
        elif unique_times >= 30:
            coverage_status = "WARNING"
            coverage_note = "Temporal validation is possible but may be unstable."
        else:
            coverage_status = "FAIL"
            coverage_note = "There are too few distinct time points for dependable temporal validation."

        add_check(
            checks,
            coverage_status,
            "Temporal coverage",
            f"{unique_times:,} distinct time points",
            coverage_note,
        )

    group_column = detect_group_column(frame, requested_group)
    if requested_group and group_column is None:
        add_check(
            checks,
            "FAIL",
            "Panel/entity column",
            f"Requested column not found: {requested_group}",
            "Use an existing group column or omit --group for automatic detection.",
        )
    elif group_column:
        group_count = int(frame[group_column].nunique(dropna=True))
        add_check(
            checks,
            "PASS",
            "Panel/entity structure",
            f"Detected {group_column} with {group_count:,} groups",
            "Repeated dates across groups are expected in panel data.",
        )
    else:
        add_check(
            checks,
            "WARNING",
            "Panel/entity structure",
            "No common group column detected",
            "This is acceptable for a single time series; otherwise provide --group.",
        )

    exact_duplicates = int(frame.duplicated().sum())
    if exact_duplicates == 0:
        duplicate_status = "PASS"
        duplicate_note = "No exact duplicate rows were found."
    else:
        duplicate_status = "WARNING"
        duplicate_note = "Review and justify or remove exact duplicate rows."

    add_check(
        checks,
        duplicate_status,
        "Exact duplicates",
        f"{exact_duplicates:,} duplicate rows",
        duplicate_note,
    )

    if parsed_time is not None:
        key_columns = [time_column]
        if group_column:
            key_columns.insert(0, group_column)
        key_duplicates = int(frame.duplicated(key_columns).sum())

        if key_duplicates == 0:
            key_status = "PASS"
            key_note = "Each analytical unit has one row per time point."
        else:
            key_status = "FAIL"
            key_note = "The analytical key is not unique and must be resolved before modeling."

        add_check(
            checks,
            key_status,
            "Analytical key",
            f"{key_duplicates:,} duplicates for {key_columns}",
            key_note,
        )

    split_column = "data_split" if "data_split" in frame.columns else None
    if split_column and parsed_time is not None:
        split_labels = frame[split_column].astype(str).str.upper()
        split_counts = split_labels.value_counts().to_dict()
        expected = ["TRAIN", "VALIDATION", "TEST"]
        missing_splits = [name for name in expected if name not in split_counts]

        if missing_splits:
            split_status = "FAIL"
            split_note = f"Missing required split groups: {', '.join(missing_splits)}."
        else:
            split_bounds = {}
            for name in expected:
                values = parsed_time.loc[split_labels.eq(name)].dropna()
                split_bounds[name] = (values.min(), values.max())

            chronological = (
                split_bounds["TRAIN"][1] < split_bounds["VALIDATION"][0]
                and split_bounds["VALIDATION"][1] < split_bounds["TEST"][0]
            )
            if chronological:
                split_status = "PASS"
                split_note = "Train, Validation, and Test are separated chronologically."
            else:
                split_status = "FAIL"
                split_note = "The temporal ranges overlap or are in the wrong order."

        split_evidence = ", ".join(
            f"{name}={split_counts.get(name, 0):,}" for name in expected
        )
        add_check(
            checks,
            split_status,
            "Temporal split",
            split_evidence,
            split_note,
        )
    elif parsed_time is not None:
        add_check(
            checks,
            "WARNING",
            "Temporal split",
            "No data_split column found",
            "Create chronological Train, Validation, and Test groups; do not randomly split rows.",
        )

    if specified_features is not None:
        feature_columns = list(specified_features)
        feature_contract = "pre-specified main predictors"
    else:
        excluded_columns = {target, time_column, "data_split"}
        if group_column:
            excluded_columns.add(group_column)
        feature_columns = [
            column
            for column in frame.columns
            if column not in excluded_columns and not is_metadata_column(column)
        ]
        feature_contract = "automatically detected candidate predictors"
    numeric_features = [
        column for column in feature_columns if pd.api.types.is_numeric_dtype(frame[column])
    ]

    if not feature_columns:
        add_check(
            checks,
            "FAIL",
            "Predictor availability",
            "No candidate predictor columns",
            "At least one predictor is required.",
        )
    else:
        rows_per_feature = row_count / len(feature_columns)
        if rows_per_feature >= 20:
            feature_status = "PASS"
            feature_note = "The sample-to-feature ratio supports a controlled model comparison."
        elif rows_per_feature >= 10:
            feature_status = "WARNING"
            feature_note = "Keep the model simple and avoid broad hyperparameter searching."
        else:
            feature_status = "FAIL"
            feature_note = "There are too many candidate predictors for the available rows."

        add_check(
            checks,
            feature_status,
            "Predictor availability",
            (
                f"{len(feature_columns):,} {feature_contract} "
                f"({len(numeric_features):,} numeric); {rows_per_feature:.1f} rows per predictor"
            ),
            feature_note,
        )

        missingness = frame[feature_columns].isna().mean().sort_values(ascending=False)
        high_missing = missingness[missingness > 0.20]
        if high_missing.empty:
            missing_status = "PASS"
            missing_evidence = "No candidate predictor exceeds 20% missingness"
            missing_note = "Predictor missingness is manageable."
        else:
            missing_status = "WARNING"
            missing_evidence = "; ".join(
                f"{column}={value * 100:.1f}%" for column, value in high_missing.head(8).items()
            )
            missing_note = "Review, exclude, or pre-specify imputation for these predictors."

        add_check(
            checks,
            missing_status,
            "Predictor missingness",
            missing_evidence,
            missing_note,
        )

    if numeric_target is not None and numeric_features:
        suspicious = []
        for column in numeric_features:
            pair = pd.concat(
                [numeric_target.rename("target"), frame[column]], axis=1
            ).dropna()
            if len(pair) < 20 or pair[column].nunique() < 2:
                continue
            correlation = pair["target"].corr(pair[column])
            if pd.notna(correlation) and abs(correlation) >= 0.999:
                suspicious.append(f"{column} (r={correlation:.4f})")

        if suspicious:
            leakage_status = "WARNING"
            leakage_evidence = "; ".join(suspicious[:8])
            leakage_note = (
                "Near-perfect target correlation may indicate leakage; inspect timing and derivation."
            )
        else:
            leakage_status = "PASS"
            leakage_evidence = "No numeric predictor has |r| >= 0.999 with the target"
            leakage_note = (
                "This heuristic found no obvious copied target, but timing must still be reviewed."
            )

        add_check(
            checks,
            leakage_status,
            "Basic leakage screen",
            leakage_evidence,
            leakage_note,
        )

    return {
        "rows": row_count,
        "columns": column_count,
        "group_column": group_column,
    }


def build_report(args, checks, details, command_text):
    """Create a readable Markdown feasibility report."""
    fail_count = sum(item["status"] == "FAIL" for item in checks)
    warning_count = sum(item["status"] == "WARNING" for item in checks)
    pass_count = sum(item["status"] == "PASS" for item in checks)

    if fail_count:
        overall = "NOT YET FEASIBLE"
        conclusion = (
            "At least one blocking data problem must be corrected before the proposed "
            "modeling workflow is defensible."
        )
    elif warning_count:
        overall = "FEASIBLE WITH WARNINGS"
        conclusion = (
            "The data can support the proposed workflow, provided that the warnings "
            "are explained or addressed in the proposal."
        )
    else:
        overall = "FEASIBLE"
        conclusion = "No blocking technical feasibility problem was detected."

    lines = [
        "# Dataset Feasibility Report",
        "",
        f"**Overall result:** {overall}",
        "",
        conclusion,
        "",
        "> This report evaluates technical data support only. It does not determine "
        "whether the research question is interesting, clinically useful, or causal.",
        "",
        "## Inputs",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Target: `{args.target}`",
        f"- Time column: `{args.time}`",
        f"- Panel/entity column: `{details.get('group_column') or 'not detected'}`",
        f"- Fetch script: `{args.fetch_script}`",
        f"- Feature dictionary: `{args.dictionary or 'not supplied'}`",
        f"- Generated (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "## Command",
        "",
        "```bash",
        command_text,
        "```",
        "",
        "## Summary",
        "",
        f"- PASS: {pass_count}",
        f"- WARNING: {warning_count}",
        f"- FAIL: {fail_count}",
        "",
        "## Detailed checks",
        "",
        "| Status | Check | Evidence | Interpretation |",
        "|---|---|---|---|",
    ]

    for item in checks:
        lines.append(
            "| {status} | {check} | {evidence} | {interpretation} |".format(
                status=markdown_text(item["status"]),
                check=markdown_text(item["check"]),
                evidence=markdown_text(item["evidence"]),
                interpretation=markdown_text(item["interpretation"]),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation for the proposal",
            "",
            conclusion,
            "The final proposal should also document the unit of analysis, feature timing, "
            "baseline, temporal split strategy, data coverage, and limitations.",
            "",
        ]
    )

    return "\n".join(lines), fail_count


def main():
    """Run the checks, write the report, and return a useful exit status."""
    args = parse_arguments()
    dataset_path = Path(args.dataset)
    fetch_path = Path(args.fetch_script)
    dictionary_path = Path(args.dictionary) if args.dictionary else None
    output_path = Path(args.output)
    checks = []
    details = {"group_column": None}

    check_fetch_script(fetch_path, checks)

    if not dataset_path.exists() or not dataset_path.is_file():
        add_check(
            checks,
            "FAIL",
            "Dataset file",
            f"Not found: {dataset_path}",
            "Run the preparation script or correct the dataset path.",
        )
    else:
        try:
            frame, encoding = read_csv_with_fallback(dataset_path)
            add_check(
                checks,
                "PASS",
                "Dataset file",
                f"Loaded with {encoding}",
                "The CSV can be read successfully.",
            )
            specified_features = read_feature_dictionary(
                dictionary_path, frame, checks
            )
            details = check_dataset(
                frame=frame,
                target=args.target,
                time_column=args.time,
                requested_group=args.group,
                checks=checks,
                specified_features=specified_features,
            )
        except Exception as error:
            add_check(
                checks,
                "FAIL",
                "Dataset loading",
                f"{type(error).__name__}: {error}",
                "Correct the file format or data problem and rerun the check.",
            )

    command_parts = [
        "python",
        Path(sys.argv[0]).as_posix(),
        args.dataset,
        "--target",
        args.target,
        "--time",
        args.time,
        "--fetch-script",
        args.fetch_script,
    ]
    if args.group:
        command_parts.extend(["--group", args.group])
    if args.dictionary:
        command_parts.extend(["--dictionary", args.dictionary])
    if args.output != "outputs/quality/feasibility_report.md":
        command_parts.extend(["--output", args.output])
    command_text = " \\\n    ".join(shlex.quote(part) for part in command_parts)

    report_text, fail_count = build_report(args, checks, details, command_text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")

    overall_line = report_text.splitlines()[2]
    print(overall_line.replace("**", ""))
    print(f"Report: {output_path}")

    return 1 if fail_count else 0


if __name__ == "__main__":
    sys.exit(main())
