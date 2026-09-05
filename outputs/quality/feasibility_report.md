# Dataset Feasibility Report

**Overall result:** FEASIBLE

No blocking technical feasibility problem was detected.

> This report evaluates technical data support only. It does not determine whether the research question is interesting, clinically useful, or causal.

## Inputs

- Dataset: `data/processed/analysis_ready/env_data_notnull.csv`
- Target: `observed_pm25_median_ugm3`
- Time column: `analysis_date`
- Panel/entity column: `province_key`
- Fetch script: `src/fetch_data.py`
- Feature dictionary: `outputs/metadata/data_dictionary/env_data_dictionary.csv`
- Generated (UTC): 2026-09-02T15:50:17+00:00

## Command

```bash
python \
    check_dataset.py \
    data/processed/analysis_ready/env_data_notnull.csv \
    --target \
    observed_pm25_median_ugm3 \
    --time \
    analysis_date \
    --fetch-script \
    src/fetch_data.py \
    --dictionary \
    outputs/metadata/data_dictionary/env_data_dictionary.csv
```

## Summary

- PASS: 15
- WARNING: 0
- FAIL: 0

## Detailed checks

| Status | Check | Evidence | Interpretation |
|---|---|---|---|
| PASS | Fetch script | src/fetch_data.py (1,786 lines) | The script contains acquisition/verification logic and code that records data. |
| PASS | Dataset file | Loaded with utf-8-sig | The CSV can be read successfully. |
| PASS | Feature dictionary | 25 main features; loaded with utf-8-sig | Predictor checks will use only the pre-specified main model features. |
| PASS | Dataset size | 8,222 rows and 38 columns | The dataset has enough rows for a basic modeling workflow. |
| PASS | Target validity | 0 missing (0.00%); 0 invalid numeric values | The target is numeric and complete. |
| PASS | Target variation | 2,785 unique values; min=0.000; median=17.100; max=423.500; SD=26.452 | The continuous target has usable variation for regression. |
| PASS | Time validity | 0 invalid (0.00%); 1,242 unique times; 2021-08-17 to 2025-12-31 | All time values can be parsed. |
| PASS | Temporal coverage | 1,242 distinct time points | The time axis is long enough for a chronological split. |
| PASS | Panel/entity structure | Detected province_key with 8 groups | Repeated dates across groups are expected in panel data. |
| PASS | Exact duplicates | 0 duplicate rows | No exact duplicate rows were found. |
| PASS | Analytical key | 0 duplicates for ['province_key', 'analysis_date'] | Each analytical unit has one row per time point. |
| PASS | Temporal split | TRAIN=3,657, VALIDATION=2,153, TEST=2,412 | Train, Validation, and Test are separated chronologically. |
| PASS | Predictor availability | 25 pre-specified main predictors (25 numeric); 328.9 rows per predictor | The sample-to-feature ratio supports a controlled model comparison. |
| PASS | Predictor missingness | No candidate predictor exceeds 20% missingness | Predictor missingness is manageable. |
| PASS | Basic leakage screen | No numeric predictor has \|r\| >= 0.999 with the target | This heuristic found no obvious copied target, but timing must still be reviewed. |

## Interpretation for the proposal

No blocking technical feasibility problem was detected.
The final proposal should also document the unit of analysis, feature timing, baseline, temporal split strategy, data coverage, and limitations.
