# Northern Thailand PM2.5 — Simple Run Guide

## RUN GUIDE — Purpose

> 🟦 This README explains only how to run and verify the code. It does not contain the research questions, hypotheses, report narrative, or interpretation of results.

## Introduction

This repository turns governed raw-source files into source-aligned tables, analysis-ready datasets, technical quality reports, figures, fitted models, and machine-readable model results. Run all commands from the repository root—the directory containing <code>README.md</code>, <code>requirements.txt</code>, and <code>src/</code>.

The current executable path contains six Python scripts. The old names <code>analyse.py</code> and <code>model.py</code> are not part of the current run.

### Input

| Input | Location | Used for |
|---|---|---|
| Python scripts | <code>src/</code> | Acquisition, preparation, checking, descriptive analysis, and modeling |
| Python packages | <code>requirements.txt</code> | Recreate the required environment |
| Git visibility rules | <code>.gitignore</code> | Keep credentials and machine-specific files out of the repository |
| Frozen raw files | <code>data/raw/</code> | Reproduce the delivered pipeline without downloading new data |
| API keys | Shell environment variables | Acquire missing OpenAQ and NASA FIRMS files in Live mode only |

Run all commands from the repository root: the folder that contains <code>README.md</code>, <code>requirements.txt</code>, and <code>src/</code>.

### Process

There are two routes:

- 🟢 **Frozen** — go from Process 0 directly to Process 6. Process 6 runs the complete pipeline with the existing raw snapshot.
- 🟠 **Live** — run Processes 1–5 in order. This route creates or completes a raw snapshot before processing it.

### Output

| Output layer | Location | What it contains |
|---|---|---|
| Acquisition | <code>data/raw/</code>, <code>outputs/metadata/</code> | Source files, manifest, and completeness status |
| Prepared data | <code>data/processed/</code>, <code>outputs/quality/</code> | Standardized tables, analysis-ready files, dictionaries, and checks |
| Descriptive analysis | <code>outputs/analysis/</code> | Tables, figures, summaries, and model-ready files |
| AT1 and AT2 | <code>outputs/modeling/</code>, <code>outputs/models/</code> | Model results, predictions, diagnostics, and fitted model |

## Approach

### Choose one route

~~~mermaid
flowchart TD
    P0["Process 0<br/>Set up repository"] --> MODE{"Choose route"}
    MODE -->|Frozen| P6["Process 6<br/>Run complete Frozen pipeline"]
    MODE -->|Live| P1["Process 1<br/>Acquire missing source files"]
    P1 --> P2["Process 2<br/>Prepare data"]
    P2 --> P3{"Process 3<br/>Run feasibility gate"}
    P3 -->|Pass| P4["Process 4<br/>Descriptive analysis"]
    P4 --> P5["Process 5<br/>Run AT1 and AT2"]
    classDef setup fill:#DBEAFE,stroke:#2563EB,color:#172554
    classDef choice fill:#F3E8FF,stroke:#7C3AED,color:#3B0764
    classDef frozen fill:#DCFCE7,stroke:#16A34A,color:#14532D
    classDef live fill:#FFEDD5,stroke:#EA580C,color:#7C2D12
    classDef work fill:#FCE7F3,stroke:#DB2777,color:#831843
    class P0 setup
    class MODE,P3 choice
    class P6 frozen
    class P1 live
    class P2,P4,P5 work
~~~

| Route | Use it when | Run next |
|---|---|---|
| 🟢 Frozen | You want to reproduce the project from the existing <code>data/raw/</code> files | Process 6 |
| 🟠 Live | You intentionally want to download missing source files | Process 1, then 2, 3, 4, and 5 |

> 🟥 **Stop rule**  
> If a process fails, fix its input or script. Do not manually edit a downstream output CSV to make the next process pass.

---

### Process 0 — Set up the repository

#### 0.1 Check the repository

The executable files must use these exact names:

~~~text
.
├── .gitignore
├── README.md
├── requirements.txt
├── check_dataset.py
└── src/
    ├── fetch_data.py
    ├── prepare_tables.py
    ├── prepare_data.py
    ├── analysis.py
    └── modeling.py
~~~

Run:

~~~bash
cd "<repository-root>"
pwd
ls
ls src
~~~

The <code>ls src</code> output must include all six filenames above. Do not use the older names <code>analyse.py</code> or <code>model.py</code>.

#### 0.2 Create the Python environment

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python --version
~~~

Use Python 3.12. If activation succeeds, the terminal normally shows <code>(.venv)</code>.

#### 0.3 Install and test requirements

~~~bash
python -m pip install -r requirements.txt
python -m pip check
python -m py_compile check_dataset.py src/fetch_data.py src/prepare_tables.py src/prepare_data.py  src/analysis.py src/modeling.py
~~~

| Output | Meaning | Decision |
|---|---|---|
| <code>No broken requirements found.</code> | Installed packages have no declared conflicts | Continue |
| No output from <code>py_compile</code> | All six scripts compile | Continue |
| Installation, dependency, or syntax error | The environment is not ready | Stop and fix Process 0 |

> 🟦 **Optional**  
> Run <code>python -m pip install --upgrade pip</code> before installing requirements if the installer itself is too old.

#### 0.4 Configure <code>.gitignore</code>

Create <code>.gitignore</code> in the repository root. It controls files that must stay on the local computer and must not be visible in GitHub.

##### Files that must not be visible in GitHub

| File type | Examples | Reason |
|---|---|---|
| Credentials | <code>.env</code>, API keys, <code>*.pem</code>, <code>*.key</code> | Anyone with repository access could reuse the secret |
| Local Python environment | <code>.venv/</code>, <code>venv/</code> | Large, machine-specific, and recreated from <code>requirements.txt</code> |
| Cache files | <code>__pycache__/</code>, <code>*.pyc</code>, notebook checkpoints | Generated automatically and not required to run the project |
| Editor or operating-system settings | <code>.vscode/</code>, <code>.idea/</code>, <code>.DS_Store</code> | Personal machine settings, not project inputs |
| Incomplete downloads and local logs | <code>*.part</code>, <code>*.crdownload</code>, <code>*.log</code> | Temporary files are not governed pipeline outputs |

Copy these rules into <code>.gitignore</code>:

~~~gitignore
# Local Python environments
.venv/
venv/
env/

# Generated caches
__pycache__/
*.py[cod]
.pytest_cache/
.ipynb_checkpoints/

# Credentials
.env
.env.*
!.env.example
*.pem
*.key

# Local editor and OS files
.vscode/
.idea/
.DS_Store

# Temporary files
*.tmp
*.part
*.crdownload
*.log
outputs/logs/
~~~

##### Files that must remain visible and trackable

Do not add these paths to <code>.gitignore</code>:

~~~text
README.md
requirements.txt
src/
data/raw/
data/processed/
outputs/metadata/
outputs/quality/
outputs/analysis/
outputs/modeling/
outputs/models/
~~~

These files are needed to reproduce or verify the submitted pipeline. If a required file is too large for normal Git, use Git LFS or the submission method approved for the project instead of silently ignoring it.

Verify Git visibility before publishing:

~~~bash
git status --short --ignored
~~~

| Git status prefix | Meaning | Decision |
|---|---|---|
| <code>!!</code> before <code>.venv/</code>, cache, or <code>.env</code> | The local-only path is ignored | Correct |
| <code>??</code> before a secret file | The file is untracked but visible to Git | Stop and correct <code>.gitignore</code> |
| <code>A</code> or <code>M</code> before code, governed data, or required outputs | A required file is added or modified | Review, then commit intentionally |

> 🟥 **If a key was already committed**  
> Adding it to <code>.gitignore</code> does not remove it from Git history. Revoke or rotate the key first, then remove the secret from the repository history before sharing the repository.

> 🟦 **Optional — safe template**  
> A tracked <code>.env.example</code> may contain variable names such as <code>OPENAQ_API_KEY=</code> and <code>NASA_FIRMS_MAP_KEY=</code>, but it must never contain real values.

**Result of Process 0**

> ✅ The environment is active, the requirements are installed, the six correctly named scripts compile, and private or machine-specific files are excluded from Git.

---

### Process 1 — Live acquisition

Process 1 is used only for the **Live** route. Frozen users skip this section and go directly to Process 6.

#### 1.1 Source websites

**A. Public download or website sources**

| Source | Pipeline input | Website |
|---|---|---|
| Department of Disease Control | Respiratory records and dictionary | [DDC ODPC1 dataset](https://opendata.ddc.moph.go.th/dataset/odpc1-01) |
| Department of Provincial Administration | Registered population | [DOPA population statistics](https://stat.bora.dopa.go.th/new_stat/webPage/statByYear.php) |
| Ministry of Public Health | Facility master | [MOPH HCODE](https://hcode.moph.go.th/code/) |
| Healthcare Accreditation Institute | Facility capacity | [HAI dataset](https://data.ha.or.th/dataset/hospital/resource/4e20e752-25f8-468c-b155-33be7aecc0d4) |
| Open Development Mekong | Province boundaries | [Thailand provincial boundaries](https://data.thailand.opendevelopmentmekong.net/th/dataset/thailand-provincial-boundaries) |
| Open-Meteo | Weather and modeled PM2.5 | [Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) · [Air Quality API](https://open-meteo.com/en/docs/air-quality-api) |

**B. API sources that require a key**

| Source | Pipeline input | Get a key |
|---|---|---|
| OpenAQ v3 | Observed PM2.5 | [Create an OpenAQ account](https://explore.openaq.org/register) · [API-key guide](https://docs.openaq.org/using-the-api/api-key) |
| NASA FIRMS Area API | Fire hotspots | [Request a FIRMS MAP_KEY](https://firms.modaps.eosdis.nasa.gov/api/map_key/) |

Keep keys in environment variables. Do not put them inside Python files or commit them to Git.

#### Frozen and Live are different

| | 🟢 Frozen | 🟠 Live |
|---|---|---|
| Raw input | Reuses the existing snapshot | Downloads only missing files |
| Network | No live API calls | Uses websites and APIs |
| API keys | Not needed | Needed for missing OpenAQ or FIRMS files |
| Result | Reproducible from the delivered files | A new or completed snapshot that may produce different outputs |
| Next step | Process 6 | Process 1.3 |

#### 1.2 Frozen choice

Do not run a separate acquisition command here. Go directly to **Process 6**, whose first command runs <code>fetch_data.py</code> in its default Frozen mode.

#### 1.3 Run Live acquisition

**Input**

- <code>src/fetch_data.py</code>
- Internet access
- API keys when OpenAQ or FIRMS files are missing

Run:

~~~bash
export FETCH_LIVE=1
export OPENAQ_API_KEY="YOUR_OPENAQ_API_KEY"
export NASA_FIRMS_MAP_KEY="YOUR_NASA_FIRMS_MAP_KEY"
python src/fetch_data.py
~~~

After the run, remove the values from the current shell:

~~~bash
unset FETCH_LIVE
unset OPENAQ_API_KEY
unset NASA_FIRMS_MAP_KEY
~~~

| Console output | Meaning | Decision |
|---|---|---|
| <code>LIVE FETCH MODE: creating or completing a new raw snapshot</code> | Live mode is active | Continue only if this was intentional |
| <code>Fetched ...</code> or <code>Imported downloaded source: ...</code> | A missing file was added | Continue |
| <code>OpenAQ skipped: ...</code> | No OpenAQ key was available | Accept only if required OpenAQ files already existed |
| <code>NASA FIRMS skipped: ...</code> | No FIRMS key was available | Accept only if required FIRMS files already existed |
| <code>All source groups were fetched or verified successfully.</code> | All required groups are complete | Continue to verification |
| <code>Source acquisition is incomplete:</code> | At least one required group is missing | Stop |

> 🟦 **Optional — manually downloaded files**  
> If a publisher blocks automatic download, place the correct publisher files in one folder, set <code>DOWNLOAD_DIR</code> to that folder, and rerun Live acquisition.

~~~bash
export DOWNLOAD_DIR="/absolute/path/to/downloaded-files"
python src/fetch_data.py
~~~

#### 1.4 Verify acquisition

No additional command is required. Open:

- <code>outputs/metadata/fetch_source_status.csv</code> — every value in <code>complete</code> must be <code>True</code>.
- <code>outputs/metadata/fetch_manifest.csv</code> — every raw file must have a unique path, positive size, and SHA256 value.
- <code>data/raw/</code> — source files must exist and must not be temporary or empty files.

If any source has <code>complete = False</code>, do not start Process 2.

**Outputs of Process 1**

~~~text
data/raw/**
outputs/metadata/fetch_manifest.csv
outputs/metadata/fetch_source_status.csv
~~~

**Result of Process 1**

> ✅ Live acquisition produces a complete raw snapshot plus a manifest and source-completeness file. Continue to Process 2 only when every required source group is complete.

---

### Process 2 — Prepare the data

#### 2.1 Run the preparation scripts

**Input**

| Script | Reads |
|---|---|
| <code>src/prepare_tables.py</code> | <code>data/raw/**</code> and acquisition status |
| <code>src/prepare_data.py</code> | The source tables created by <code>prepare_tables.py</code> |

Run:

~~~bash
python src/prepare_tables.py
python src/prepare_data.py
~~~

| Console output | Meaning | Decision |
|---|---|---|
| <code>Source-table preparation completed successfully.</code> | Standardized source tables were created | Continue |
| <code>Analysis-ready data preparation completed successfully.</code> | Merged analysis-ready files and dictionaries were created | Verify outputs |
| <code>Raw acquisition is incomplete</code> | Process 1 did not pass | Return to Process 1 |
| Key, duplicate, or merge error | Table grain is invalid | Stop and fix the upstream input or preparation code |

**Source tables**

~~~text
data/processed/source_tables/provinces.csv
data/processed/source_tables/location.csv
data/processed/source_tables/observepm25_sensorday.csv
data/processed/source_tables/observepm25_provinceday.csv
data/processed/source_tables/weather_provinceday.csv
data/processed/source_tables/modelpm25_provinceday.csv
data/processed/source_tables/hotspot.csv
data/processed/source_tables/facility.csv
data/processed/source_tables/population.csv
data/processed/source_tables/hospital.csv
data/processed/source_tables/diagnosis.csv
~~~

**Analysis-ready files and dictionaries**

~~~text
data/processed/analysis_ready/data_per_day.csv
data/processed/analysis_ready/env_data_full.csv
data/processed/analysis_ready/env_data_notnull.csv
data/processed/analysis_ready/respi_disease_data.csv

outputs/metadata/data_dictionary/data_per_day_dictionary.csv
outputs/metadata/data_dictionary/env_data_dictionary.csv
outputs/metadata/data_dictionary/respi_disease_data_dictionary.csv
~~~

**Quality files**

~~~text
outputs/quality/table_quality_report.csv
outputs/quality/missing_value_report.csv
outputs/quality/dtype_report.csv
outputs/quality/duplicate_report.csv
outputs/quality/duplicate_key_examples.csv
outputs/quality/foreign_key_report.csv
outputs/quality/source_coverage_report.csv
outputs/quality/processing_decisions.csv
outputs/quality/source_processing_counts.csv
outputs/quality/merge_retention_report.csv
outputs/quality/coverage_report.csv
outputs/quality/model_eligibility_report.csv
outputs/quality/model_exclusion_reason_report.csv
outputs/quality/analytical_decisions.csv
outputs/metadata/prepare_tables_summary.json
outputs/metadata/prepare_data_summary.json
~~~

#### 2.2 Verify preparation

No additional command is required.

| File | What to check | Decision |
|---|---|---|
| <code>table_quality_report.csv</code> | Source-table status | <code>FAIL</code> blocks the pipeline |
| <code>duplicate_key_examples.csv</code> | Unexpected duplicate keys | Unexpected rows must be fixed upstream |
| <code>foreign_key_report.csv</code> | Invalid province keys | Must be zero |
| <code>merge_retention_report.csv</code> | Left-table grain is preserved | Any failed merge blocks the pipeline |
| <code>model_eligibility_report.csv</code> | Required model groups contain eligible rows | Empty required groups block modeling |

> 🟦 **Optional**  
> Compare the quality CSVs with the last accepted run when using a new Live snapshot. A difference is not automatically an error, but it should be understood.

**Result of Process 2**

> ✅ Process 2 produces standardized source tables, four analysis-ready datasets, three dictionaries, and quality evidence for the next gate.

---

### Process 3 — Run the feasibility gate

#### 3.1 Run the gate

**Input**

~~~text
check_dataset.py
src/fetch_data.py
data/processed/analysis_ready/env_data_notnull.csv
outputs/metadata/data_dictionary/env_data_dictionary.csv
~~~

The <code>--dictionary</code> option is optional for the generic checker, but **use it for this project**. It limits the check to the features declared by the project.

Run:

~~~bash
python check_dataset.py data/processed/analysis_ready/env_data_notnull.csv --target observed_pm25_median_ugm3 --time analysis_date --group province_key --fetch-script src/fetch_data.py --dictionary outputs/metadata/data_dictionary/env_data_dictionary.csv --output outputs/quality/feasibility_report.md
~~~

**Output**

~~~text
outputs/quality/feasibility_report.md
~~~

#### 3.2 Read the gate result

| Result | Meaning | Decision |
|---|---|---|
| <code>FEASIBLE</code> | No blocking check failed | Continue |
| <code>FEASIBLE WITH WARNINGS</code> | No blocking failure, but warnings need review | Review, document the decision, then continue |
| <code>NOT YET FEASIBLE</code> | At least one blocking check failed | Stop |

Open <code>feasibility_report.md</code> and confirm that it refers to the intended dataset, target, group, acquisition script, and dictionary. The report must show <code>FAIL: 0</code> before Process 4.

> 🟦 **Optional**  
> When testing another candidate dataset, give <code>--output</code> a different filename so the accepted gate file is not replaced.

**Result of Process 3**

> ✅ Process 3 produces one clear technical decision: continue only when the gate has zero failures.

---

### Process 4 — Descriptive analysis

#### 4.1 Run descriptive analysis

**Input**

~~~text
src/analysis.py
data/processed/analysis_ready/data_per_day.csv
data/processed/analysis_ready/env_data_full.csv
data/processed/analysis_ready/env_data_notnull.csv
data/processed/analysis_ready/respi_disease_data.csv
~~~

Run:

~~~bash
python src/analysis.py
~~~

| Console output | Meaning | Decision |
|---|---|---|
| <code>Creating model-ready datasets</code> | AT1 and AT2 inputs are being created | Continue |
| <code>Creating eight pre-model report tables</code> | Descriptive tables are being written | Continue |
| <code>Creating report Figure 1 plus twelve supporting figures</code> | Descriptive images are being written | Continue |
| <code>Analysis completed successfully.</code> | All declared outputs were created | Verify outputs |
| Missing-input, duplicate-key, or missing-feature error | Process 2 output is not valid for analysis | Stop and correct Process 2 |

**Model-ready outputs**

~~~text
data/processed/model_ready/at1_model_ready.csv
data/processed/model_ready/at2_model_ready.csv
~~~

**Descriptive tables**

~~~text
outputs/analysis/tables/province_pm25.csv
outputs/analysis/tables/month_pm25.csv
outputs/analysis/tables/exceed_pm25.csv
outputs/analysis/tables/hotspot_summary.csv
outputs/analysis/tables/correlation.csv
outputs/analysis/tables/correlation_comparison.csv
outputs/analysis/tables/province_respi.csv
outputs/analysis/tables/hospital_capacity.csv
~~~

**Descriptive figures**

~~~text
outputs/analysis/figures/fig01_pm25_descriptive_overview.png
outputs/analysis/figures/figS01_data_coverage.png
outputs/analysis/figures/figS02_pm25_temporal_trend.png
outputs/analysis/figures/figS03_hotspot_pm25_relationship.png
outputs/analysis/figures/figS04_pm25_distribution_by_province.png
outputs/analysis/figures/figS05_pm25_exceedance.png
outputs/analysis/figures/figS06_weather_pm25_relationship.png
outputs/analysis/figures/figS07_predictor_correlation.png
outputs/analysis/figures/figS08_observed_modeled_benchmark.png
outputs/analysis/figures/figS09_hospital_capacity.png
outputs/analysis/figures/figS10_sensor_location_map.png
outputs/analysis/figures/figS11_model_eligibility_by_split.png
outputs/analysis/figures/figS12_pm25_observation_coverage.png
~~~

**Control outputs**

~~~text
outputs/analysis/analysis_manifest.csv
outputs/analysis/analysis_summary.json
outputs/analysis/discussion_decisions.md
~~~

#### 4.2 Verify descriptive analysis

No additional command is required. Open <code>analysis_manifest.csv</code> and confirm that:

- two model-ready files exist and are not empty;
- eight descriptive tables exist;
- thirteen PNG figures exist and open correctly;
- <code>analysis_summary.json</code> agrees with the manifest.

> 🟦 **Optional**  
> Open every PNG at normal size and check for blank panels, clipped labels, or unreadable legends.

**Result of Process 4**

> ✅ Process 4 is the **descriptive-analysis stage**. It produces descriptive tables and figures plus the two model-ready inputs required by Process 5.

---

### Process 5 — Run AT1 and AT2

| Component | What the script does | Input |
|---|---|---|
| AT1 | Predicts the next PM2.5 value for each province using lagged pollution, weather, calendar, and fire-hotspot features | <code>at1_model_ready.csv</code> |
| AT2 | Estimates an exploratory association between monthly PM2.5 and respiratory-record counts per active facility | <code>at2_model_ready.csv</code> |

AT2 is exploratory; its output should not be read as proof of causation.

#### 5.1 Run modeling

**Input**

~~~text
src/modeling.py
data/processed/model_ready/at1_model_ready.csv
data/processed/model_ready/at2_model_ready.csv
~~~

Run:

~~~bash
python src/modeling.py
~~~

| Console output | Meaning | Decision |
|---|---|---|
| <code>Selecting Random Forest configurations ...</code> | AT1 model selection is running | Continue |
| <code>Frozen selection: ...</code> | The selected AT1 configuration is locked before final evaluation | Continue |
| <code>Fitting exploratory AT2 regression</code> | AT2 analysis is running | Continue |
| <code>Modeling completed successfully.</code> | Results, figures, and fitted model were created | Verify outputs |
| Split, missing-value, duplicate-key, or missing-file error | Model-ready input is invalid | Stop and return to Process 4 or earlier |

Here, <code>Frozen selection</code> describes the locked **model choice**. It is not the Frozen/Live acquisition route.

**AT1 result files**

~~~text
outputs/modeling/results/validation_tuning_results.csv
outputs/modeling/results/validation_model_summary.csv
outputs/modeling/results/test_results.csv
outputs/modeling/results/test_predictions.csv
outputs/modeling/results/warning_metrics.csv
outputs/modeling/results/cross_validation_results.csv
outputs/modeling/results/province_season_errors.csv
outputs/modeling/results/feature_importance.csv
~~~

**AT2, summary, model, and figures**

~~~text
outputs/analysis/tables/pm25_respi.csv
outputs/modeling/modeling_summary.json
outputs/modeling/modeling_manifest.csv
outputs/models/selected_random_forest.joblib
outputs/analysis/figures/fig02_model_comparison.png
outputs/analysis/figures/fig03_test_predictions.png
outputs/analysis/figures/fig04_error_by_province_season.png
outputs/analysis/figures/fig05_at2_exploratory_association.png
~~~

#### 5.2 Verify AT1 and AT2

No additional command is required.

- <code>modeling_manifest.csv</code> must point to existing, non-empty outputs.
- <code>modeling_summary.json</code> must name the selected model and confirm that final evaluation happened after model selection.
- <code>test_predictions.csv</code> must contain non-missing predictions with unique keys.
- <code>pm25_respi.csv</code> must contain the AT2 result rows.
- <code>selected_random_forest.joblib</code> must exist and be non-empty.

> 🟦 **Optional**  
> Review <code>cross_validation_results.csv</code>, <code>province_season_errors.csv</code>, <code>warning_metrics.csv</code>, and <code>feature_importance.csv</code> before reusing the model.

**Result of Process 5**

> ✅ Process 5 produces AT1 predictions, evaluation files, diagnostics, and the fitted pipeline, together with the exploratory AT2 association output.

---

<a id="full-repository-tree"></a>

### Process 5.3 — FULL REPOSITORY TREE

> 🟦 **Required end-of-Process-5 check**  
> Use this complete tree to check that the final repository contains the code, inputs, processed data, and generated outputs in the expected locations.

~~~text
.
├── .gitignore
├── README.md
├── requirements.txt
├── check_dataset.py
├── src/
│   ├── fetch_data.py
│   ├── prepare_tables.py
│   ├── prepare_data.py
│   ├── analysis.py
│   └── modeling.py
├── data/
│   ├── raw/
│   │   ├── DDC/
│   │   ├── PROVIDER/
│   │   ├── POPULATIONS/
│   │   ├── BOUNDARY/
│   │   └── api/
│   │       ├── open_meteo_weather/
│   │       ├── open_meteo_modeled_pm25/
│   │       ├── openaq/
│   │       └── nasa_firms/
│   └── processed/
│       ├── source_tables/
│       │   ├── provinces.csv
│       │   ├── location.csv
│       │   ├── observepm25_sensorday.csv
│       │   ├── observepm25_provinceday.csv
│       │   ├── weather_provinceday.csv
│       │   ├── modelpm25_provinceday.csv
│       │   ├── hotspot.csv
│       │   ├── facility.csv
│       │   ├── population.csv
│       │   ├── hospital.csv
│       │   └── diagnosis.csv
│       ├── analysis_ready/
│       │   ├── data_per_day.csv
│       │   ├── env_data_full.csv
│       │   ├── env_data_notnull.csv
│       │   └── respi_disease_data.csv
│       └── model_ready/
│           ├── at1_model_ready.csv
│           └── at2_model_ready.csv
└── outputs/
    ├── metadata/
    │   ├── fetch_manifest.csv
    │   ├── fetch_source_status.csv
    │   ├── prepare_tables_summary.json
    │   ├── prepare_data_summary.json
    │   └── data_dictionary/
    │       ├── data_per_day_dictionary.csv
    │       ├── env_data_dictionary.csv
    │       └── respi_disease_data_dictionary.csv
    ├── quality/
    │   ├── table_quality_report.csv
    │   ├── missing_value_report.csv
    │   ├── dtype_report.csv
    │   ├── duplicate_report.csv
    │   ├── duplicate_key_examples.csv
    │   ├── foreign_key_report.csv
    │   ├── source_coverage_report.csv
    │   ├── processing_decisions.csv
    │   ├── source_processing_counts.csv
    │   ├── merge_retention_report.csv
    │   ├── coverage_report.csv
    │   ├── model_eligibility_report.csv
    │   ├── model_exclusion_reason_report.csv
    │   ├── analytical_decisions.csv
    │   └── feasibility_report.md
    ├── analysis/
    │   ├── analysis_manifest.csv
    │   ├── analysis_summary.json
    │   ├── discussion_decisions.md
    │   ├── tables/
    │   │   ├── province_pm25.csv
    │   │   ├── month_pm25.csv
    │   │   ├── exceed_pm25.csv
    │   │   ├── hotspot_summary.csv
    │   │   ├── correlation.csv
    │   │   ├── correlation_comparison.csv
    │   │   ├── province_respi.csv
    │   │   ├── hospital_capacity.csv
    │   │   └── pm25_respi.csv
    │   └── figures/
    │       ├── fig01_pm25_descriptive_overview.png
    │       ├── fig02_model_comparison.png
    │       ├── fig03_test_predictions.png
    │       ├── fig04_error_by_province_season.png
    │       ├── fig05_at2_exploratory_association.png
    │       ├── figS01_data_coverage.png
    │       ├── figS02_pm25_temporal_trend.png
    │       ├── figS03_hotspot_pm25_relationship.png
    │       ├── figS04_pm25_distribution_by_province.png
    │       ├── figS05_pm25_exceedance.png
    │       ├── figS06_weather_pm25_relationship.png
    │       ├── figS07_predictor_correlation.png
    │       ├── figS08_observed_modeled_benchmark.png
    │       ├── figS09_hospital_capacity.png
    │       ├── figS10_sensor_location_map.png
    │       ├── figS11_model_eligibility_by_split.png
    │       └── figS12_pm25_observation_coverage.png
    ├── modeling/
    │   ├── modeling_summary.json
    │   ├── modeling_manifest.csv
    │   └── results/
    │       ├── validation_tuning_results.csv
    │       ├── validation_model_summary.csv
    │       ├── test_results.csv
    │       ├── test_predictions.csv
    │       ├── warning_metrics.csv
    │       ├── cross_validation_results.csv
    │       ├── province_season_errors.csv
    │       └── feature_importance.csv
    └── models/
        └── selected_random_forest.joblib
~~~

---

### Process 6 — Run the complete Frozen pipeline

Use Process 6 immediately after Process 0 when the **Frozen** route is selected. Do not run Processes 1–5 separately first.

Before running, make sure <code>FETCH_LIVE</code> is not set to <code>1</code>. With no Live flag, <code>fetch_data.py</code> uses Frozen mode by default.

#### 6.1 Run all scripts in order

**Input**

~~~text
requirements.txt
src/fetch_data.py
src/prepare_tables.py
src/prepare_data.py
check_dataset.py
src/analysis.py
src/modeling.py
data/raw/**
~~~

Run these same commands, in this order:

~~~bash
python src/fetch_data.py
python src/prepare_tables.py
python src/prepare_data.py
python check_dataset.py data/processed/analysis_ready/env_data_notnull.csv --target observed_pm25_median_ugm3 --time analysis_date --group province_key --fetch-script src/fetch_data.py --dictionary outputs/metadata/data_dictionary/env_data_dictionary.csv --output outputs/quality/feasibility_report.md
python src/analysis.py
python src/modeling.py
~~~

| Console output | Meaning | Decision |
|---|---|---|
| <code>FROZEN MODE: live APIs were not called.</code> | The existing raw snapshot is being used | Continue |
| Each script ends with <code>completed successfully</code> | That stage completed | Run the next command |
| Gate result is <code>FEASIBLE</code> or <code>FEASIBLE WITH WARNINGS</code> with zero failures | Technical gate passed | Continue |
| Any traceback, incomplete-source message, or <code>NOT YET FEASIBLE</code> | The Frozen pipeline is incomplete | Stop at the first failed process |

#### 6.2 Verify the Frozen pipeline

No additional command is required. Confirm:

- the first command printed <code>FROZEN MODE: live APIs were not called.</code>;
- <code>fetch_source_status.csv</code> has no incomplete source;
- <code>feasibility_report.md</code> has zero failures;
- <code>analysis_manifest.csv</code> resolves every Process 4 output;
- <code>modeling_manifest.csv</code> resolves every Process 5 output;
- the two model-ready CSVs and the fitted <code>.joblib</code> file are not empty.

If the first command prints <code>LIVE FETCH MODE</code>, stop describing the run as Frozen.

> 🟦 **Optional**  
> For a clean-room check, copy the repository and its governed <code>data/raw/</code> snapshot to a fresh folder, recreate the environment, and repeat Processes 0 and 6.

**Outputs of Process 6**

Process 6 creates the same acquisition, preparation, gate, descriptive, AT1, and AT2 files listed in Processes 1–5.

---

<a id="reference-run"></a>

### Process 6.3 — REFERENCE RUN

> 🟣 **COMPARISON ONLY — DO NOT EDIT OUTPUTS TO MATCH**  
> The values below come from the accepted successful run of the current <code>analysis.py</code> and <code>modeling.py</code>. Use them only as comparison values for the supplied Frozen snapshot. A Live run may produce different values.

| Check | REFERENCE RUN value | Verify in |
|---|---:|---|
| Observed-target rows | 10,047 | <code>analysis_summary.json</code> |
| AT1 model-ready rows | 8,222 | <code>analysis_summary.json</code> |
| AT1 TRAIN / VALIDATION / TEST rows | 3,657 / 2,153 / 2,412 | <code>modeling_summary.json</code> |
| Selected AT1 model | <code>RF_WITH_HOTSPOTS</code> | <code>modeling_summary.json</code> |
| Validation MAE | 5.822 µg/m³ | <code>validation_model_summary.csv</code> |
| Locked-test MAE | 4.369 µg/m³ | <code>test_results.csv</code> |
| Locked-test RMSE | 7.535 µg/m³ | <code>test_results.csv</code> |
| Locked-test R² | 0.891 | <code>test_results.csv</code> |
| AT2 model-ready rows | 35 | <code>analysis_summary.json</code> |
| AT2 Spearman result | ρ = −0.655; p < 0.001 | <code>pm25_respi.csv</code> |
| AT2 adjusted OLS-HC3 result | coefficient = +12.98; 95% CI −30.59 to +56.55; p = 0.544 | <code>pm25_respi.csv</code> |
| Descriptive outputs from <code>analysis.py</code> | 8 tables and 13 figures | <code>analysis_manifest.csv</code> |

> 🟣 **How to use the REFERENCE RUN**  
> First confirm that every script completed and the feasibility gate has zero failures. Then compare the generated summaries and CSV files with this table. A numerical difference is a review signal only. Never edit an output CSV to force it to match the REFERENCE RUN.

**Result of Process 6**

> ✅ Process 6 produces the complete result from the **Frozen pipeline**: verified raw inputs, prepared datasets, quality checks, descriptive outputs, AT1 and AT2 outputs, and the fitted model.

---

### Troubleshooting

| Process | Problem | What to do |
|---:|---|---|
| 0 | A script is missing | Restore the exact filename under <code>src/</code> |
| 0 | Package import fails | Activate <code>.venv</code> and reinstall <code>requirements.txt</code> |
| 1 | OpenAQ returns <code>401</code> or <code>403</code> | Check or replace <code>OPENAQ_API_KEY</code> |
| 1 | FIRMS rejects the request | Check <code>NASA_FIRMS_MAP_KEY</code> and API limits |
| 1 | A public download returns an HTML error page | Download the exact source file manually and use <code>DOWNLOAD_DIR</code> |
| 2 | Duplicate, foreign-key, or merge error | Inspect the named quality CSV and fix the upstream parser/input |
| 3 | Gate returns <code>NOT YET FEASIBLE</code> | Open <code>feasibility_report.md</code> and resolve every failure |
| 4 | A figure is missing or blank | Fix the first <code>analysis.py</code> error and rerun Process 4 |
| 5 | Modeling input is missing | Rerun Process 4, then Process 5 |
| 6 | The run started in Live mode | Unset <code>FETCH_LIVE</code> and restart Process 6 |

### References

#### A. Data and API sources

- [OpenAQ documentation](https://docs.openaq.org/)
- [OpenAQ account registration](https://explore.openaq.org/register)
- [OpenAQ API-key guide](https://docs.openaq.org/using-the-api/api-key)
- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
- [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api)
- [NASA FIRMS API](https://firms.modaps.eosdis.nasa.gov/api/)
- [NASA FIRMS MAP_KEY request](https://firms.modaps.eosdis.nasa.gov/api/map_key/)
- [DDC ODPC1 dataset](https://opendata.ddc.moph.go.th/dataset/odpc1-01)
- [DOPA population statistics](https://stat.bora.dopa.go.th/new_stat/webPage/statByYear.php)
- [MOPH HCODE](https://hcode.moph.go.th/code/)
- [Healthcare Accreditation Institute dataset](https://data.ha.or.th/dataset/hospital/resource/4e20e752-25f8-468c-b155-33be7aecc0d4)
- [Thailand provincial boundaries](https://data.thailand.opendevelopmentmekong.net/th/dataset/thailand-provincial-boundaries)

#### B. Runtime and technical sources

- <code>requirements.txt</code> — required Python package versions.
- <code>src/*.py</code> — executable source and the authoritative command order.
- <code>outputs/metadata/fetch_manifest.csv</code> — source-file inventory and hashes.
- <code>outputs/quality/processing_decisions.csv</code> — preparation decisions.
- <code>outputs/quality/analytical_decisions.csv</code> — analytical-data decisions.
- <code>outputs/analysis/analysis_manifest.csv</code> — descriptive-output inventory.
- <code>outputs/modeling/modeling_manifest.csv</code> — modeling-output inventory.

## AI disclosure

Generative AI was used to organize and edit this run guide and to cross-check command order, filenames, inputs, outputs, and pass/stop rules against the current scripts. AI did not provide API keys, change raw observations, or replace execution of the code. Users should decide whether a run passed from the generated status files, quality checks, manifests, and script exit results.

## Report

PDF file report: report/report.pdf
PDF supplement tables and figures: report/supplement.pdf
VDO presentation: report/presentation.mp4
Youtube link: report/youtube_link.txt
VDO unlisted youtube link: https://youtu.be/6GcEJrkdypA?si=HnrfhERILaRKIuPM
Reseracher student ID: report/student_check.txt
