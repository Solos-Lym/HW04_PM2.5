# Analysis decisions for Methods and Discussion

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
