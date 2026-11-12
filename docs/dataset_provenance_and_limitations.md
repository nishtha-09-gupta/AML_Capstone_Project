# Dataset provenance, missingness, and patient-linkage limitations

## Source and provenance

The project attributes its data to the [UCI Machine Learning Repository Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease), DOI [10.24432/C52P4X](https://doi.org/10.24432/C52P4X). UCI records the dataset as donated in 1988, identifies four source databases—Cleveland, Hungary, Switzerland, and VA Long Beach—and cites Janosi, Steinbrunn, Pfisterer, and Detrano (1989). The original task records `num` as angiographic disease status from 0 through 4; this project maps `num == 0` to negative and `num > 0` to positive. This is classification of recorded disease presence, not prediction of future cardiovascular risk.

The repository's `heart_disease_uci.csv` has 920 rows and 16 columns: `id`, 11 selected predictors, `dataset`, `ca`, `thal`, and `num`. Its `dataset` field identifies the four source labels. The row counts in this supplied file are Cleveland 303, Hungary 294, Switzerland 123, and VA Long Beach 200. These are facts verified from the checked-in CSV. The repository does not contain a complete acquisition and merge log that establishes precisely how this particular CSV was assembled from the UCI source files; attribution to UCI should not be read as verification of that transformation history.

## Dataset composition and audit counts

The audit below is reproducible from the checked-in CSV using `deduplicate_records` and `clean_features` in `data_loader.py`. Deduplication compares all columns except `id` and retains the first exact match. Two rows are removed, leaving 918 analysis rows: Cleveland 303, Hungary 293, Switzerland 123, and VA Long Beach 199. The original target distribution in the CSV is `num=0`: 411, `num=1`: 265, `num=2`: 109, `num=3`: 107, and `num=4`: 28. After the two removed rows, the binary analysis labels are 410 negative and 508 positive.

The feature missingness table reports the 918-row, deduplicated data after the code's documented conversion of zero `trestbps` and `chol` values to missing. The counts by hospital use the same cleaned features. Percentages use 918 as the overall denominator and each site's row count as its site denominator.

| Predictor | Missing, all sites | Percent | Cleveland | Hungary | Switzerland | VA Long Beach |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `age` | 0 | 0.00% | 0 | 0 | 0 | 0 |
| `trestbps` | 60 | 6.54% | 0 | 1 | 2 | 57 |
| `chol` | 201 | 21.90% | 0 | 22 | 123 | 56 |
| `thalch` | 55 | 5.99% | 0 | 1 | 1 | 53 |
| `oldpeak` | 62 | 6.75% | 0 | 0 | 6 | 56 |
| `sex` | 0 | 0.00% | 0 | 0 | 0 | 0 |
| `cp` | 0 | 0.00% | 0 | 0 | 0 | 0 |
| `fbs` | 90 | 9.80% | 0 | 8 | 75 | 7 |
| `restecg` | 2 | 0.22% | 0 | 1 | 1 | 0 |
| `exang` | 55 | 5.99% | 0 | 1 | 1 | 53 |
| `slope` | 307 | 33.44% | 0 | 189 | 17 | 101 |

The source CSV contains explicit nulls as well as sentinel-like zero values. The next table counts explicit nulls in the original 920-row CSV; it does not treat zeros as missing.

| Original CSV field | Explicit nulls (920 rows) |
| --- | ---: |
| `age` | 0 |
| `trestbps` | 59 |
| `chol` | 30 |
| `thalch` | 55 |
| `oldpeak` | 62 |
| `sex` | 0 |
| `cp` | 0 |
| `fbs` | 90 |
| `restecg` | 2 |
| `exang` | 55 |
| `slope` | 309 |
| `ca` | 611 |
| `thal` | 486 |

Before deduplication there is one zero `trestbps` and 172 zero `chol` values. `clean_features` converts these to `NaN`, giving cleaned missingness totals of 60 and 201 respectively after deduplication. This is the project's data convention: these zero entries are treated as unrecorded measurements, not as observed physiological values. Other zero values, including valid `oldpeak` values, are not globally converted to missing.

The original `ca` and `thal` columns have 611 and 486 explicit missing values in the 920-row CSV; after deduplication, they have 609 and 484 missing values among 918 rows (66.34% and 52.72%). They are excluded from the 11-predictor model matrix because of this substantial missingness. This is a modeling choice, not evidence that the variables are clinically uninformative. Any alternative imputation or missingness-aware use would require a separate, training-fold-only evaluation.

## Preprocessing and excluded fields

The eligible predictors in `data_loader.py` are exactly five numeric features—`age`, `trestbps`, `chol`, `thalch`, `oldpeak`—and six categorical features—`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`.

| Field | Treatment | Reason in this project |
| --- | --- | --- |
| `id` | Excluded from model predictors; ignored when checking exact duplicate clinical rows | Row identifier, not a clinical feature. Unique values in this CSV do not establish unique people or cross-hospital identity. |
| `dataset` | Excluded from model predictors; retained for site-level grouping and evaluation | Source-site label. Using it as a predictor could encode cohort-specific differences; it is useful for descriptive and hospital-holdout analyses. |
| `ca` | Excluded | 609/918 missing after deduplication. |
| `thal` | Excluded | 484/918 missing after deduplication. |
| `num` | Used only to construct the binary outcome (`num > 0`) | Original target, not an input feature. The binary conversion collapses severity categories 1–4 into one positive class. |

`load_clean_dataset` validates the raw schema and target, removes exact duplicate rows while ignoring `id`, and then selects and cleans the eleven predictors. Downstream Phase 1 and Phase 2 model pipelines perform learned imputation, scaling, encoding, and related transformations inside training fits; those transformations must remain fold-local. The missingness counts above are descriptive full-dataset audit statistics, not values used to fit a predictive transformation.

## Patient linkage and independence limits

The supplied CSV contains a unique, sequential-looking `id` for each row, and `validate_dataset` checks that this field is nonmissing and unique. The code does not establish that these values are stable patient identifiers in the original hospital systems or that they link a person across the four source databases. The official UCI description notes that names and Social Security numbers were removed from the original database and replaced with dummy values; it does not provide a verified cross-hospital linkage key for this merged file.

The project removes two exact duplicate clinical rows when `id` is ignored. That prevents those exact repeated rows from appearing twice, but it cannot detect the same person recorded with different measurements, coding, or missingness. Nor can it identify a patient represented in more than one hospital file. It is therefore not justified to claim that all 918 retained rows are 918 different people or that every train/test fold is patient-disjoint.

This uncertainty affects the independence assumption behind row-level stratified cross-validation, confidence intervals, and bootstrap estimates. If related records cross partitions, the effective sample size may be smaller and uncertainty may be understated. Hospital-held-out analyses test some site-shift behavior but do not resolve patient overlap or establish independent cohorts. Results should be described as retrospective experiments on this supplied dataset, not prospective cardiovascular-risk estimates, patient-level external validation, or evidence of clinical utility.

## References

1. A. Janosi, W. Steinbrunn, M. Pfisterer, and R. Detrano, “Heart Disease,” UCI Machine Learning Repository, 1989, [doi:10.24432/C52P4X](https://doi.org/10.24432/C52P4X). The [UCI dataset record](https://archive.ics.uci.edu/dataset/45/heart+disease) documents the four databases, target meaning, missing values, and original data description.
2. R. Detrano et al., “International application of a new probability algorithm for the diagnosis of coronary artery disease,” *The American Journal of Cardiology*, 1989. This is listed by UCI as the introductory paper associated with the dataset; consult the UCI record for bibliographic metadata.

## Reproduction note

All repository-specific counts and tables in this document were recomputed from `heart_disease_uci.csv` with the existing `data_loader.py` functions. The corresponding saved audit is `artifacts/data_audit.json`; the source CSV and transformation code are the primary reproducible evidence. UCI provenance statements above are from the linked UCI record. No patient-level linkage or unrecorded source-history facts are inferred.
