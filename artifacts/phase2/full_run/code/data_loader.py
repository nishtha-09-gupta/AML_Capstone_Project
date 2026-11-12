"""Data ingestion, clinical cleaning, and schema validation for the UCI Heart Disease dataset."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# Core feature definitions
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalch", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "num"
ID_COL = "id"
SITE_COL = "dataset"
EXCLUDED_COLS = ["ca", "thal"]  # Excluded due to severe cross-hospital missingness

CATEGORY_OPTIONS = {
    "sex": ["Female", "Male"],
    "cp": ["typical angina", "atypical angina", "non-anginal", "asymptomatic"],
    "fbs": [False, True],
    "restecg": ["normal", "st-t abnormality", "lv hypertrophy"],
    "exang": [False, True],
    "slope": ["upsloping", "flat", "downsloping"],
}

CLINICAL_BOUNDS = {
    "age": (18, 100),
    "trestbps": (0, 260),
    "chol": (0, 700),
    "thalch": (40, 240),
    "oldpeak": (-3, 10),
}

DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "heart_disease_uci.csv"


def load_dataset(filepath: str | Path | None = None) -> pd.DataFrame:
    """Load raw Heart Disease UCI dataset from CSV into a pandas DataFrame."""
    path = Path(filepath) if filepath else DEFAULT_DATA_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)


def deduplicate_records(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate clinical records while ignoring the arbitrary source row ID."""
    feature_cols = [c for c in frame.columns if c != ID_COL]
    return frame.drop_duplicates(subset=feature_cols).copy()


def clean_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply standard missingness and physiological sanity conventions.

    Converts zero blood pressure (trestbps) and zero serum cholesterol (chol)
    readings to NaN, as 0 values in the UCI registry represent unrecorded entries
    rather than true physiological measurements.
    """
    forbidden = set(FEATURES) & {ID_COL, SITE_COL, TARGET, "ca", "thal"}
    if forbidden:
        raise ValueError("Forbidden predictors: " + ", ".join(sorted(forbidden)))

    result = frame[FEATURES].copy()

    for col in NUMERIC_FEATURES:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    # Treat physiologically impossible zeros as missing
    for col in ("trestbps", "chol"):
        result[col] = result[col].replace(0, np.nan)

    # Standardize boolean flags
    for col in ("fbs", "exang"):
        result[col] = result[col].map(
            lambda v: (
                True if str(v).strip().lower() in {"true", "1", "yes"}
                else False if str(v).strip().lower() in {"false", "0", "no"}
                else np.nan
            )
        ).astype(object)

    for col in ("sex", "cp", "restecg", "slope"):
        result[col] = result[col].where(result[col].notna(), np.nan)

    return result


def validate_features(frame: pd.DataFrame) -> list[str]:
    """Inspect feature values against clinical boundaries and category sets."""
    errors: list[str] = []
    missing_cols = [col for col in FEATURES if col not in frame.columns]
    if missing_cols:
        return ["Missing feature columns: " + ", ".join(missing_cols)]

    if frame["age"].isna().any():
        errors.append(f"age: {int(frame['age'].isna().sum())} missing value(s); age is required")

    for col, (low, high) in CLINICAL_BOUNDS.items():
        vals = pd.to_numeric(frame[col], errors="coerce")
        bad = vals.notna() & ~vals.between(low, high)
        if bad.any():
            errors.append(f"{col}: {int(bad.sum())} value(s) outside {low}–{high}")

    for col, options in CATEGORY_OPTIONS.items():
        if col in {"fbs", "exang"}:
            normalized = frame[col].astype(str).str.strip().str.lower()
            valid = frame[col].isna() | normalized.isin(
                ["true", "false", "1", "0", "yes", "no", "nan", "none", ""]
            )
        else:
            valid = frame[col].isna() | frame[col].isin(options)
        if (~valid).any():
            errors.append(f"{col}: {int((~valid).sum())} unrecognized value(s)")

    return errors


def validate_dataset(frame: pd.DataFrame) -> None:
    """Ensure raw dataset strictly adheres to schema and domain contracts."""
    if frame.empty:
        raise ValueError("Dataset is empty")
    if not frame.columns.is_unique:
        raise ValueError("Dataset has duplicate column names")

    required = FEATURES + [ID_COL, SITE_COL, TARGET, "ca", "thal"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError("Missing required dataset columns: " + ", ".join(missing))

    target = frame[TARGET]
    boolean_target = target.map(lambda v: isinstance(v, (bool, np.bool_)))
    if target.isna().any() or boolean_target.any() or not target.isin([0, 1, 2, 3, 4]).all():
        raise ValueError("Target 'num' must contain only integers in range [0, 4] with no missing values")

    if frame[ID_COL].isna().any() or not frame[ID_COL].is_unique:
        raise ValueError("Source patient row IDs must be unique and non-missing")

    if frame[SITE_COL].isna().any() or frame[SITE_COL].astype(str).str.strip().eq("").any():
        raise ValueError("Hospital site labels must be non-empty strings")

    feature_errors = validate_features(frame)
    if feature_errors:
        raise ValueError("Invalid dataset features: " + "; ".join(feature_errors))


def load_clean_dataset(filepath: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, validate, deduplicate, and clean the UCI Heart Disease dataset.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - dedup_data: Full validated dataset after deduplication (918 rows)
            - cleaned_features: Predictor matrix with standardized missingness conventions
    """
    raw = load_dataset(filepath)
    validate_dataset(raw)
    dedup_data = deduplicate_records(raw)
    cleaned = clean_features(dedup_data)
    return dedup_data, cleaned


if __name__ == "__main__":
    raw = load_dataset()
    validate_dataset(raw)
    dedup, cleaned = load_clean_dataset()
    print("=" * 60)
    print("Heart Disease UCI Dataset Validation & Cleaning Passed")
    print("=" * 60)
    print(f"Raw Records:          {len(raw)}")
    print(f"Duplicates Removed:   {len(raw) - len(dedup)}")
    print(f"Validated Analysis:   {len(dedup)} records across {dedup[SITE_COL].nunique()} clinical sites")
    print(f"Binary Target:        Positive={int((dedup[TARGET] > 0).sum())}, Negative={int((dedup[TARGET] == 0).sum())}")
    print(f"Zero BP Replaced:     {int(cleaned['trestbps'].isna().sum())} missing after cleaning")
    print(f"Zero Chol Replaced:   {int(cleaned['chol'].isna().sum())} missing after cleaning")
    print("=" * 60)
