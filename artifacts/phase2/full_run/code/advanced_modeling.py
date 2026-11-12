"""Shared, deployment-safe feature definitions for the heart disease project."""

from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalch", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CATEGORY_OPTIONS = {
    "sex": ["Female", "Male"],
    "cp": ["typical angina", "atypical angina", "non-anginal", "asymptomatic"],
    "fbs": [False, True],
    "restecg": ["normal", "st-t abnormality", "lv hypertrophy"],
    "exang": [False, True],
    "slope": ["upsloping", "flat", "downsloping"],
}


def deduplicate_records(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop repeated source records while ignoring their arbitrary row ID."""
    return frame.drop_duplicates(subset=frame.columns.difference(["id"])).copy()


def clean_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the same missing-value conventions during training and inference."""
    forbidden = set(FEATURES) & {"id", "dataset", "num", "ca", "thal"}
    if forbidden:
        raise ValueError("Forbidden predictors: " + ", ".join(sorted(forbidden)))
    result = frame[FEATURES].copy()
    for column in NUMERIC_FEATURES:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    for column in ("trestbps", "chol"):
        result[column] = result[column].replace(0, np.nan)
    for column in ("fbs", "exang"):
        result[column] = result[column].map(
            lambda value: (
                True if str(value).strip().lower() in {"true", "1", "yes"}
                else False if str(value).strip().lower() in {"false", "0", "no"}
                else np.nan
            )
        ).astype(object)
    for column in ("sex", "cp", "restecg", "slope"):
        result[column] = result[column].where(result[column].notna(), np.nan)
    return result


def make_preprocessor():
    # Data checks and EDA need pandas, but do not need scikit-learn.
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        sparse_threshold=0,
    )


def validate_features(frame: pd.DataFrame) -> list[str]:
    """Return readable data-entry errors before a batch is scored."""
    errors = []
    missing = [column for column in FEATURES if column not in frame.columns]
    if missing:
        return ["Missing columns: " + ", ".join(missing)]
    if frame["age"].isna().any():
        errors.append(f"age: {int(frame['age'].isna().sum())} missing value(s); age is required")
    for column in ("age", "trestbps", "chol", "thalch", "oldpeak"):
        values = pd.to_numeric(frame[column], errors="coerce")
        bad = frame[column].notna() & values.isna()
        if bad.any():
            errors.append(f"{column}: {int(bad.sum())} non-numeric value(s)")
    limits = {"age": (18, 100), "trestbps": (0, 260), "chol": (0, 700), "thalch": (40, 240), "oldpeak": (-3, 10)}
    for column, (low, high) in limits.items():
        values = pd.to_numeric(frame[column], errors="coerce")
        bad = values.notna() & ~values.between(low, high)
        if bad.any():
            errors.append(f"{column}: {int(bad.sum())} value(s) outside {low}–{high}")
    for column, options in CATEGORY_OPTIONS.items():
        if column in {"fbs", "exang"}:
            normalized = frame[column].astype(str).str.strip().str.lower()
            valid = frame[column].isna() | normalized.isin(["true", "false", "1", "0", "yes", "no", "nan", "none", ""])
        else:
            valid = frame[column].isna() | frame[column].isin(options)
        if (~valid).any():
            errors.append(f"{column}: {int((~valid).sum())} unrecognized value(s)")
    return errors


def validate_dataset(frame: pd.DataFrame) -> None:
    """Reject malformed source data before deriving labels or fitting a model."""
    if frame.empty:
        raise ValueError("Dataset is empty")
    if not frame.columns.is_unique:
        raise ValueError("Dataset has duplicate column names")
    required = FEATURES + ["id", "dataset", "num", "ca", "thal"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("Missing dataset columns: " + ", ".join(missing))
    target = frame["num"]
    boolean_target = target.map(lambda value: isinstance(value, (bool, np.bool_)))
    if target.isna().any() or boolean_target.any() or not target.isin([0, 1, 2, 3, 4]).all():
        raise ValueError("num must contain only numeric values 0, 1, 2, 3, 4; no missing values or booleans")
    if frame["id"].isna().any() or not frame["id"].is_unique:
        raise ValueError("Source row IDs must be non-missing and unique")
    if frame["dataset"].isna().any() or frame["dataset"].astype(str).str.strip().eq("").any():
        raise ValueError("Hospital names in dataset must be non-missing and non-blank")
    errors = validate_features(frame)
    if errors:
        raise ValueError("Invalid dataset features: " + "; ".join(errors))


def split_records(frame: pd.DataFrame, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split already-deduplicated records using the existing stratified 80/20 rule."""
    from sklearn.model_selection import train_test_split

    validate_dataset(frame)
    if not frame.index.is_unique:
        raise ValueError("Row indices must be unique to align features, labels and source IDs")
    train, test = train_test_split(
        frame, test_size=0.2, stratify=(frame["num"] > 0).astype(int), random_state=random_state
    )
    return train, test
