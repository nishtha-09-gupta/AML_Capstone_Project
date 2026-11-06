"""Phase 1 selection and evaluation helpers; real-data execution requires opt-in.

S selects families/configurations, C selects cutoffs, and T is retrospective
scoring only. An authorized exploratory run writes a unique new directory;
legacy artifacts are never overwritten. Synthetic tests use these same helpers.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, log_loss, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from advanced_modeling import clean_features, deduplicate_records, make_preprocessor, split_records, validate_dataset

ROOT = Path(__file__).resolve().parent
PROTOCOL = json.loads((ROOT / "config/phase1_protocol.json").read_text())


def scores(y_true, probability, threshold=0.5):
    predicted = (np.asarray(probability) >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    tn, fp = matrix[0]
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "average_precision": float(average_precision_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, np.clip(probability, np.finfo(float).eps, 1 - np.finfo(float).eps), labels=[0, 1])),
        "confusion_matrix": matrix.tolist(),
    }


def prepare_partitions(raw):
    """Reconstruct historical D/T; do not fit anything or read/write artifacts."""
    validate_dataset(raw)
    return split_records(deduplicate_records(raw), random_state=PROTOCOL["partitions"]["historical_split"]["random_state"])


def make_candidates(*, include_bn=True, allow_provisional=False, protocol=PROTOCOL):
    """Keep the existing four models/grids; require an explicit resolved BN grid."""
    grids = protocol["models"]["grids"]
    if include_bn and grids["Bayesian network"] is None:
        raise ValueError("BN search grid is unresolved; comparative experiments remain blocked")
    # Optional import: data-contract and isolation tests do not require XGBoost.
    from xgboost import XGBClassifier

    seed = protocol["models"]["estimator_random_state"]
    settings = protocol["models"]["baseline_fixed_settings"]
    classifiers = {
        "Logistic regression": LogisticRegression(random_state=seed, **settings["Logistic regression"]),
        "Random forest": RandomForestClassifier(random_state=seed, **settings["Random forest"]),
        "Histogram gradient boosting": HistGradientBoostingClassifier(random_state=seed, **settings["Histogram gradient boosting"]),
        "XGBoost": XGBClassifier(random_state=seed, **settings["XGBoost"]),
    }
    candidates = {
        name: (Pipeline([("preprocess", make_preprocessor()), ("model", model)]), grids[name])
        for name, model in classifiers.items()
    }
    if include_bn:
        from bayesian_network import HeartBayesianNetwork
        candidates["Bayesian network"] = (HeartBayesianNetwork(allow_provisional=allow_provisional), grids["Bayesian network"])
    return candidates


def positive_probability(model, x):
    probability = np.asarray(model.predict_proba(x), dtype=float)
    if not np.array_equal(model.classes_, [0, 1]) or probability.shape != (len(x), 2):
        raise ValueError("Expected probabilities in class order [0, 1]")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any() or not np.allclose(probability.sum(axis=1), 1):
        raise ValueError("Invalid predicted probabilities")
    return probability[:, 1]


def default_f1(model, x, y):
    return f1_score(y, positive_probability(model, x) >= 0.5, zero_division=0)


def probability_auc(model, x, y):
    return roc_auc_score(y, positive_probability(model, x))


def best_configuration(results):
    """Exact full-precision F1/AUC ties use canonical JSON, as documented."""
    f1, auc = results["mean_test_f1"], results["mean_test_roc_auc"]
    if not np.isfinite(f1).all() or not np.isfinite(auc).all():
        raise ValueError("Undefined selection score; do not silently drop candidates")
    return min(range(len(f1)), key=lambda i: (
        -f1[i], -auc[i], json.dumps(results["params"][i], sort_keys=True, separators=(",", ":"), allow_nan=False)
    ))


def stratified_cv(y, count, seed):
    if set(y.unique()) != {0, 1} or y.value_counts().min() < count:
        raise ValueError("Not estimable: insufficient class support for fixed stratified folds")
    return StratifiedKFold(n_splits=count, shuffle=True, random_state=seed)


def search_model(estimator, grid, x, y, count, seed):
    started = time.perf_counter()
    search = GridSearchCV(
        estimator, grid, cv=stratified_cv(y, count, seed),
        scoring={"f1": default_f1, "roc_auc": probability_auc},
        refit=best_configuration, n_jobs=1, error_score="raise",
    )
    search.fit(x, y)
    search.elapsed_seconds_ = time.perf_counter() - started
    return search


def choose_threshold(y, probability):
    """C-only F1 objective; nearest to 0.50, then lower index breaks ties."""
    values = [float(f1_score(y, probability >= k / 100, zero_division=0)) for k in range(101)]
    selected = min(range(101), key=lambda k: (-values[k], abs(k - 50), k))
    return {"threshold": selected / 100, "grid_index": selected,
            "scores": values, "maxima": [k for k, value in enumerate(values) if value == values[selected]]}


def develop(data, candidates, *, forbidden_ids=(), seeds=None, protocol=PROTOCOL, progress=None):
    """Select only on S, refit only on S, then choose each frozen model's C cutoff.

    candidates maps family name to (unfitted estimator/pipeline, parameter grid).
    Explicit small grids in synthetic tests do not resolve the real BN grid.
    """
    validate_dataset(data)
    if not data.index.is_unique or set(data["id"]) & set(forbidden_ids):
        raise ValueError("Development rows overlap excluded rows or have duplicate indices")
    order = protocol["selection"]["fixed_family_tie_order"]
    if not candidates or any(name not in order or grid is None for name, (_, grid) in candidates.items()):
        raise ValueError("Every declared family needs a resolved grid and protocol name")
    seeds = seeds or {
        "split": protocol["partitions"]["development_split"]["random_state"],
        "outer": protocol["selection"]["outer_cv"]["random_state"],
        "inner": protocol["selection"]["inner_cv"]["random_state"],
        "final_search": protocol["selection"]["final_search_cv"]["random_state"],
    }
    selection, cutoff = train_test_split(
        data, test_size=protocol["partitions"]["development_split"]["threshold_size"],
        stratify=(data["num"] > 0).astype(int), random_state=seeds["split"],
    )
    x, y = clean_features(selection), (selection["num"] > 0).astype(int)
    cutoff_y = (cutoff["num"] > 0).astype(int)
    if set(cutoff_y.unique()) != {0, 1}:
        raise ValueError("Not estimable: cutoff partition requires both classes")
    outer = list(stratified_cv(y, protocol["selection"]["outer_cv"]["n_splits"], seeds["outer"]).split(x, y))
    comparisons, oof, searches = [], [], []
    for name, (estimator, grid) in candidates.items():
        fold_f1, fold_auc = [], []
        for fold, (train_idx, valid_idx) in enumerate(outer):
            search = search_model(estimator, grid, x.iloc[train_idx], y.iloc[train_idx],
                                  protocol["selection"]["inner_cv"]["n_splits"], seeds["inner"])
            probability = positive_probability(search.best_estimator_, x.iloc[valid_idx])
            fold_f1.append(float(f1_score(y.iloc[valid_idx], probability >= 0.5, zero_division=0)))
            fold_auc.append(float(roc_auc_score(y.iloc[valid_idx], probability)))
            searches.append({"model": name, "fold": fold,
                             "train_source_ids": selection.iloc[train_idx]["id"].tolist(),
                             "results": search.cv_results_, "elapsed_seconds": search.elapsed_seconds_})
            if progress:
                progress(f"{name}: outer fold {fold + 1}/{len(outer)} completed in {search.elapsed_seconds_:.2f}s")
            oof.extend({"model": name, "fold": fold, "source_id": source_id,
                        "target": int(target), "probability": float(p)}
                       for source_id, target, p in zip(selection.iloc[valid_idx]["id"], y.iloc[valid_idx], probability))
        comparisons.append({"model": name, "cv_f1": float(np.mean(fold_f1)),
                            "cv_auc": float(np.mean(fold_auc)), "fold_f1": fold_f1, "fold_auc": fold_auc})
    selected = min(comparisons, key=lambda row: (-row["cv_f1"], -row["cv_auc"], order.index(row["model"])))
    models = {}
    # Final searches happen after family selection; C does not participate.
    for name, (estimator, grid) in candidates.items():
        search = search_model(estimator, grid, x, y,
                              protocol["selection"]["final_search_cv"]["n_splits"], seeds["final_search"])
        models[name] = search.best_estimator_
        searches.append({"model": name, "fold": None, "train_source_ids": selection["id"].tolist(),
                         "results": search.cv_results_, "elapsed_seconds": search.elapsed_seconds_})
        if progress:
            progress(f"{name}: final S-only search/refit completed in {search.elapsed_seconds_:.2f}s")
    cutoffs, cutoff_predictions = {}, {}
    for name, model in models.items():
        probability = positive_probability(model, clean_features(cutoff))
        cutoffs[name] = choose_threshold(cutoff_y, probability)
        cutoff_predictions[name] = probability.tolist()
    return {"selected_model": selected["model"], "comparison": comparisons, "models": models,
            "thresholds": cutoffs, "cutoff_probabilities": cutoff_predictions,
            "selection_source_ids": selection["id"].tolist(), "cutoff_source_ids": cutoff["id"].tolist(),
            "cutoff_targets": cutoff_y.tolist(), "oof": oof, "searches": searches, "frozen": True}


def evaluate_frozen(data, result, *, label=PROTOCOL["evaluation"]["final_label"]):
    """Score already fitted/frozen candidates; never select or fit a model here."""
    validate_dataset(data)
    used = set(result["selection_source_ids"]) | set(result["cutoff_source_ids"])
    if not result["frozen"] or used & set(data["id"]):
        raise ValueError("Evaluation must be disjoint from development and use frozen decisions")
    x, y = clean_features(data), (data["num"] > 0).astype(int)
    if set(y.unique()) != {0, 1}:
        raise ValueError("Not estimable: evaluation requires both classes")
    families = {}
    for name, model in result["models"].items():
        probability = positive_probability(model, x)
        families[name] = {"probability": probability.tolist(), "default": scores(y, probability, 0.5),
                          "tuned": scores(y, probability, result["thresholds"][name]["threshold"])}
    return {"label": label, "selected_model": result["selected_model"],
            "source_ids": data["id"].tolist(), "target": y.tolist(), "families": families}


def evaluate_hospitals(development, candidates, *, retrospective_ids, protocol=PROTOCOL):
    """D-only hospital holdouts; select each winner using its other hospitals."""
    validate_dataset(development)
    if set(development["id"]) & set(retrospective_ids):
        raise ValueError("Hospital experiments cannot contain retrospective test rows")
    if set(development["dataset"]) != set(protocol["hospital_evaluation"]["site_order"]):
        raise ValueError("Expected all four source hospitals in development")
    results = {}
    for site in protocol["hospital_evaluation"]["site_order"]:
        held = development.loc[development["dataset"] == site]
        other = development.loc[development["dataset"] != site]
        local = develop(other, candidates, forbidden_ids=list(retrospective_ids) + held["id"].tolist(),
                        seeds=protocol["hospital_evaluation"]["seed_map"][site], protocol=protocol)
        results[site] = {"development": local,
                         "evaluation": evaluate_frozen(held, local, label="D-only held-out hospital: " + site)}
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Phase 1 exploratory experiment; guarded by default")
    parser.add_argument("--run-experiment", action="store_true")
    parser.add_argument("--allow-provisional-bn", action="store_true")
    args = parser.parse_args()
    if not (args.run_experiment and args.allow_provisional_bn):
        raise SystemExit("Real-data training is disabled by default. An authorized exploratory run requires both --run-experiment and --allow-provisional-bn.")
    from experiment import run_experiment
    run_experiment(allow_provisional=True)


if __name__ == "__main__":
    main()
