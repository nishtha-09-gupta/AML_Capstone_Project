"""Small provisional discrete BN; fitting is always an explicit caller action.

The graph is disease -> each feature, plus oldpeak -> slope. Inference sums
the oldpeak/slope branch exactly; all other missing leaf nodes sum to one.
Directions are statistical modeling assumptions, not medical causal claims.
See docs/bn_design.md for edge evidence and unverified independences. The app
loads frozen fitted instances; importing this module never trains a model.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from advanced_modeling import CATEGORY_OPTIONS, FEATURES, NUMERIC_FEATURES, clean_features


def validate_graph(edges):
    """Check node coverage, duplicate edges and acyclicity without a graph library."""
    nodes = ["disease"] + FEATURES
    parents = {node: [] for node in nodes}
    seen = set()
    for parent, child in edges:
        if parent not in parents or child not in parents or parent == child:
            raise ValueError("Invalid graph edge")
        if (parent, child) in seen:
            raise ValueError("Duplicate graph edge")
        seen.add((parent, child))
        parents[child].append(parent)
    if set(node for edge in edges for node in edge) != set(nodes):
        raise ValueError("Graph must cover disease and all eleven features")
    remaining = set(nodes)
    while remaining:
        ready = {node for node in remaining if not (set(parents[node]) & remaining)}
        if not ready:
            raise ValueError("Graph contains a cycle")
        remaining -= ready
    return parents


class HeartBayesianNetwork(ClassifierMixin, BaseEstimator):
    """Fit smoothed CPTs on supplied training rows and infer P(disease | evidence).

    Use allow_provisional=True explicitly for the unapproved graph. Fit accepts
    binary y (the caller derives num > 0); prediction accepts partial evidence.
    No dataset is read, split, or fitted automatically.
    """

    def __init__(self, alpha=1.0, allow_provisional=False):
        self.alpha = alpha
        self.allow_provisional = allow_provisional

    def _prepare(self, frame, fitting):
        if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
            raise ValueError("Supply a DataFrame with unique column names")
        if fitting and not set(FEATURES).issubset(frame.columns):
            raise ValueError("Training requires all eleven feature columns")
        frame = frame.reindex(columns=FEATURES).copy()
        for name in NUMERIC_FEATURES:
            frame[name] = pd.to_numeric(frame[name], errors="raise")
            values = frame[name].dropna().to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"{name}: non-finite evidence")
        for name, options in CATEGORY_OPTIONS.items():
            values = frame[name]
            if name in {"fbs", "exang"}:
                valid = values.isna() | values.astype(str).str.strip().str.lower().isin(
                    ["true", "false", "1", "0", "yes", "no"]
                )
            else:
                valid = values.isna() | values.isin(options)
            if not valid.all():
                raise ValueError(f"{name}: unrecognized category; use a missing value for unavailable evidence")
        return clean_features(frame)

    def _encode(self, frame, fitting=False):
        frame = self._prepare(frame, fitting)
        encoded = {}
        for name in FEATURES:
            values = frame[name]
            if name in NUMERIC_FEATURES:
                if fitting:
                    if values.isna().all():
                        raise ValueError(f"{name}: all training values are missing")
                    self.fill_values_[name] = float(values.median())
                    completed = values.fillna(self.fill_values_[name]).to_numpy(dtype=float)
                    cuts = np.unique(np.quantile(completed, [1 / 3, 2 / 3]))
                    # Remove empty extreme bins and collapse constant columns to one state.
                    self.bin_edges_[name] = cuts[(cuts > completed.min()) & (cuts < completed.max())]
                    self.states_[name] = list(range(len(self.bin_edges_[name]) + 1))
                codes = np.searchsorted(self.bin_edges_[name], values.to_numpy(dtype=float), side="right")
                codes[values.isna().to_numpy()] = -1
                if fitting:
                    codes[codes < 0] = np.searchsorted(self.bin_edges_[name], self.fill_values_[name], side="right")
            else:
                self_states = list(CATEGORY_OPTIONS[name]) if fitting else self.states_[name]
                if fitting:
                    self.states_[name] = self_states
                codes = values.map({value: i for i, value in enumerate(self_states)}).fillna(-1).to_numpy(dtype=int, copy=True)
                if fitting:
                    if (codes < 0).all():
                        raise ValueError(f"{name}: all training values are missing")
                    # First declared category wins a mode tie; no unseen state is discarded.
                    mode = int(np.bincount(codes[codes >= 0], minlength=len(self_states)).argmax())
                    self.fill_values_[name] = self_states[mode]
                    codes[codes < 0] = mode
            encoded[name] = codes
        return encoded

    def fit(self, frame, y):
        graph = json.loads((Path(__file__).parent / "config/bn_graph.json").read_text())
        if not graph["approved"] and not self.allow_provisional:
            raise ValueError("Graph is not approved; explicitly set allow_provisional=True for provisional use")
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        self.parents_ = validate_graph(graph["edges"])
        expected = {node: ["disease"] for node in FEATURES}
        expected.update(disease=[], slope=["disease", "oldpeak"])
        if self.parents_ != expected:
            raise ValueError("Exact branch inference requires the documented provisional topology and parent order")
        target = np.asarray(y)
        if target.ndim != 1 or len(target) != len(frame) or len(target) == 0 or not np.isin(target, [0, 1]).all():
            raise ValueError("y must be a nonempty, aligned binary target containing only 0 and 1")
        if isinstance(y, pd.Series) and not y.index.equals(frame.index):
            raise ValueError("Feature and target indices do not align")
        self.states_ = {"disease": [0, 1]}
        self.bin_edges_, self.fill_values_ = {}, {}
        encoded = self._encode(frame, fitting=True)
        encoded["disease"] = target.astype(int)
        # P(Y,X) = P(Y) * product_{i != slope} P(X_i|Y) * P(slope|Y,oldpeak).
        # Training imputation precedes counting; this is not missing-data EM.
        self.cpds_ = {}
        for node, parents in self.parents_.items():
            shape = tuple(len(self.states_[name]) for name in [node] + parents)
            # Axes are child then ordered parents; alpha is per cell, not total ESS.
            counts = np.full(shape, float(self.alpha))
            np.add.at(counts, tuple(encoded[name] for name in [node] + parents), 1)
            self.cpds_[node] = counts / counts.sum(axis=0, keepdims=True)
        self.classes_ = np.array([0, 1])
        self.graph_ = graph
        return self

    def predict_proba(self, frame):
        """Marginalize missing nodes; do not impute prediction-time evidence."""
        if not hasattr(self, "cpds_"):
            raise ValueError("Fit the Bayesian Network before predicting")
        encoded = self._encode(frame)
        probabilities = []
        for row in range(len(frame)):
            log_weights = np.log(self.cpds_["disease"].copy())
            for disease in [0, 1]:
                for node in FEATURES:
                    state = encoded[node][row]
                    if node not in {"oldpeak", "slope"} and state >= 0:
                        log_weights[disease] += np.log(self.cpds_[node][state, disease])
                # Sum_o P(o|Y) P(observed slope|Y,o); restrict o if observed.
                # Missing slope contributes 1 after summing its normalized CPT.
                branch = self.cpds_["oldpeak"][:, disease].copy()
                oldpeak, slope = encoded["oldpeak"][row], encoded["slope"][row]
                if oldpeak >= 0:
                    branch[np.arange(len(branch)) != oldpeak] = 0
                if slope >= 0:
                    branch *= self.cpds_["slope"][slope, disease, :]
                log_weights[disease] += np.log(branch.sum())
            weights = np.exp(log_weights - log_weights.max())
            probabilities.append(weights / weights.sum())
        return np.asarray(probabilities).reshape(-1, 2)

    def query(self, evidence):
        """Return the class-1 posterior for a dictionary of observed features."""
        if set(evidence) - set(FEATURES):
            raise ValueError("Evidence can contain only the eleven eligible features")
        return float(self.predict_proba(pd.DataFrame([evidence]))[0, 1])

    def predict(self, frame):
        return (self.predict_proba(frame)[:, 1] >= 0.5).astype(int)
