"""Small pgmpy BN and Naive Bayes models using Phase 1's feature contract."""
import numpy as np
import pandas as pd
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

from advanced_modeling import FEATURES
from bayesian_network import HeartBayesianNetwork


class DiscreteHeartModel(HeartBayesianNetwork):
    """Domain scaffold, or its Naive Bayes ablation, with Dirichlet CPTs.

    Reuse Phase 1's training-only bins and missing-evidence semantics. pgmpy
    estimates the CPTs. The small graph permits vectorized exact marginalization;
    predict_pgmpy is the independent general-purpose inference reference.
    """
    def __init__(self, structured=True, alpha=1.0):
        self.structured = structured
        self.alpha = alpha

    def fit(self, frame, y):
        target = np.asarray(y)
        if (target.ndim != 1 or len(target) != len(frame)
                or set(np.unique(target)) != {0, 1}):
            raise ValueError("Training requires aligned binary labels with both classes")
        if isinstance(y, pd.Series) and not y.index.equals(frame.index):
            raise ValueError("Feature and target indices do not align")
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        self.states_ = {"disease": [0, 1]}
        self.bin_edges_, self.fill_values_ = {}, {}
        encoded = self._encode(frame, fitting=True)
        encoded["disease"] = target.astype(int)
        edges = [("disease", feature) for feature in FEATURES]
        if self.structured:
            edges.append(("oldpeak", "slope"))
        self.model_ = DiscreteBayesianNetwork(edges)
        self.model_.fit(pd.DataFrame(encoded), estimator=BayesianEstimator,
                        state_names={k: list(range(len(v))) for k, v in self.states_.items()},
                        prior_type="dirichlet",
                        pseudo_counts=self.alpha, n_jobs=1)
        self.model_.check_model()
        self.cpds_ = {cpd.variable: np.asarray(cpd.values) for cpd in self.model_.get_cpds()}
        self.classes_ = np.array([0, 1])
        self.fit_indices_ = frame.index.tolist()
        return self

    def predict_proba(self, frame):
        encoded = self._encode(frame)
        n = len(frame)
        log_p = np.tile(np.log(self.cpds_["disease"]), (n, 1))
        for name in FEATURES:
            if self.structured and name in {"oldpeak", "slope"}:
                continue
            observed = encoded[name] >= 0
            log_p[observed] += np.log(self.cpds_[name][encoded[name][observed]])
        if self.structured:
            for disease in [0, 1]:
                branch = np.tile(self.cpds_["oldpeak"][:, disease], (n, 1))
                oldpeak, slope = encoded["oldpeak"], encoded["slope"]
                branch *= (oldpeak[:, None] < 0) | (np.arange(branch.shape[1]) == oldpeak[:, None])
                observed = slope >= 0
                branch[observed] *= self.cpds_["slope"][slope[observed], disease, :]
                log_p[:, disease] += np.log(branch.sum(axis=1))
        weights = np.exp(log_p - log_p.max(axis=1, keepdims=True))
        return weights / weights.sum(axis=1, keepdims=True)

    def predict_pgmpy(self, frame):
        """Reference inference for checking fast inference, including missing data."""
        encoded = self._encode(frame)
        inference = VariableElimination(self.model_)
        return np.array([inference.query(["disease"], evidence={
            name: int(encoded[name][i]) for name in FEATURES if encoded[name][i] >= 0
        }, show_progress=False).values for i in range(len(frame))])

    def explanation(self):
        """JSON-ready graph, cut points, imputation values and all learned CPTs."""
        return {"structured": self.structured, "alpha_per_cell": self.alpha,
                "edges": list(self.model_.edges()), "states": self.states_,
                "cut_points": self.bin_edges_, "training_fill_values": self.fill_values_,
                "cpds": [{"variables": cpd.variables, "values": cpd.values}
                         for cpd in self.model_.get_cpds()],
                "interpretation": "Provisional statistical graph; not causal or clinically validated"}
