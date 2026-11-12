"""A two-hidden-layer MLP, optionally assisted by cross-fitted BN probabilities."""
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from advanced_modeling import make_preprocessor
from phase2_baselines import DiscreteHeartModel


def log_odds(probability):
    p = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def cross_fit_bn(x, y, *, structured=True, seed=1729):
    """Each score is produced without its row or label in the fitted BN."""
    probabilities = np.empty(len(x))
    folds = []
    for fit, valid in StratifiedKFold(3, shuffle=True, random_state=seed).split(x, y):
        model = DiscreteHeartModel(structured=structured).fit(x.iloc[fit], y.iloc[fit])
        probabilities[valid] = model.predict_proba(x.iloc[valid])[:, 1]
        folds.append({"train_indices": x.index[fit].tolist(),
                      "validation_indices": x.index[valid].tolist(),
                      "cut_points": model.bin_edges_})
    return probabilities, folds


class HeartMLP(ClassifierMixin, BaseEstimator):
    """CPU PyTorch classifier with training-only early stopping and optional BN.

    hybrid: 'none', 'bn', or 'nb'. The same architecture and optimizer are used
    in each case; hybrid adds just one learned input coefficient per first-layer
    unit. fit owns preprocessing to isolate even the early-stop subset.
    """
    def __init__(self, hybrid="none", hidden=(32, 16), dropout=0.2,
                 weight_decay=0.001, learning_rate=0.001, max_epochs=150,
                 patience=20, batch_size=32, random_state=42):
        self.hybrid = hybrid
        self.hidden = hidden
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.random_state = random_state

    def _prepare_fit(self, x, y):
        preprocessor = make_preprocessor()
        matrix = preprocessor.fit_transform(x)
        bundle = {"preprocessor": preprocessor, "fit_indices": x.index.tolist()}
        if self.hybrid != "none":
            p, folds = cross_fit_bn(x, y, structured=self.hybrid == "bn")
            scaler = StandardScaler().fit(log_odds(p))
            matrix = np.column_stack([matrix, scaler.transform(log_odds(p))])
            bundle.update(bn=DiscreteHeartModel(structured=self.hybrid == "bn").fit(x, y),
                          scaler=scaler, cross_fit_folds=folds)
        return np.asarray(matrix, dtype=np.float32), bundle

    def _transform(self, x, bundle):
        matrix = bundle["preprocessor"].transform(x)
        if self.hybrid != "none":
            p = bundle["bn"].predict_proba(x)[:, 1]
            matrix = np.column_stack([matrix, bundle["scaler"].transform(log_odds(p))])
        return np.asarray(matrix, dtype=np.float32)

    def _network(self, dimensions):
        layers = []
        for width in self.hidden:
            layers.extend([nn.Linear(dimensions, width), nn.ReLU(), nn.Dropout(self.dropout)])
            dimensions = width
        return nn.Sequential(*layers, nn.Linear(dimensions, 1))

    def _train(self, matrix, y, epochs, validation=None):
        # fork_rng keeps fitting from changing the caller's random state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.random_state)
            network = self._network(matrix.shape[1])
            optimizer = torch.optim.AdamW(network.parameters(), lr=self.learning_rate,
                                          weight_decay=self.weight_decay)
            loss_fn = nn.BCEWithLogitsLoss()
            x_tensor = torch.as_tensor(matrix)
            y_tensor = torch.as_tensor(np.asarray(y, dtype=np.float32)).reshape(-1, 1)
            history, best, best_epoch, stale = [], float("inf"), 1, 0
            for epoch in range(1, epochs + 1):
                network.train()
                for batch in torch.randperm(len(matrix)).split(self.batch_size):
                    optimizer.zero_grad()
                    loss = loss_fn(network(x_tensor[batch]), y_tensor[batch])
                    loss.backward()
                    optimizer.step()
                network.eval()
                with torch.no_grad():
                    train_loss = float(loss_fn(network(x_tensor), y_tensor))
                    row = {"epoch": epoch, "train_bce": train_loss}
                    if validation is not None:
                        vx, vy = validation
                        valid_loss = float(loss_fn(network(torch.as_tensor(vx)),
                            torch.as_tensor(np.asarray(vy, dtype=np.float32)).reshape(-1, 1)))
                        row["validation_bce"] = valid_loss
                        if valid_loss < best - 0.0001:
                            best, best_epoch, stale = valid_loss, epoch, 0
                        else:
                            stale += 1
                if not np.isfinite(list(row.values())).all():
                    raise ValueError("Nonfinite training loss")
                history.append(row)
                if validation is not None and stale >= self.patience:
                    break
        return network, history, best_epoch

    def fit(self, x, y):
        if self.hybrid not in {"none", "bn", "nb"}:
            raise ValueError("hybrid must be none, bn or nb")
        if (not isinstance(x, pd.DataFrame) or not x.index.is_unique
                or len(x) != len(y) or set(np.unique(y)) != {0, 1}):
            raise ValueError("Unique row indices and aligned binary training data are required")
        if isinstance(y, pd.Series) and not y.index.equals(x.index):
            raise ValueError("Feature and target indices do not align")
        if (self.max_epochs < 1 or self.patience < 1 or self.batch_size < 1
                or not 0 <= self.dropout < 1 or self.weight_decay < 0
                or self.learning_rate <= 0 or not self.hidden or min(self.hidden) < 1):
            raise ValueError("Invalid architecture or training settings")
        y = pd.Series(np.asarray(y), index=x.index)
        fit, valid = train_test_split(np.arange(len(x)), test_size=0.2,
                                     stratify=y, random_state=1729)
        warm_x, warm_bundle = self._prepare_fit(x.iloc[fit], y.iloc[fit])
        valid_x = self._transform(x.iloc[valid], warm_bundle)
        _, self.history_, self.best_epoch_ = self._train(
            warm_x, y.iloc[fit], self.max_epochs, (valid_x, y.iloc[valid]))
        self.audit_ = {"fit_indices": x.index.tolist(),
                       "early_train_indices": x.index[fit].tolist(),
                       "early_validation_indices": x.index[valid].tolist(),
                       "early_bn_folds": warm_bundle.get("cross_fit_folds", [])}
        final_x, self.bundle_ = self._prepare_fit(x, y)
        self.audit_["final_bn_folds"] = self.bundle_.get("cross_fit_folds", [])
        self.network_, self.refit_history_, _ = self._train(final_x, y, self.best_epoch_)
        self.network_.eval()
        self.classes_ = np.array([0, 1])
        self.parameter_count_ = sum(p.numel() for p in self.network_.parameters())
        self.input_features_ = list(self.bundle_["preprocessor"].get_feature_names_out())
        if self.hybrid != "none":
            self.input_features_.append(self.hybrid + "_standardized_log_odds")
        return self

    def predict_proba(self, x):
        check_is_fitted(self, "network_")
        self.network_.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.network_(torch.as_tensor(self._transform(x, self.bundle_)))).numpy().ravel()
        return np.column_stack([1 - p, p]).astype(float)

    def predict(self, x):
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)
