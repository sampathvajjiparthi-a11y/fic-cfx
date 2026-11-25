"""Classical machine learning models (supervisor note: Random Forest, Logistic Regression).

They work on flattened pixels (N, C*H*W). Two helpers make them usable by the
rest of the pipeline:

* SklearnModel: fit / predict / predict_proba on numpy or torch inputs.
* LogisticRegressionTorch: copies a fitted sklearn Logistic Regression into a
  differentiable PyTorch layer, so gradient-based counterfactuals (Method A and
  Method B) can be generated for it exactly like for the CNNs.

Random Forest has no gradients, so it is explained with the black-box
counterfactual method in fic.counterfactuals.blackbox.
"""

from __future__ import annotations

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from torch import nn


class SklearnModel:
    def __init__(self, kind: str, params: dict, seed: int = 0, image_shape: tuple = (1, 28, 28)):
        self.kind = kind
        self.image_shape = tuple(image_shape)
        if kind == "logistic_regression":
            self.model = LogisticRegression(C=params.get("C", 1.0),
                                            max_iter=params.get("max_iter", 1000),
                                            random_state=seed)
        elif kind == "random_forest":
            self.model = RandomForestClassifier(n_estimators=params.get("n_estimators", 300),
                                                max_depth=params.get("max_depth"),
                                                n_jobs=params.get("n_jobs", -1),
                                                random_state=seed)
        else:
            raise ValueError(kind)

    @staticmethod
    def _flat(x) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return x.reshape(x.shape[0], -1)

    def fit(self, X, y) -> "SklearnModel":
        self.model.fit(self._flat(X), y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(self._flat(X))

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(self._flat(X))

    def save(self, path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "SklearnModel":
        return joblib.load(path)


class LogisticRegressionTorch(nn.Module):
    """softmax(W x + b): identical outputs to the fitted sklearn model."""

    def __init__(self, sk: SklearnModel):
        super().__init__()
        lr = sk.model
        assert isinstance(lr, LogisticRegression)
        n_classes, n_features = lr.coef_.shape
        self.linear = nn.Linear(n_features, n_classes)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor(lr.coef_, dtype=torch.float32))
            self.linear.bias.copy_(torch.tensor(lr.intercept_, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.reshape(x.shape[0], -1))
