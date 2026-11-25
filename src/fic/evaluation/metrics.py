"""Metrics.

Classification (RQ1): accuracy, Wilson 95% CI, confusion matrix.
Counterfactual quality (RQ2, RQ3), as defined in the approval form:
  validity      model now predicts the target class
  proximity     L1 and L2 distance to the original image
  sparsity      number of pixel values changed by more than eps (0.01)
  plausibility  autoencoder reconstruction error (lower = more realistic)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from sklearn.metrics import confusion_matrix

from fic.counterfactuals.autoencoder import reconstruction_error
from fic.counterfactuals.base import CFBatch


def wilson_ci(correct: int, n: int, level: float = 0.95) -> tuple[float, float]:
    z = norm.ppf(1 - (1 - level) / 2)
    p = correct / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return centre - half, centre + half


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    correct = int((y_true == y_pred).sum())
    lo, hi = wilson_ci(correct, len(y_true))
    return {"accuracy": correct / len(y_true), "ci_low": lo, "ci_high": hi, "n": len(y_true)}


def confusion(y_true, y_pred, num_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


@torch.no_grad()
def counterfactual_metrics(cf: CFBatch, ae, eps: float = 0.01) -> pd.DataFrame:
    d = (cf.x_cf - cf.x_orig).flatten(1)
    return pd.DataFrame({
        "original_class": cf.original_class.cpu().numpy(),
        "target_class": cf.target_class.cpu().numpy(),
        "cf_class": cf.cf_class.cpu().numpy(),
        "valid": cf.valid.cpu().numpy().astype(int),
        "l1": d.abs().sum(1).cpu().numpy(),
        "l2": d.norm(dim=1).cpu().numpy(),
        "sparsity": (d.abs() > eps).sum(1).cpu().numpy(),
        "plausibility_ae": reconstruction_error(ae, cf.x_cf).cpu().numpy(),
        "steps": cf.steps.cpu().numpy(),
        "seconds_per_image": cf.seconds / len(cf.x_orig),
    })
