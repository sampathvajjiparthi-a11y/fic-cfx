"""GLOBAL explanations: explain the model as a whole, not one image.

* Logistic Regression: weight map per class (what pixels push towards each class)
* Random Forest: feature (pixel) importance map
* CNN / ResNet-18: class-average Grad-CAM over many correctly classified images
"""

from __future__ import annotations

import numpy as np
import torch

from .local import grad_cam


def logistic_weight_maps(sk_model, image_shape) -> np.ndarray:
    """(K, H, W) weight maps; channels averaged for colour images."""
    coef = sk_model.model.coef_.reshape(-1, *image_shape)
    return coef.mean(1)


def random_forest_importance(sk_model, image_shape) -> np.ndarray:
    imp = sk_model.model.feature_importances_.reshape(image_shape)
    return imp.mean(0)


def class_average_gradcam(model, x: torch.Tensor, y: torch.Tensor, num_classes: int,
                          batch_size: int = 64) -> torch.Tensor:
    """Mean Grad-CAM per class over the given (correctly classified) images. (K, H, W)."""
    sums = torch.zeros(num_classes, *x.shape[-2:])
    counts = torch.zeros(num_classes)
    for i in range(0, len(x), batch_size):
        xb, yb = x[i:i + batch_size], y[i:i + batch_size]
        cam = grad_cam(model, xb, yb).cpu()
        for c in yb.unique():
            sums[int(c)] += cam[yb.cpu() == c].sum(0)
            counts[int(c)] += (yb == c).sum().item()
    return sums / counts.clamp(min=1)[:, None, None]
