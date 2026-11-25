"""Black-box prototype counterfactuals for models without gradients (Random Forest).

Model-agnostic version of Method B: it only calls predict_proba().
  1. Find the target-class prototype in autoencoder latent space and decode it.
  2. Line search: x' = x + t * (prototype - x); find the smallest t in [0, 1]
     that makes the model predict the target class (bisection).
  3. Sparsify: try to reset groups of changed pixels back to the original,
     keeping each reset only if the counterfactual stays valid.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from .autoencoder import ConvAutoencoder
from .base import CFBatch
from .prototype import PrototypeBank


def _predict(model, x: torch.Tensor) -> np.ndarray:
    return np.asarray(model.predict(x.detach().cpu().numpy().reshape(len(x), -1)))


@torch.no_grad()
def blackbox_counterfactual(model, ae: ConvAutoencoder, bank: PrototypeBank, x: torch.Tensor,
                            target: torch.Tensor, max_iter: int = 50, n_samples: int = 64,
                            k_neighbours: int = 20, seed: int = 0) -> CFBatch:
    t0 = time.time()
    rng = np.random.default_rng(seed)
    ae.eval()
    orig_class = torch.as_tensor(_predict(model, x), device=x.device)
    z = ae.encode(x)
    proto_img = ae.decode(bank.prototype(z, target, k_neighbours)).clamp(0, 1)
    # fallback end point: nearest real training image the model labels as target (valid by design)
    nearest = bank.nearest_image(z, target) if bank.img is not None else proto_img
    tgt = target.cpu().numpy()
    best = x.clone()
    used = torch.zeros(len(x), device=x.device)

    for i in range(len(x)):
        xi, pi = x[i:i + 1], proto_img[i:i + 1]
        # 1) bisection on the interpolation weight
        if _predict(model, pi)[0] != tgt[i]:
            pi = nearest[i:i + 1]
        if _predict(model, pi)[0] != tgt[i]:
            best[i] = pi[0]                     # no valid end point: keep it (invalid)
            used[i] = 1
            continue
        lo, hi, calls = 0.0, 1.0, 0
        for _ in range(12):
            mid = (lo + hi) / 2
            calls += 1
            if _predict(model, xi + mid * (pi - xi))[0] == tgt[i]:
                hi = mid
            else:
                lo = mid
        cf = xi + hi * (pi - xi)
        # 2) sparsify: reset random groups of changed pixels
        flat_cf, flat_x = cf.flatten(), xi.flatten()
        for _ in range(max_iter):
            changed = torch.nonzero((flat_cf - flat_x).abs() > 1e-6).flatten().cpu().numpy()
            if len(changed) == 0:
                break
            group = rng.choice(changed, size=min(n_samples, len(changed)), replace=False)
            trial = flat_cf.clone()
            trial[group] = flat_x[group]
            calls += 1
            if _predict(model, trial.view_as(xi))[0] == tgt[i]:
                flat_cf = trial
        best[i] = flat_cf.view_as(xi)[0]
        used[i] = calls

    cf_class = torch.as_tensor(_predict(model, best), device=x.device)
    return CFBatch(x, best, orig_class, target, cf_class, used, time.time() - t0)
