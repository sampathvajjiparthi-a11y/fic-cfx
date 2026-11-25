"""Choose the test images to explain (approval form, generated data step 3).

100 test images per class (1,000 total), chosen with seed 42 from the images
that ALL seed-0 models classify correctly, so every model is explained on
identical images.
"""

from __future__ import annotations

import numpy as np

from fic.config import output_path


def predictions_file(cfg, dataset: str, model: str, seed: int):
    return output_path(cfg, "predictions", dataset, f"{model}_seed{seed}.npz")


def select_images(cfg, dataset: str, models: list[str], n_per_class: int,
                  seed: int = 42) -> np.ndarray:
    correct, labels = None, None
    for m in models:
        f = predictions_file(cfg, dataset, m, 0)
        if not f.exists():
            raise FileNotFoundError(f"{f} missing. Run scripts/05_evaluate_models.py first.")
        data = np.load(f)
        ok = data["probs"].argmax(1) == data["labels"]
        correct = ok if correct is None else correct & ok
        labels = data["labels"]
    rng = np.random.default_rng(seed)
    chosen = []
    for c in np.unique(labels):
        pool = np.where(correct & (labels == c))[0]
        take = min(n_per_class, len(pool))
        chosen.extend(rng.choice(pool, size=take, replace=False).tolist())
    idx = np.sort(np.array(chosen))
    np.save(output_path(cfg, "counterfactuals", dataset, "selected_indices.npy"), idx)
    return idx
