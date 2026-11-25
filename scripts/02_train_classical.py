"""Step 2: train classical ML models (Logistic Regression, Random Forest) on flattened pixels.

    python scripts/02_train_classical.py --dataset small
    python scripts/02_train_classical.py --dataset small --models random_forest --seeds 0
"""

import _bootstrap  # noqa: F401
import json
import time

from fic.cli import base_parser, load_bundle, setup
from fic.config import output_path
from fic.data import to_numpy
from fic.models import CLASSICAL_MODELS, SklearnModel, model_file


def main():
    p = base_parser(__doc__)
    p.add_argument("--models", nargs="+", default=list(CLASSICAL_MODELS), choices=CLASSICAL_MODELS)
    args = p.parse_args()
    cfg, seeds, _, log = setup(args, "02_train_classical")
    b = load_bundle(args, cfg)
    X_tr, y_tr = to_numpy(b.train)
    X_va, y_va = to_numpy(b.val)
    log.info(f"{b.name}: X_train {X_tr.shape}, X_val {X_va.shape}")
    for name in args.models:
        for seed in seeds:
            t0 = time.time()
            m = SklearnModel(name, dict(cfg.models[name]), seed,
                             (b.in_channels, b.image_size, b.image_size)).fit(X_tr, y_tr)
            val_acc = float((m.predict(X_va) == y_va).mean())
            m.save(model_file(cfg, name, b.name, seed))
            secs = time.time() - t0
            log.info(f"{name} seed {seed}: val acc {val_acc:.4f} ({secs:.0f}s)")
            json.dump({"val_acc": val_acc, "train_seconds": secs},
                      open(output_path(cfg, "logs", b.name, f"{name}_seed{seed}.json"), "w"))


if __name__ == "__main__":
    main()
