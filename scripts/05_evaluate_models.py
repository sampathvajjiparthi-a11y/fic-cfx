"""Step 5 (RQ1, H1): test accuracy of every trained model + McNemar comparison.

Saves test-set predictions (needed later to pick the images to explain),
an accuracy table (mean and SD over seeds, Wilson 95% CI) and confusion matrices.

    python scripts/05_evaluate_models.py --dataset small
"""

import _bootstrap  # noqa: F401
from itertools import combinations

import numpy as np
import pandas as pd

from fic.cli import base_parser, load_bundle, setup
from fic.config import output_path
from fic.data import make_loaders, to_numpy
from fic.evaluation import classification_report, confusion, predictions_file
from fic.models import ALL_MODELS, TORCH_MODELS, SklearnModel, load_differentiable, model_file
from fic.stats import holm_correction, mcnemar_test
from fic.training import predict_torch
from fic.visualization import confusion_plot


def main():
    p = base_parser(__doc__)
    p.add_argument("--models", nargs="+", default=list(ALL_MODELS), choices=ALL_MODELS)
    args = p.parse_args()
    cfg, seeds, device, log = setup(args, "05_evaluate_models")
    b = load_bundle(args, cfg)
    test_loader = make_loaders(b, cfg)["test"]
    X_te, y_te = to_numpy(b.test)

    rows, correct_seed0 = [], {}
    for name in args.models:
        for seed in seeds:
            f = model_file(cfg, name, b.name, seed)
            if not f.exists():
                log.warning(f"skip {name} seed {seed}: {f.name} not found")
                continue
            if name in TORCH_MODELS:
                net = load_differentiable(cfg, name, b.name, seed, b.in_channels, b.num_classes, device)
                probs, labels = predict_torch(net, test_loader, device)
            else:
                probs, labels = SklearnModel.load(f).predict_proba(X_te), y_te
            np.savez_compressed(predictions_file(cfg, b.name, name, seed), probs=probs, labels=labels)
            pred = probs.argmax(1)
            rep = classification_report(labels, pred)
            rows.append({"model": name, "seed": seed, **rep})
            log.info(f"{name} seed {seed}: accuracy {rep['accuracy']:.4f} "
                     f"[{rep['ci_low']:.4f}, {rep['ci_high']:.4f}]")
            if seed == seeds[0]:
                correct_seed0[name] = pred == labels
                confusion_plot(confusion(labels, pred, b.num_classes), b.classes,
                               output_path(cfg, "figures", b.name, f"confusion_{name}.png"),
                               title=f"{name} on {b.name}")

    if not rows:
        log.error("no trained models found")
        return
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(output_path(cfg, "metrics", b.name, "accuracy_per_seed.csv"), index=False)
    summary = per_seed.groupby("model")["accuracy"].agg(["mean", "std", "count"]).reset_index()
    summary.to_csv(output_path(cfg, "metrics", b.name, "accuracy_summary.csv"), index=False)
    log.info("\n" + summary.to_string(index=False))

    # H1: pairwise McNemar tests between models (first seed), Holm-corrected
    tests = []
    for a, c in combinations(correct_seed0, 2):
        tests.append({"comparison": f"{a} vs {c}", **mcnemar_test(correct_seed0[a], correct_seed0[c])})
    if tests:
        res = holm_correction(pd.DataFrame(tests), cfg.evaluation.alpha)
        res.to_csv(output_path(cfg, "metrics", b.name, "h1_mcnemar.csv"), index=False)
        log.info("\n" + res[["comparison", "p", "p_holm", "effect", "significant"]].to_string(index=False))


if __name__ == "__main__":
    main()
