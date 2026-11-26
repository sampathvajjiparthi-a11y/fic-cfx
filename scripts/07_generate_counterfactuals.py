"""Step 7 (RQ2, RQ3): generate counterfactual explanations and score them.

  --method wachter     Method A, gradient-based (cnn, resnet18, logistic_regression)
  --method prototype   Method B, prototype-guided (cnn, resnet18, logistic_regression)
  --method blackbox    prototype-guided without gradients (ALL four models; the only one for random_forest)

Images: 100 per class that all chosen models classify correctly (seed 0).
Target: the class each model ranks second.

    python scripts/07_generate_counterfactuals.py --dataset small --model cnn --method wachter
    python scripts/07_generate_counterfactuals.py --dataset small --model random_forest --method blackbox
"""

import _bootstrap  # noqa: F401

import pandas as pd
import torch

from fic.cli import base_parser, load_bundle, setup
from fic.config import output_path
from fic.counterfactuals import (ConvAutoencoder, PrototypeBank, blackbox_counterfactual,
                                 prototype_counterfactual, second_best_target,
                                 wachter_counterfactual)
from fic.evaluation import counterfactual_metrics, select_images
from fic.models import ALL_MODELS, SklearnModel, load_differentiable, model_file
from fic.visualization import counterfactual_grid


def main():
    p = base_parser(__doc__)
    p.add_argument("--model", required=True, choices=ALL_MODELS)
    p.add_argument("--method", required=True, choices=["wachter", "prototype", "blackbox"])
    p.add_argument("--select_from", nargs="+", default=None,
                   help="models used to pick commonly-correct images (default: all trained models)")
    p.add_argument("--batch_size", type=int, default=100)
    args = p.parse_args()
    cfg, seeds, device, log = setup(args, f"07_cf_{args.model}_{args.method}")
    b = load_bundle(args, cfg)
    seed = seeds[0]

    if args.model == "random_forest" and args.method != "blackbox":
        raise SystemExit("Random Forest has no gradients: use --method blackbox")

    # 1) images to explain
    pool = args.select_from or [m for m in ALL_MODELS if model_file(cfg, m, b.name, 0).exists()]
    idx = select_images(cfg, b.name, pool, cfg.counterfactuals.n_per_class, cfg.project.split_seed)
    X = torch.stack([b.test[int(i)][0] for i in idx]).to(device)
    log.info(f"explaining {len(idx)} images (commonly correct for: {pool})")

    # 2) model, autoencoder, prototypes
    ae = ConvAutoencoder(b.in_channels, b.image_size, cfg.autoencoder.latent_dim).to(device)
    ae.load_state_dict(torch.load(output_path(cfg, "models", b.name, "autoencoder.pt"), map_location=device))
    ae.eval()
    if args.model == "random_forest":
        clf = SklearnModel.load(model_file(cfg, args.model, b.name, seed))
        probs_fn = lambda x: torch.tensor(clf.predict_proba(x), device=device)  # noqa: E731
    else:
        clf = load_differentiable(cfg, args.model, b.name, seed, b.in_channels, b.num_classes, device)
        probs_fn = lambda x: torch.softmax(clf(x), 1)  # noqa: E731
    bank = None
    if args.method in ("prototype", "blackbox"):
        bank = PrototypeBank.build(ae, clf, b.train, device, keep_images=args.method == "blackbox")

    # 3) generate in batches
    frames, first = [], None
    for s in range(0, len(X), args.batch_size):
        xb = X[s:s + args.batch_size]
        with torch.no_grad():
            target = second_best_target(probs_fn(xb))
        if args.method == "wachter":
            cf = wachter_counterfactual(clf, xb, target, **dict(cfg.counterfactuals.wachter))
        elif args.method == "prototype":
            cf = prototype_counterfactual(clf, ae, bank, xb, target, **dict(cfg.counterfactuals.prototype))
        else:
            bb = dict(cfg.counterfactuals.blackbox)
            cf = blackbox_counterfactual(clf, ae, bank, xb, target, bb["max_iter"], bb["n_samples"],
                                         cfg.counterfactuals.prototype.k_neighbours, seed)
        df = counterfactual_metrics(cf, ae, cfg.evaluation.sparsity_eps)
        df.insert(0, "test_index", idx[s:s + len(xb)])
        frames.append(df)
        first = first or cf
        log.info(f"batch {s // args.batch_size + 1}: validity {df.valid.mean():.2%}, "
                 f"{cf.seconds:.0f}s")

    # 4) save
    out = pd.concat(frames, ignore_index=True)
    out.insert(1, "model", args.model)
    out.insert(2, "method", args.method)
    out.insert(3, "dataset", b.name)
    csv = output_path(cfg, "counterfactuals", b.name, f"{args.model}_{args.method}.csv")
    out.to_csv(csv, index=False)
    counterfactual_grid(first.x_orig, first.x_cf, first.original_class, first.cf_class, b.classes,
                        output_path(cfg, "figures", b.name, f"cf_{args.model}_{args.method}.png"),
                        n=10, title=f"{args.method} counterfactuals for {args.model} on {b.name}")
    log.info("\n" + out[["valid", "l1", "l2", "sparsity", "plausibility_ae"]].describe()
             .loc[["mean", "50%"]].to_string())
    log.info(f"saved {csv}")


if __name__ == "__main__":
    main()
