"""Step 6: LOCAL and GLOBAL XAI explanations for every model (supervisor note: XAI 'how').

  model                 local (one image)       global (whole model)
  cnn, resnet18         Grad-CAM                class-average Grad-CAM
  logistic_regression   gradient saliency       weight map per class
  random_forest         (none: no gradients)    pixel importance map

    python scripts/06_explain_xai.py --dataset small
"""

import _bootstrap  # noqa: F401

import torch

from fic.cli import base_parser, load_bundle, setup
from fic.config import output_path
from fic.models import ALL_MODELS, SklearnModel, load_differentiable, model_file
from fic.xai import (class_average_gradcam, grad_cam, gradient_saliency, logistic_weight_maps,
                     random_forest_importance)
from fic.visualization import class_maps, heatmap_overlay, single_map


def main():
    p = base_parser(__doc__)
    p.add_argument("--models", nargs="+", default=list(ALL_MODELS), choices=ALL_MODELS)
    p.add_argument("--n_global", type=int, default=50, help="images per class for global Grad-CAM")
    args = p.parse_args()
    cfg, seeds, device, log = setup(args, "06_explain_xai")
    b = load_bundle(args, cfg)
    seed = seeds[0]
    shape = (b.in_channels, b.image_size, b.image_size)

    # first test images of each class
    xs, ys, per_class = [], [], {}
    for i in range(len(b.test)):
        x, y = b.test[i]
        if per_class.get(y, 0) < args.n_global:
            xs.append(x)
            ys.append(y)
            per_class[y] = per_class.get(y, 0) + 1
        if len(per_class) == b.num_classes and min(per_class.values()) >= args.n_global:
            break
    X, Y = torch.stack(xs).to(device), torch.tensor(ys, device=device)
    local_idx = torch.tensor([ys.index(c) for c in sorted(set(ys))][:8])
    fig = lambda name: output_path(cfg, "figures", b.name, "xai", name)  # noqa: E731

    for name in args.models:
        if not model_file(cfg, name, b.name, seed).exists():
            log.warning(f"skip {name}: not trained")
            continue
        if name == "random_forest":
            sk = SklearnModel.load(model_file(cfg, name, b.name, seed))
            single_map(random_forest_importance(sk, shape), fig("global_random_forest.png"),
                       "Random Forest pixel importance (global)")
            log.info("random_forest: global importance map saved")
            continue
        net = load_differentiable(cfg, name, b.name, seed, b.in_channels, b.num_classes, device)
        xl, yl = X[local_idx], Y[local_idx]
        if name == "logistic_regression":
            heatmap_overlay(xl, gradient_saliency(net, xl), yl, b.classes,
                            fig("local_logistic_regression.png"), title="LR gradient saliency (local)")
            sk = SklearnModel.load(model_file(cfg, name, b.name, seed))
            class_maps(logistic_weight_maps(sk, shape), b.classes, fig("global_logistic_regression.png"),
                       "LR weights per class (global; red = pushes towards class)", diverging=True)
        else:
            heatmap_overlay(xl, grad_cam(net, xl), yl, b.classes, fig(f"local_{name}.png"),
                            title=f"{name} Grad-CAM (local)")
            with torch.no_grad():
                ok = net(X).argmax(1) == Y
            maps = class_average_gradcam(net, X[ok], Y[ok], b.num_classes)
            class_maps(maps.numpy(), b.classes, fig(f"global_{name}.png"),
                       f"{name} class-average Grad-CAM (global)")
        log.info(f"{name}: local and global maps saved")


if __name__ == "__main__":
    main()
