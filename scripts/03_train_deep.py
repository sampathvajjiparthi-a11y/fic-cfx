"""Step 3: train the self-built CNN (from scratch) or fine-tune the pretrained ResNet-18.

    python scripts/03_train_deep.py --model cnn --dataset small
    python scripts/03_train_deep.py --model resnet18 --dataset small
"""

import _bootstrap  # noqa: F401
import json

import torch

from fic.cli import base_parser, load_bundle, setup
from fic.config import output_path
from fic.data import make_loaders
from fic.models import TORCH_MODELS, build_torch_model, model_file
from fic.training import train_torch_model
from fic.utils import set_seed
from fic.visualization import training_curves


def main():
    p = base_parser(__doc__)
    p.add_argument("--model", required=True, choices=TORCH_MODELS)
    args = p.parse_args()
    cfg, seeds, device, log = setup(args, f"03_train_{args.model}")
    b = load_bundle(args, cfg)
    loaders = make_loaders(b, cfg)
    params = dict(cfg.models[args.model])
    for seed in seeds:
        set_seed(seed)
        net = build_torch_model(args.model, b.in_channels, b.num_classes, cfg)
        log.info(f"{args.model} seed {seed}: {sum(p.numel() for p in net.parameters()):,} parameters")
        result = train_torch_model(net, loaders, params, device, log,
                                   max_batches=20 if args.quick else None)
        torch.save(net.state_dict(), model_file(cfg, args.model, b.name, seed))
        json.dump(result, open(output_path(cfg, "logs", b.name, f"{args.model}_seed{seed}.json"), "w"),
                  indent=2)
        training_curves(result["history"],
                        output_path(cfg, "figures", b.name, f"curves_{args.model}_seed{seed}.png"),
                        title=f"{args.model} on {b.name} (seed {seed})")


if __name__ == "__main__":
    main()
