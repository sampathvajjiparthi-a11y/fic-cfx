"""Step 4: train one autoencoder per dataset (plausibility score + prototypes).

    python scripts/04_train_autoencoder.py --dataset small
"""

import _bootstrap  # noqa: F401
import json

import torch

from fic.cli import base_parser, load_bundle, setup
from fic.config import output_path
from fic.counterfactuals import ConvAutoencoder, train_autoencoder
from fic.data import make_loaders
from fic.utils import set_seed


def main():
    args = base_parser(__doc__).parse_args()
    cfg, _, device, log = setup(args, "04_train_autoencoder")
    set_seed(cfg.project.split_seed)
    b = load_bundle(args, cfg)
    ae = ConvAutoencoder(b.in_channels, b.image_size, cfg.autoencoder.latent_dim)
    res = train_autoencoder(ae, make_loaders(b, cfg), cfg.autoencoder.epochs, cfg.autoencoder.lr,
                            device, log, max_batches=20 if args.quick else None)
    torch.save(ae.state_dict(), output_path(cfg, "models", b.name, "autoencoder.pt"))
    json.dump(res, open(output_path(cfg, "logs", b.name, "autoencoder.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
