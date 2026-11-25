"""Step 1: download/load the dataset, print its size and save a sample image grid.

    python scripts/01_check_data.py --dataset small
    python scripts/01_check_data.py --dataset big
"""

import _bootstrap  # noqa: F401
import json

import torch

from fic.cli import base_parser, load_bundle, setup
from fic.config import output_path
from fic.visualization import sample_grid


def main():
    args = base_parser(__doc__).parse_args()
    cfg, _, _, log = setup(args, "01_check_data")
    b = load_bundle(args, cfg)
    info = {"dataset": b.name, "train": len(b.train), "val": len(b.val), "test": len(b.test),
            "classes": b.classes, "channels": b.in_channels, "image_size": b.image_size}
    log.info(json.dumps(info, indent=2))
    xs, ys = zip(*[b.train[i] for i in range(20)])
    sample_grid(torch.stack(xs), ys, b.classes, output_path(cfg, "figures", b.name, "samples.png"),
                title=f"{b.name}: training samples")
    json.dump(info, open(output_path(cfg, "metrics", b.name, "dataset_info.json"), "w"), indent=2)
    log.info("saved sample grid to outputs/figures/")


if __name__ == "__main__":
    main()
