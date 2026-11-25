"""Shared command-line options for the scripts in scripts/."""

from __future__ import annotations

import argparse

from fic.config import load_config, output_path
from fic.utils import get_device, get_logger


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None, help="path to YAML config (default: configs/default.yaml)")
    p.add_argument("--dataset", default="small", help="'small', 'big' or a dataset name")
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="override seeds, e.g. --seeds 0")
    p.add_argument("--quick", action="store_true",
                   help="smoke test: tiny subsets and 1 epoch, to check the code runs")
    return p


def load_bundle(args, cfg):
    """Load the dataset; in --quick mode keep only small subsets so it runs in seconds."""
    from torch.utils.data import Subset

    from fic.data import load_dataset

    bundle = load_dataset(args.dataset, cfg)
    if args.quick:
        bundle.train = Subset(bundle.train, range(min(3000, len(bundle.train))))
        bundle.val = Subset(bundle.val, range(min(500, len(bundle.val))))
        bundle.test = Subset(bundle.test, range(min(1000, len(bundle.test))))
    return bundle


def setup(args, script_name: str):
    cfg = load_config(args.config)
    if args.quick:
        cfg["models"]["cnn"]["epochs"] = 1
        cfg["models"]["resnet18"]["epochs"] = 1
        cfg["autoencoder"]["epochs"] = 1
        cfg["counterfactuals"]["n_per_class"] = 2
        cfg["counterfactuals"]["wachter"]["steps"] = 100
        cfg["counterfactuals"]["prototype"]["steps"] = 100
        cfg["models"]["random_forest"]["n_estimators"] = 50
        cfg["evaluation"]["bootstrap"] = 200
    seeds = args.seeds if args.seeds is not None else cfg.project.seeds
    if args.quick and args.seeds is None:
        seeds = [0]
    device = get_device(cfg.project.device)
    logger = get_logger(script_name, output_path(cfg, "logs", f"{script_name}.log"))
    logger.info(f"device={device} dataset={args.dataset} seeds={seeds} quick={args.quick}")
    return cfg, seeds, device, logger
