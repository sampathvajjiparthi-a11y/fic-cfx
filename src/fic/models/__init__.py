"""Model registry: one place to build or load any model by name."""

from __future__ import annotations

from pathlib import Path

import torch

from .classical import LogisticRegressionTorch, SklearnModel
from .cnn import SimpleCNN
from .resnet import ResNet18Classifier

TORCH_MODELS = ("cnn", "resnet18")
CLASSICAL_MODELS = ("logistic_regression", "random_forest")
ALL_MODELS = TORCH_MODELS + CLASSICAL_MODELS


def build_torch_model(name: str, in_channels: int, num_classes: int, cfg) -> torch.nn.Module:
    if name == "cnn":
        return SimpleCNN(in_channels, num_classes)
    if name == "resnet18":
        m = cfg.models.resnet18
        return ResNet18Classifier(in_channels, num_classes, m.pretrained, m.input_size)
    raise ValueError(f"Unknown torch model '{name}'")


def model_file(cfg, model: str, dataset: str, seed: int) -> Path:
    from fic.config import output_path
    ext = "pt" if model in TORCH_MODELS else "joblib"
    return output_path(cfg, "models", dataset, f"{model}_seed{seed}.{ext}")


def load_differentiable(cfg, model: str, dataset: str, seed: int,
                        in_channels: int, num_classes: int, device) -> torch.nn.Module | None:
    """Load a model as a torch module (logits out). Returns None for Random Forest."""
    path = model_file(cfg, model, dataset, seed)
    if model in TORCH_MODELS:
        net = build_torch_model(model, in_channels, num_classes, cfg)
        net.load_state_dict(torch.load(path, map_location=device))
        return net.to(device).eval()
    if model == "logistic_regression":
        return LogisticRegressionTorch(SklearnModel.load(path)).to(device).eval()
    return None


__all__ = [
    "SimpleCNN", "ResNet18Classifier", "SklearnModel", "LogisticRegressionTorch",
    "build_torch_model", "model_file", "load_differentiable",
    "TORCH_MODELS", "CLASSICAL_MODELS", "ALL_MODELS",
]
