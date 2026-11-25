"""Load the experiment configuration from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


class Config(dict):
    """A dict that also allows attribute access: cfg.models.cnn.lr"""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        cfg = Config(yaml.safe_load(f))
    return cfg


def output_path(cfg: Config, *parts: str) -> Path:
    """Return a path inside the output folder and make sure its folder exists."""
    p = PROJECT_ROOT / cfg.project.output_dir / Path(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def data_path(cfg: Config) -> Path:
    p = PROJECT_ROOT / cfg.project.data_dir
    p.mkdir(parents=True, exist_ok=True)
    return p
