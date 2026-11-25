"""Figures for the report. All functions save a PNG and close the figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


def _img(t) -> np.ndarray:
    a = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    if a.ndim == 3:
        a = a[0] if a.shape[0] == 1 else np.transpose(a, (1, 2, 0))
    return np.clip(a, 0, 1)


def sample_grid(images, labels, classes, path: Path, n: int = 20, title: str = "") -> None:
    cols = 10
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.3, rows * 1.5))
    for ax, img, lab in zip(np.ravel(axes), images[:n], labels[:n]):
        ax.imshow(_img(img), cmap="gray")
        ax.set_title(classes[int(lab)], fontsize=7)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def counterfactual_grid(x, x_cf, orig_cls, cf_cls, classes, path: Path, n: int = 8,
                        title: str = "") -> None:
    """Rows: original, counterfactual, difference (red = brighter, blue = darker)."""
    n = min(n, len(x))
    fig, axes = plt.subplots(3, n, figsize=(n * 1.5, 4.8), squeeze=False)
    for i in range(n):
        diff = (x_cf[i] - x[i]).mean(0).detach().cpu().numpy()
        axes[0, i].imshow(_img(x[i]), cmap="gray")
        axes[0, i].set_title(classes[int(orig_cls[i])], fontsize=7)
        axes[1, i].imshow(_img(x_cf[i]), cmap="gray")
        axes[1, i].set_title(classes[int(cf_cls[i])], fontsize=7)
        m = max(abs(diff).max(), 1e-6)
        axes[2, i].imshow(diff, cmap="bwr", vmin=-m, vmax=m)
        for r in range(3):
            axes[r, i].axis("off")
    for r, name in enumerate(["original", "counterfactual", "change"]):
        axes[r, 0].text(-0.3, 0.5, name, transform=axes[r, 0].transAxes, rotation=90,
                        va="center", ha="right", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def heatmap_overlay(x, maps, labels, classes, path: Path, n: int = 8, title: str = "") -> None:
    n = min(n, len(x))
    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3.4), squeeze=False)
    for i in range(n):
        axes[0, i].imshow(_img(x[i]), cmap="gray")
        axes[0, i].set_title(classes[int(labels[i])], fontsize=7)
        axes[1, i].imshow(_img(x[i]), cmap="gray")
        axes[1, i].imshow(maps[i].cpu().numpy() if isinstance(maps[i], torch.Tensor) else maps[i],
                          cmap="jet", alpha=0.5)
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def class_maps(maps, classes, path: Path, title: str = "", diverging: bool = False) -> None:
    k = len(classes)
    fig, axes = plt.subplots(1, k, figsize=(k * 1.4, 1.9))
    for i, ax in enumerate(np.ravel(axes)):
        m = np.asarray(maps[i])
        if diverging:
            v = max(abs(m).max(), 1e-9)
            ax.imshow(m, cmap="bwr", vmin=-v, vmax=v)
        else:
            ax.imshow(m, cmap="hot")
        ax.set_title(classes[i], fontsize=7)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def single_map(m, path: Path, title: str = "") -> None:
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(m, cmap="hot")
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def training_curves(history: list[dict], path: Path, title: str = "") -> None:
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].plot(ep, [h["train_loss"] for h in history], label="train")
    axes[0].plot(ep, [h["val_loss"] for h in history], label="val")
    axes[0].set_title("loss")
    axes[1].plot(ep, [h["train_acc"] for h in history], label="train")
    axes[1].plot(ep, [h["val_acc"] for h in history], label="val")
    axes[1].set_title("accuracy")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def confusion_plot(cm: np.ndarray, classes, path: Path, title: str = "") -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(classes)), classes, fontsize=7)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
