"""Training loop for the CNN and ResNet-18, with early stopping on validation loss."""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def run_epoch(model, loader, device, optimizer=None, max_batches: int | None = None):
    train = optimizer is not None
    model.train(train)
    loss_fn = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for i, (x, y) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
    return loss_sum / max(total, 1), correct / max(total, 1)


def train_torch_model(model: nn.Module, loaders: dict[str, DataLoader], params, device,
                      logger=None, max_batches: int | None = None) -> dict:
    """Train with AdamW + early stopping. Returns history; best weights are loaded into model.

    max_batches limits batches per epoch (only for quick smoke tests).
    """
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=params["epochs"])
    best_loss, best_state, bad_epochs = np.inf, None, 0
    history = []
    for epoch in range(1, params["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, loaders["train"], device, opt, max_batches)
        va_loss, va_acc = run_epoch(model, loaders["val"], device, None, max_batches)
        sched.step()
        history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
                            val_loss=va_loss, val_acc=va_acc, seconds=time.time() - t0))
        if logger:
            logger.info(f"epoch {epoch:02d} | train {tr_loss:.4f}/{tr_acc:.4f} "
                        f"| val {va_loss:.4f}/{va_acc:.4f} | {time.time() - t0:.0f}s")
        if va_loss < best_loss - 1e-4:
            best_loss, best_state, bad_epochs = va_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= params["patience"]:
                if logger:
                    logger.info(f"early stopping at epoch {epoch}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"history": history, "best_val_loss": float(best_loss)}


@torch.no_grad()
def predict_torch(model: nn.Module, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray]:
    """Return (probabilities [N, K], true labels [N])."""
    model.eval().to(device)
    probs, labels = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x.to(device)), 1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)
