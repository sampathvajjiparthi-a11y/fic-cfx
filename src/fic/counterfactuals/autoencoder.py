"""Convolutional autoencoder.

Used for two things (approval form, testing procedure step 3):
  1. plausibility score = reconstruction error of a counterfactual
     (low error = looks like real data)
  2. latent space for class prototypes (Method B)
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels: int = 1, image_size: int = 28, latent_dim: int = 32):
        super().__init__()
        assert image_size % 4 == 0, "image size must be divisible by 4"
        s = image_size // 4
        self.s = s
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(True),
        )
        self.to_latent = nn.Linear(64 * s * s, latent_dim)
        self.from_latent = nn.Linear(latent_dim, 64 * s * s)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(32, in_channels, 4, stride=2, padding=1), nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_latent(self.encoder_conv(x).flatten(1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z).view(-1, 64, self.s, self.s)
        return self.decoder_conv(F.relu(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def train_autoencoder(ae: ConvAutoencoder, loaders, epochs: int, lr: float, device,
                      logger=None, max_batches: int | None = None) -> dict:
    ae.to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    best, best_state, history = np.inf, None, []
    for epoch in range(1, epochs + 1):
        losses = {}
        for split in ("train", "val"):
            ae.train(split == "train")
            total, n = 0.0, 0
            with torch.set_grad_enabled(split == "train"):
                for i, (x, _) in enumerate(loaders[split]):
                    if max_batches and i >= max_batches:
                        break
                    x = x.to(device)
                    loss = F.mse_loss(ae(x), x)
                    if split == "train":
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                    total += loss.item() * len(x)
                    n += len(x)
            losses[split] = total / max(n, 1)
        history.append({"epoch": epoch, **losses})
        if logger:
            logger.info(f"AE epoch {epoch:02d} | train {losses['train']:.5f} | val {losses['val']:.5f}")
        if losses["val"] < best:
            best, best_state = losses["val"], copy.deepcopy(ae.state_dict())
    ae.load_state_dict(best_state)
    return {"history": history, "best_val_mse": float(best)}


@torch.no_grad()
def reconstruction_error(ae: ConvAutoencoder, x: torch.Tensor) -> torch.Tensor:
    """Per-image mean squared reconstruction error (plausibility; lower is better)."""
    ae.eval()
    return ((ae(x) - x) ** 2).flatten(1).mean(1)
