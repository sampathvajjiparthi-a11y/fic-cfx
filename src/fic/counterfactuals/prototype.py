"""Method B: prototype-guided counterfactuals (Van Looveren and Klaise, 2021).

    minimise   c * CE(f(x'), target)                    prediction term
             + beta * ||d||_1 + ||d||_2^2                elastic net (d = x' - x)
             + gamma * ||x' - AE(x')||_2^2               stay close to the data
             + theta * ||ENC(x') - proto_target||_2^2    move towards target prototype

The prototype of the target class is the mean encoding of the k training
images of that class that are closest to ENC(x) in latent space (as in the
paper). This is a PyTorch re-implementation of the method in the Alibi
library, so it runs with the same models and settings as Method A.
The prediction term uses the same cross-entropy as Method A. Unlike Method A,
the search does not stop at the first valid image: it runs all steps with an L1
shrinkage (proximal) step, as in the paper's FISTA optimiser, and keeps the valid
counterfactual with the lowest explanation cost.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .autoencoder import ConvAutoencoder
from .base import CFBatch


class PrototypeBank:
    """Stores autoencoder encodings of training images grouped by predicted class."""

    def __init__(self, encodings: torch.Tensor, labels: torch.Tensor,
                 images: torch.Tensor | None = None):
        self.enc = {int(c): encodings[labels == c] for c in labels.unique()}
        self.img = ({int(c): images[labels == c] for c in labels.unique()}
                    if images is not None else None)

    def nearest_image(self, z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Nearest real training image (in latent space) that the model labels as target."""
        assert self.img is not None, "build the bank with keep_images=True"
        out = []
        for zi, t in zip(z, target):
            d = ((self.enc[int(t)].to(z.device) - zi) ** 2).sum(1)
            out.append(self.img[int(t)][d.argmin()].to(z.device))
        return torch.stack(out)

    @classmethod
    @torch.no_grad()
    def build(cls, ae: ConvAutoencoder, model, dataset, device, max_per_class: int = 2000,
              batch_size: int = 512, keep_images: bool = False) -> "PrototypeBank":
        ae.eval()
        encs, labs, imgs = [], [], []
        counts: dict[int, int] = {}
        for x, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            x = x.to(device)
            pred = predict_labels(model, x)
            z = ae.encode(x)
            keep = torch.tensor([counts.get(int(p), 0) < max_per_class for p in pred], device=device)
            for p in pred[keep]:
                counts[int(p)] = counts.get(int(p), 0) + 1
            encs.append(z[keep].cpu())
            labs.append(pred[keep].cpu())
            if keep_images:
                imgs.append(x[keep].cpu())
            if counts and min(counts.values()) >= max_per_class and len(counts) > 1:
                break
        return cls(torch.cat(encs), torch.cat(labs), torch.cat(imgs) if keep_images else None)

    def prototype(self, z: torch.Tensor, target: torch.Tensor, k: int) -> torch.Tensor:
        """For each row of z, mean of the k nearest encodings of its target class."""
        out = []
        for zi, t in zip(z, target):
            bank = self.enc[int(t)].to(z.device)
            d = ((bank - zi) ** 2).sum(1)
            idx = d.topk(min(k, len(bank)), largest=False).indices
            out.append(bank[idx].mean(0))
        return torch.stack(out)


def predict_labels(model, x: torch.Tensor) -> torch.Tensor:
    """Works for torch models and for sklearn wrappers (Random Forest)."""
    if isinstance(model, torch.nn.Module):
        return model(x).argmax(1)
    import numpy as np
    return torch.as_tensor(np.asarray(model.predict(x)), device=x.device)


def prototype_counterfactual(model, ae: ConvAutoencoder, bank: PrototypeBank, x: torch.Tensor,
                             target: torch.Tensor, steps: int = 500, lr: float = 0.01,
                             beta: float = 0.1, gamma: float = 1.0, theta: float = 1.0,
                             k_neighbours: int = 20, confidence: float = 0.5) -> CFBatch:
    t0 = time.time()
    model.eval()
    ae.eval()
    x = x.detach()
    B = x.shape[0]
    D = x[0].numel()
    with torch.no_grad():
        orig_class = model(x).argmax(1)
        proto = bank.prototype(ae.encode(x), target, k_neighbours)

    for p in list(model.parameters()) + list(ae.parameters()):
        p.requires_grad_(False)

    # Optimise the perturbation delta. After each gradient step on the smooth terms,
    # an L1 proximal (soft-threshold) step pushes small changes to exactly zero, like
    # the shrinkage-thresholding (FISTA) optimiser in the original paper -> sparse changes.
    delta = torch.zeros_like(x, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    shrink = lr * beta
    c = torch.ones(B, device=x.device)
    found = torch.zeros(B, dtype=torch.bool, device=x.device)
    best = x.clone()
    best_score = torch.full((B,), float("inf"), device=x.device)
    used = torch.full((B,), steps, device=x.device)

    for step in range(steps):
        x_cf = (x + delta).clamp(0, 1)
        d = (x_cf - x).flatten(1)
        pred_term = F.cross_entropy(model(x_cf), target, reduction="none")
        l2 = (d ** 2).sum(1) / D
        ae_term = ((ae(x_cf) - x_cf) ** 2).flatten(1).mean(1)
        proto_term = ((ae.encode(x_cf) - proto) ** 2).mean(1)
        loss = (c * pred_term + l2 + gamma * ae_term + theta * proto_term).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            delta.copy_(torch.sign(delta) * (delta.abs() - shrink).clamp(min=0))   # L1 prox
            delta.copy_((x + delta).clamp(0, 1) - x)
            x_cf = x + delta
            probs = torch.softmax(model(x_cf), 1)
            ok = (probs.argmax(1) == target) & (probs.gather(1, target[:, None]).squeeze(1) >= confidence)
            # keep the valid counterfactual with the lowest explanation cost seen so far
            dd = delta.flatten(1)
            score = (beta * dd.abs().sum(1) + (dd ** 2).sum(1)) / D \
                + gamma * ((ae(x_cf) - x_cf) ** 2).flatten(1).mean(1) \
                + theta * ((ae.encode(x_cf) - proto) ** 2).mean(1)
            better = ok & (score < best_score)
            best[better] = x_cf[better]
            best_score[better] = score[better]
            used[ok & ~found] = step + 1
            found |= ok
            if (step + 1) % max(steps // 5, 1) == 0:
                c[~found] *= 2.0     # raise the prediction weight only until a CF is found

    with torch.no_grad():
        best[~found] = (x + delta)[~found]
        cf_class = model(best).argmax(1)
    for p in list(model.parameters()) + list(ae.parameters()):
        p.requires_grad_(True)
    return CFBatch(x, best.detach(), orig_class, target, cf_class, used, time.time() - t0)
