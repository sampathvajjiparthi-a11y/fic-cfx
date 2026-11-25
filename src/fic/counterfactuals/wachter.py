"""Method A: gradient-based counterfactuals (Wachter, Mittelstadt and Russell, 2018).

    minimise   lambda * CE(f(x'), target)  +  ||x' - x||_1 / D

x' is kept inside [0, 1]. If an image is not yet valid, its lambda is doubled
every `steps // 5` steps, so the prediction term gets more weight (the original
paper also increases lambda until a counterfactual is found).
Works for any differentiable model: CNN, ResNet-18, Logistic Regression.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .base import CFBatch


def wachter_counterfactual(model, x: torch.Tensor, target: torch.Tensor, steps: int = 500,
                           lr: float = 0.01, lambda_init: float = 0.1,
                           confidence: float = 0.5) -> CFBatch:
    t0 = time.time()
    model.eval()
    x = x.detach()
    B = x.shape[0]
    D = x[0].numel()
    with torch.no_grad():
        orig_class = model(x).argmax(1)

    x_cf = x.clone().requires_grad_(True)
    opt = torch.optim.Adam([x_cf], lr=lr)
    lam = torch.full((B,), lambda_init, device=x.device)
    done = torch.zeros(B, dtype=torch.bool, device=x.device)
    best = x.clone()
    used = torch.full((B,), steps, device=x.device)
    for p in model.parameters():
        p.requires_grad_(False)

    for step in range(steps):
        logits = model(x_cf)
        ce = F.cross_entropy(logits, target, reduction="none")
        dist = (x_cf - x).abs().flatten(1).sum(1) / D
        loss = (lam * ce + dist)[~done].sum()
        opt.zero_grad()
        if loss.requires_grad:
            loss.backward()
            x_cf.grad[done] = 0
            opt.step()
        with torch.no_grad():
            x_cf.clamp_(0, 1)
            probs = torch.softmax(model(x_cf), 1)
            ok = (probs.argmax(1) == target) & (probs.gather(1, target[:, None]).squeeze(1) >= confidence)
            newly = ok & ~done
            best[newly] = x_cf[newly]
            used[newly] = step + 1
            done |= ok
            if (step + 1) % max(steps // 5, 1) == 0:
                lam[~done] *= 2.0
            if done.all():
                break

    with torch.no_grad():
        best[~done] = x_cf[~done].detach()   # keep last attempt for failed ones
        cf_class = model(best).argmax(1)
    for p in model.parameters():
        p.requires_grad_(True)
    return CFBatch(x, best.detach(), orig_class, target, cf_class, used, time.time() - t0)
