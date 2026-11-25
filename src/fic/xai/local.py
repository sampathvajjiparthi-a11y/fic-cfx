"""LOCAL explanations: explain one prediction for one image.

* Grad-CAM (Selvaraju et al., 2017) for CNN and ResNet-18
* Vanilla gradient saliency for any differentiable model (incl. Logistic Regression)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gradient_saliency(model, x: torch.Tensor, cls: torch.Tensor | None = None) -> torch.Tensor:
    """|d logit_c / d x|, max over channels. Returns (B, H, W) scaled to [0, 1]."""
    model.eval()
    x = x.clone().requires_grad_(True)
    logits = model(x)
    cls = logits.argmax(1) if cls is None else cls
    logits.gather(1, cls[:, None]).sum().backward()
    sal = x.grad.abs().amax(1)
    sal = sal / (sal.flatten(1).amax(1)[:, None, None] + 1e-12)
    return sal.detach()


def grad_cam(model, x: torch.Tensor, cls: torch.Tensor | None = None) -> torch.Tensor:
    """Grad-CAM heatmap on model.gradcam_layer. Returns (B, H, W) in [0, 1]."""
    layer = model.gradcam_layer
    store = {}
    h1 = layer.register_forward_hook(lambda m, i, o: store.__setitem__("act", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("grad", go[0]))
    try:
        model.eval()
        x = x.clone().requires_grad_(True)
        logits = model(x)
        cls = logits.argmax(1) if cls is None else cls
        model.zero_grad()
        logits.gather(1, cls[:, None]).sum().backward()
        weights = store["grad"].mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * store["act"]).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam / (cam.flatten(1).amax(1)[:, None, None] + 1e-12)
    finally:
        h1.remove()
        h2.remove()
    return cam.detach()
