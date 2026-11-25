"""Shared pieces for all counterfactual methods."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CFBatch:
    """Result of a counterfactual search for a batch of images."""
    x_orig: torch.Tensor      # (B, C, H, W) originals in [0, 1]
    x_cf: torch.Tensor        # (B, C, H, W) counterfactuals in [0, 1]
    original_class: torch.Tensor
    target_class: torch.Tensor
    cf_class: torch.Tensor    # class the model predicts for x_cf
    steps: torch.Tensor       # optimisation steps used per image
    seconds: float            # wall time for the whole batch

    @property
    def valid(self) -> torch.Tensor:
        return self.cf_class == self.target_class


def second_best_target(probs: torch.Tensor) -> torch.Tensor:
    """Approval form: the target class is the class the model ranks second."""
    return probs.topk(2, dim=1).indices[:, 1]
