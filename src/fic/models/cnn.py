"""Self-built CNN, trained from scratch (supervisor note: 'CNN (self-built)')."""

import torch
from torch import nn


def _block(c_in: int, c_out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2), nn.Dropout(dropout),
    )


class SimpleCNN(nn.Module):
    """Three conv blocks + small classifier head.

    Works for any input size (28, 32, 64 ...) because of the adaptive pooling.
    Input: float tensor in [0, 1], shape (B, C, H, W). Output: class logits.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            _block(in_channels, 32, 0.1),
            _block(32, 64, 0.2),
            _block(64, 128, 0.3),
        )
        self.pool = nn.AdaptiveAvgPool2d(2)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 2 * 2, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.4), nn.Linear(256, num_classes),
        )

    @property
    def gradcam_layer(self) -> nn.Module:
        """Last convolutional block, used by Grad-CAM."""
        return self.features[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))
