"""ImageNet-pretrained ResNet-18 (He et al., 2016), fine-tuned on our data.

The wrapper takes the ORIGINAL small image (e.g. 1x28x28) and does the
pre-processing inside forward(): grey -> 3 channels, resize to 224,
ImageNet normalisation. So counterfactuals stay at the original resolution
(approval form, generated data step 2).
"""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ResNet18Classifier(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10,
                 pretrained: bool = True, input_size: int = 224):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.in_channels = in_channels
        self.input_size = input_size
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    @property
    def gradcam_layer(self) -> nn.Module:
        return self.backbone.layer4

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size),
                              mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.preprocess(x))
