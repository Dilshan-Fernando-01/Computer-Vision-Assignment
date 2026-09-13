from __future__ import annotations

import timm
import torch
import torch.nn as nn
from coral_pytorch.dataset import levels_from_labelbatch, proba_to_label
from coral_pytorch.layers import CoralLayer
from coral_pytorch.losses import coral_loss


class CoralModel(nn.Module):
    """Pretrained backbone (feature extractor) + CoralLayer head."""

    def __init__(self, backbone_name: str = "efficientnet_b0", num_classes: int = 5, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.coral_layer = CoralLayer(self.backbone.num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.coral_layer(features)  # (batch, num_classes - 1) CORAL logits


class CoralCriterion:

    def __init__(self, num_classes: int, importance_weights: torch.Tensor | None = None):
        self.num_classes = num_classes
        self.importance_weights = importance_weights

    def to(self, device):
        if self.importance_weights is not None:
            self.importance_weights = self.importance_weights.to(device)
        return self

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        levels = levels_from_labelbatch(labels, num_classes=self.num_classes).to(logits.device)
        return coral_loss(logits, levels, importance_weights=self.importance_weights)


def coral_predict(logits: torch.Tensor) -> torch.Tensor:
    """CORAL logits -> predicted integer stage (0..num_classes-1)."""
    probas = torch.sigmoid(logits)
    return proba_to_label(probas)
