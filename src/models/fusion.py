from __future__ import annotations

import timm
import torch
import torch.nn as nn


def _expand_stem_to_extra_channel(model: nn.Module) -> None:

    if not hasattr(model, "conv_stem"):
        raise ValueError("Fusion model expects an EfficientNet-family backbone with a 'conv_stem' first layer")

    stem = model.conv_stem
    old_weight = stem.weight.data  
    mean_channel = old_weight.mean(dim=1, keepdim=True) 
    new_weight = torch.cat([old_weight, mean_channel], dim=1)  

    new_stem = nn.Conv2d(
        in_channels=4,
        out_channels=stem.out_channels,
        kernel_size=stem.kernel_size,
        stride=stem.stride,
        padding=stem.padding,
        bias=stem.bias is not None,
    )
    new_stem.weight.data = new_weight
    if stem.bias is not None:
        new_stem.bias.data = stem.bias.data.clone()
    model.conv_stem = new_stem


def build_fusion_model(backbone_name: str = "efficientnet_b0", num_classes: int = 5, pretrained: bool = True) -> nn.Module:
    model = timm.create_model(backbone_name, pretrained=pretrained, num_classes=num_classes)
    _expand_stem_to_extra_channel(model)
    return model
