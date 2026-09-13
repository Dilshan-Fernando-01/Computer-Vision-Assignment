import timm
import torch.nn as nn


def build_model(backbone_name: str = "efficientnet_b0", num_classes: int = 5, pretrained: bool = True) -> nn.Module:
    model = timm.create_model(backbone_name, pretrained=pretrained, num_classes=num_classes)
    return model


def freeze_backbone(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("classifier")


def unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True
