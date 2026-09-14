import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.datasets.dataset import DRGradingDataset
from src.explainability.gradcam import generate_gradcam_overlay
from src.models.build_model import build_model
from src.models.fusion import build_fusion_model
from src.training.train import get_device

SPLITS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "splits")
CHECKPOINT_A = os.path.join(os.path.dirname(__file__), "..", "outputs", "checkpoints", "variant_a_efficientnet_b0.pt")
CHECKPOINT_C = os.path.join(os.path.dirname(__file__), "..", "outputs", "checkpoints", "variant_c_fusion_efficientnet_b0.pt")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "gradcam")

STAGE_OF_INTEREST = 1
NUM_EXAMPLES = 3


def find_stage_indices(samples, stage: int, limit: int) -> list[int]:
    return [idx for idx, (_, grade) in enumerate(samples) if grade == stage][:limit]


def main():
    device = get_device()
    print("device:", device)

    test_ds_a = DRGradingDataset(os.path.join(SPLITS_DIR, "test.csv"), image_size=512)
    test_ds_c = DRGradingDataset(os.path.join(SPLITS_DIR, "test.csv"), image_size=512, with_lesion_map=True)

    model_a = build_model("efficientnet_b0", num_classes=5, pretrained=False)
    model_a.load_state_dict(torch.load(CHECKPOINT_A, map_location=device))
    model_a.to(device).eval()

    model_c = build_fusion_model("efficientnet_b0", num_classes=5, pretrained=False)
    model_c.load_state_dict(torch.load(CHECKPOINT_C, map_location=device))
    model_c.to(device).eval()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    indices = find_stage_indices(test_ds_a.samples, STAGE_OF_INTEREST, NUM_EXAMPLES)

    fig, axes = plt.subplots(len(indices), 3, figsize=(12, 4 * len(indices)))
    if len(indices) == 1:
        axes = [axes]

    for row, idx in enumerate(indices):
        image_a, true_grade = test_ds_a[idx]
        image_c, true_grade_c = test_ds_c[idx]
        assert true_grade == true_grade_c

        input_a = image_a.unsqueeze(0).to(device)
        input_c = image_c.unsqueeze(0).to(device)

        with torch.no_grad():
            pred_a = model_a(input_a).argmax(dim=1).item()
            pred_c = model_c(input_c).argmax(dim=1).item()

        rgb_image = image_a.permute(1, 2, 0).numpy().astype(np.float32)

        overlay_a = generate_gradcam_overlay(model_a, model_a.conv_head, input_a, rgb_image, target_class=pred_a)
        overlay_c = generate_gradcam_overlay(model_c, model_c.conv_head, input_c, rgb_image, target_class=pred_c)

        axes[row][0].imshow(rgb_image)
        axes[row][0].set_title(f"Original (true: Stage {true_grade})")
        axes[row][1].imshow(overlay_a)
        axes[row][1].set_title(f"Variant A - predicted Stage {pred_a}")
        axes[row][2].imshow(overlay_c)
        axes[row][2].set_title(f"Variant C - predicted Stage {pred_c}")
        for ax in axes[row]:
            ax.axis("off")

        print(f"idx {idx}: true={true_grade}, A predicted={pred_a}, C predicted={pred_c}")

    fig.suptitle(f"Variant A vs C Grad-CAM - Stage {STAGE_OF_INTEREST} (Mild) real test examples")
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "variant_a_vs_c_stage1_comparison.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
