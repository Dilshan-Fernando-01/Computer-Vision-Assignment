import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.datasets.dataset import DRGradingDataset
from src.explainability.gradcam import generate_gradcam_overlay
from src.models.build_model import build_model
from src.training.train import get_device

SPLITS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "splits")
CHECKPOINT = os.path.join(os.path.dirname(__file__), "..", "outputs", "checkpoints", "variant_a_efficientnet_b0.pt")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "gradcam")

NUM_CLASSES = 5


def pick_one_per_stage(samples: list[tuple[str, int]]) -> dict[int, int]:
    picked = {}
    for idx, (_, grade) in enumerate(samples):
        if grade not in picked:
            picked[grade] = idx
        if len(picked) == NUM_CLASSES:
            break
    return picked


def main():
    device = get_device()
    print("device:", device)

    test_ds = DRGradingDataset(os.path.join(SPLITS_DIR, "test.csv"), image_size=512)

    model = build_model("efficientnet_b0", num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.to(device)
    model.eval()

    target_layer = model.conv_head

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    picked = pick_one_per_stage(test_ds.samples)

    for stage in sorted(picked):
        idx = picked[stage]
        image_tensor, true_grade = test_ds[idx]
        input_tensor = image_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            predicted = model(input_tensor).argmax(dim=1).item()

        rgb_image = image_tensor.permute(1, 2, 0).numpy().astype(np.float32)
        overlay = generate_gradcam_overlay(model, target_layer, input_tensor, rgb_image, target_class=predicted)

        fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
        axes[0].imshow(rgb_image)
        axes[0].set_title("Original (preprocessed)")
        axes[1].imshow(overlay)
        correctness = "correct" if predicted == true_grade else "WRONG"
        axes[1].set_title(f"Grad-CAM - True: Stage {true_grade}, Predicted: Stage {predicted} ({correctness})")
        for ax in axes:
            ax.axis("off")

        fig.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, f"stage{stage}_gradcam.png")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"Saved {out_path} (true={true_grade}, predicted={predicted})")


if __name__ == "__main__":
    main()
