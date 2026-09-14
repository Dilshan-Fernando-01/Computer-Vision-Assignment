import os
import sys

import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.lesion.detect import lesion_attention_map
from src.lesion.evaluate import _crop_resize_aligned, dice_score, load_ground_truth_mask

SEGMENTATION_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "ddr_segmentation", "lesion_segmentation")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "lesion_preview")

SAMPLE_IMAGES = [
    ("test", "007-3983-200"),
    ("test", "20170508101047342"),
    ("test", "007-1789-100"),
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for split, stem in SAMPLE_IMAGES:
        image_path = os.path.join(SEGMENTATION_DIR, split, "image", f"{stem}.jpg")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read {image_path}, skipping.")
            continue

        gt_mask = load_ground_truth_mask(SEGMENTATION_DIR, split, stem)
        image_final, gt_final = _crop_resize_aligned(image, gt_mask)

        attention = lesion_attention_map(image_final)
        pred_binary = (attention >= 0.15).astype("uint8") * 255
        dice = dice_score(pred_binary, gt_final)

        overlay = image_final.copy()
        overlay[gt_final > 0] = [0, 0, 255]  # red = ground truth
        overlay[pred_binary > 0] = [0, 255, 0]  # green = classical detector prediction

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(cv2.cvtColor(image_final, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Preprocessed input")
        axes[1].imshow(attention, cmap="hot")
        axes[1].set_title("Lesion attention map")
        axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f"Red=ground truth, Green=predicted\nDice={dice:.3f}")
        for ax in axes:
            ax.axis("off")

        fig.suptitle(f"{stem} ({split})")
        fig.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, f"{stem}_lesion_detection.png")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"Saved {out_path} (Dice={dice:.3f})")


if __name__ == "__main__":
    main()
