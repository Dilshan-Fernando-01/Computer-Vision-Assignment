import os
import sys

import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.datasets.dataset import IMAGES_DIR
from src.preprocessing.preprocess import crop_to_circle, normalize_illumination, resize

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "preprocessing_preview")

SAMPLE_IMAGES = ["007-5988-300.jpg", "007-2809-100.jpg", "007-0004-000.jpg"]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for filename in SAMPLE_IMAGES:
        path = os.path.join(IMAGES_DIR, filename)
        original = cv2.imread(path)
        if original is None:
            print(f"Could not read {path}, skipping.")
            continue

        cropped = crop_to_circle(original)
        normalized = normalize_illumination(cropped)
        final = resize(normalized, 512)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax, img, title in zip(
            axes,
            [original, cropped, normalized, final],
            ["1. Original", "2. Cropped to circle", "3. Illumination normalized (CLAHE)", "4. Resized (512x512)"],
        ):
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(f"{title}\n{img.shape[1]}x{img.shape[0]}")
            ax.axis("off")

        fig.suptitle(filename)
        fig.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(filename)[0]}_preprocessing.png")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
