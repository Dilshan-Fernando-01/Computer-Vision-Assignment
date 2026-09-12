import os
import sys

import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing.preprocess import preprocess_image
from src.augmentation.augment import get_training_augmentations

IMAGE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "raw",
    "disease_grading",
    "B. Disease Grading",
    "1. Original Images",
    "a. Training Set",
    "IDRiD_021.jpg",  
)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "augmentation_preview")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    original = cv2.imread(IMAGE_PATH)
    preprocessed = preprocess_image(original)
    transform = get_training_augmentations()

    n_samples = 5
    fig, axes = plt.subplots(1, n_samples + 1, figsize=(4 * (n_samples + 1), 4))

    axes[0].imshow(cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Preprocessed\n(before augmentation)")
    axes[0].axis("off")

    for i in range(n_samples):
        augmented = transform(image=preprocessed)["image"]
        axes[i + 1].imshow(cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB))
        axes[i + 1].set_title(f"Augmented sample {i + 1}")
        axes[i + 1].axis("off")

    fig.suptitle("IDRiD_021.jpg - real Stage 1 (Mild) example")
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "stage1_augmentation.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
