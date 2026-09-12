import csv
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocessing.preprocess import preprocess_image

IMAGES_DIR_TRAIN = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "raw", "disease_grading",
    "B. Disease Grading", "1. Original Images", "a. Training Set",
)
IMAGES_DIR_TEST = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "raw", "disease_grading",
    "B. Disease Grading", "1. Original Images", "b. Testing Set",
)


class IDRiDGradingDataset(Dataset):

    def __init__(self, csv_path: str, images_dir: str, image_size: int = 512, transform=None):
        self.images_dir = images_dir
        self.image_size = image_size
        self.transform = transform
        self.samples: list[tuple[str, int]] = []

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["image_name"], int(row["grade"])))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_name, grade = self.samples[idx]
        path = os.path.join(self.images_dir, f"{image_name}.jpg")

        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")

        image = preprocess_image(image, size=self.image_size)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW

        return image, grade
