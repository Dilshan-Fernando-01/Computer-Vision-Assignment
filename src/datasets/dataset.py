from __future__ import annotations

import csv
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocessing.preprocess import preprocess_image

cv2.setNumThreads(0)


IMAGES_DIR = os.environ.get(
    "DDR_IMAGES_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "ddr", "DR_grading", "DR_grading"),
)

CACHE_DIR = os.environ.get(
    "DDR_CACHE_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "image_cache"),
)


class DRGradingDataset(Dataset):


    def __init__(
        self,
        csv_path: str,
        images_dir: str = IMAGES_DIR,
        image_size: int = 512,
        transform=None,
        cache_dir: str | None = CACHE_DIR,
    ):
        self.images_dir = images_dir
        self.image_size = image_size
        self.transform = transform
        self.cache_dir = cache_dir
        self.samples: list[tuple[str, int]] = []

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["image_name"], int(row["grade"])))

    def __len__(self) -> int:
        return len(self.samples)

    def _load_preprocessed(self, image_name: str) -> np.ndarray:
        cache_path = os.path.join(self.cache_dir, f"{image_name}.jpg") if self.cache_dir else None

        if cache_path and os.path.exists(cache_path):
            image = cv2.imread(cache_path)
            if image is not None:
                return image

        path = os.path.join(self.images_dir, f"{image_name}.jpg")
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")

        image = preprocess_image(image, size=self.image_size)

        if cache_path:
            cv2.imwrite(cache_path, image)

        return image

    def __getitem__(self, idx: int):
        image_name, grade = self.samples[idx]
        image = self._load_preprocessed(image_name)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW

        return image, grade
