import albumentations as A
import numpy as np


def get_training_augmentations(image_size: int = 512, with_lesion: bool = False) -> A.Compose:
    additional_targets = {"lesion": "mask"} if with_lesion else None
    return A.Compose(
        [
            A.Rotate(limit=25, p=0.7),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
            A.Affine(scale=(0.9, 1.1), translate_percent=(0.0, 0.05), p=0.5),
            A.GaussNoise(std_range=(0.02, 0.08), p=0.2),
        ],
        additional_targets=additional_targets,
    )


def compute_class_weights(labels: list[int], num_classes: int = 5) -> np.ndarray:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1
    weights = 1.0 / counts
    weights = weights / weights.mean()
    return weights


def make_sample_weights(labels: list[int], num_classes: int = 5) -> np.ndarray:
    class_weights = compute_class_weights(labels, num_classes)
    return np.array([class_weights[label] for label in labels])
