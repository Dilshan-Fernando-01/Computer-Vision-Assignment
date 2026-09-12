

import cv2
import numpy as np


def crop_to_circle(image: np.ndarray, threshold: int = 10) -> np.ndarray:
    """Crop away the black background surrounding the circular retina."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray > threshold

    if not mask.any():
        return image

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]

    return image[top : bottom + 1, left : right + 1]


def normalize_illumination(image: np.ndarray, clip_limit: float = 2.5, tile_size: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_channel = clahe.apply(l_channel)

    lab = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def resize(image: np.ndarray, size: int = 512) -> np.ndarray:
    """Resize to a square of the given size."""
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def preprocess_image(image: np.ndarray, size: int = 512) -> np.ndarray:
    """Full preprocessing pipeline: crop -> normalize illumination -> resize."""
    image = crop_to_circle(image)
    image = normalize_illumination(image)
    image = resize(image, size)
    return image
