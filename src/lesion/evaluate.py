from __future__ import annotations

import glob
import os

import cv2
import numpy as np

from src.lesion.detect import lesion_attention_map
from src.preprocessing.preprocess import crop_to_circle_bbox, normalize_illumination, resize

LESION_CLASSES = ["MA", "HE", "EX", "SE"]


def load_ground_truth_mask(segmentation_dir: str, split: str, filename_stem: str) -> np.ndarray:
    """Union of all real lesion masks (MA/HE/EX/SE) for one image, at original resolution."""
    combined = None
    for lesion in LESION_CLASSES:
        mask_path = os.path.join(segmentation_dir, split, "label", lesion, f"{filename_stem}.tif")
        if not os.path.exists(mask_path):
            continue
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        if combined is None:
            combined = np.zeros_like(mask, dtype=np.uint8)
        combined = cv2.bitwise_or(combined, (mask > 0).astype(np.uint8) * 255)
    if combined is None:
        raise FileNotFoundError(f"No lesion masks found for {filename_stem} in split {split}")
    return combined


def _crop_resize_aligned(image: np.ndarray, mask: np.ndarray, size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Apply the identical crop + resize to an image and its ground-truth mask, so pixels line up."""
    top, bottom, left, right = crop_to_circle_bbox(image)
    image_cropped = image[top : bottom + 1, left : right + 1]
    mask_cropped = mask[top : bottom + 1, left : right + 1]

    image_final = resize(normalize_illumination(image_cropped), size)
    mask_final = cv2.resize(mask_cropped, (size, size), interpolation=cv2.INTER_NEAREST)
    return image_final, mask_final


def dice_score(pred_binary: np.ndarray, gt_binary: np.ndarray) -> float:
    pred = pred_binary > 0
    gt = gt_binary > 0
    intersection = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2 * intersection / denom)


def iou_score(pred_binary: np.ndarray, gt_binary: np.ndarray) -> float:
    pred = pred_binary > 0
    gt = gt_binary > 0
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def evaluate_split(
    segmentation_dir: str,
    split: str = "test",
    attention_threshold: float = 0.15,
    limit: int | None = None,
) -> dict:
    """Run the classical detector on every image in a split, score against real ground truth."""
    image_dir = os.path.join(segmentation_dir, split, "image")
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    if limit is not None:
        image_paths = image_paths[:limit]

    dice_scores = []
    iou_scores = []
    skipped = 0

    for image_path in image_paths:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        image = cv2.imread(image_path)
        if image is None:
            skipped += 1
            continue

        try:
            gt_mask = load_ground_truth_mask(segmentation_dir, split, stem)
        except FileNotFoundError:
            skipped += 1
            continue

        image_final, gt_final = _crop_resize_aligned(image, gt_mask)
        attention = lesion_attention_map(image_final)
        pred_binary = (attention >= attention_threshold).astype(np.uint8) * 255

        dice_scores.append(dice_score(pred_binary, gt_final))
        iou_scores.append(iou_score(pred_binary, gt_final))

    return {
        "n_images": len(dice_scores),
        "n_skipped": skipped,
        "mean_dice": float(np.mean(dice_scores)) if dice_scores else 0.0,
        "mean_iou": float(np.mean(iou_scores)) if iou_scores else 0.0,
    }
