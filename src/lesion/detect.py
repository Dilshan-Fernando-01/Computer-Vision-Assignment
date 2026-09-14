from __future__ import annotations

import cv2
import numpy as np


def _circularity(contour: np.ndarray) -> float:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)
    if perimeter == 0:
        return 0.0
    return float(4 * np.pi * area / (perimeter**2))


def _filter_by_shape(
    binary_mask: np.ndarray,
    min_area: float = 3.0,
    max_area: float = 6000.0,
    min_circularity: float = 0.3,
) -> np.ndarray:
    """Keep only roughly-round, reasonably-sized blobs; drops vessel fragments and noise."""
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = np.zeros_like(binary_mask)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        if _circularity(contour) < min_circularity:
            continue
        cv2.drawContours(kept, [contour], -1, 255, thickness=cv2.FILLED)
    return kept


def retina_foreground_mask(image_bgr: np.ndarray, border_margin: int = 20) -> np.ndarray:
    """The circular retina area, eroded inward so the bright/dark crop edge itself
    (background-to-retina transition) can never be mistaken for a lesion blob."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, foreground = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    foreground = cv2.erode(foreground, np.ones((border_margin, border_margin), np.uint8))
    return foreground


def detect_optic_disc(image_bgr: np.ndarray) -> np.ndarray:
    """Rough optic disc localization: the largest very-bright blob in the image.

    The exclusion zone is deliberately generous (dilated well beyond the disc itself)
    because the vessels converging on the disc create dense branch points that a
    round-blob shape filter alone cannot reliably tell apart from real lesions.
    """
    green = image_bgr[:, :, 1]
    blurred = cv2.GaussianBlur(green, (25, 25), 0)

    threshold_value = np.percentile(blurred, 99)
    _, bright = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
    bright = bright.astype(np.uint8)

    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    disc_mask = np.zeros_like(green, dtype=np.uint8)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(disc_mask, [largest], -1, 255, thickness=cv2.FILLED)
        disc_mask = cv2.dilate(disc_mask, np.ones((45, 45), np.uint8))
    return disc_mask


def _adaptive_threshold(response: np.ndarray, foreground: np.ndarray, n_std: float = 3.5) -> float:
    """Mean + n_std * std of the response, measured only over the real retina area.

    A fixed percentile (e.g. "top 1% of pixels") would always flag ~1% of the image
    even when there is no real lesion there at all. This instead asks "how unusual is
    this pixel compared to this image's own normal background texture", so a clean
    image with faint background noise stays almost entirely unflagged.
    """
    values = response[foreground > 0]
    if values.size == 0:
        return 255.0
    return float(values.mean() + n_std * values.std())


def detect_dark_lesions(image_bgr: np.ndarray, foreground: np.ndarray, kernel_size: int = 19) -> np.ndarray:
    """Candidate microaneurysms/haemorrhages: small dark round blobs."""
    green = image_bgr[:, :, 1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, kernel)

    threshold_value = _adaptive_threshold(blackhat, foreground)
    _, binary = cv2.threshold(blackhat, threshold_value, 255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_and(binary.astype(np.uint8), foreground)

    return _filter_by_shape(binary, min_area=3, max_area=3000, min_circularity=0.45)


def detect_bright_lesions(
    image_bgr: np.ndarray, disc_mask: np.ndarray, foreground: np.ndarray, kernel_size: int = 19
) -> np.ndarray:
    """Candidate hard/soft exudates: small bright round blobs, excluding the optic disc."""
    green = image_bgr[:, :, 1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    tophat = cv2.morphologyEx(green, cv2.MORPH_TOPHAT, kernel)
    tophat[disc_mask > 0] = 0

    valid = cv2.bitwise_and(foreground, cv2.bitwise_not(disc_mask))
    threshold_value = _adaptive_threshold(tophat, valid)
    _, binary = cv2.threshold(tophat, threshold_value, 255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_and(binary.astype(np.uint8), valid)

    return _filter_by_shape(binary, min_area=3, max_area=3000, min_circularity=0.45)


def lesion_attention_map(image_bgr: np.ndarray, blur_ksize: int = 9) -> np.ndarray:
    """Full pipeline: returns a soft (0-1 float32) attention map, same H x W as the input.

    Soft rather than binary on purpose - a small Gaussian blur turns hard "in/out" edges
    into a gradient, which fuses better with CNN feature maps and is more forgiving of
    imperfect localization than a hard mask would be.
    """
    foreground = retina_foreground_mask(image_bgr)
    disc_mask = detect_optic_disc(image_bgr)
    dark = detect_dark_lesions(image_bgr, foreground)
    bright = detect_bright_lesions(image_bgr, disc_mask, foreground)

    combined = cv2.bitwise_or(dark, bright)
    combined = cv2.GaussianBlur(combined, (blur_ksize, blur_ksize), 0)

    attention = combined.astype(np.float32) / 255.0
    max_value = attention.max()
    if max_value > 0:
        attention = attention / max_value
    return attention
