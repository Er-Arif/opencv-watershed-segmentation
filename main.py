"""Marker-based watershed segmentation using OpenCV."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def watershed_segment(image: np.ndarray, *, invert: bool = True,
                      foreground_ratio: float = 0.35) -> tuple[np.ndarray, np.ndarray]:
    """Return (image with red watershed boundaries, Otsu binary mask)."""
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a BGR colour image")
    if not 0 < foreground_ratio < 1:
        raise ValueError("foreground_ratio must be between 0 and 1")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, threshold = cv2.threshold(gray, 0, 255, flag | cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), dtype=np.uint8)
    opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_background = cv2.dilate(opening, kernel, iterations=3)
    distances = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_foreground = cv2.threshold(
        distances, foreground_ratio * float(distances.max()), 255, 0
    )
    sure_foreground = sure_foreground.astype(np.uint8)
    unknown = cv2.subtract(sure_background, sure_foreground)
    _, markers = cv2.connectedComponents(sure_foreground)
    markers = markers + 1
    markers[unknown == 255] = 0

    output = image.copy()
    markers = cv2.watershed(output, markers)
    output[markers == -1] = (0, 0, 255)  # BGR red
    return output, threshold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input image")
    parser.add_argument("--output", type=Path, default=Path("output/segmented.png"))
    parser.add_argument("--mask", type=Path, help="Optional Otsu binary mask path")
    parser.add_argument("--no-invert", action="store_true")
    parser.add_argument("--foreground-ratio", type=float, default=0.35)
    args = parser.parse_args()

    image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if image is None:
        parser.error(f"Cannot read image: {args.input}")
    result, mask = watershed_segment(
        image, invert=not args.no_invert, foreground_ratio=args.foreground_ratio
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), result):
        raise OSError(f"Could not write {args.output}")
    if args.mask:
        args.mask.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.mask), mask):
            raise OSError(f"Could not write {args.mask}")
    print(f"Saved segmented image: {args.output}")


if __name__ == "__main__":
    main()
