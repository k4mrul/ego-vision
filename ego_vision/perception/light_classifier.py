"""Traffic-light color classifier (HSV thresholding).

Input is the BGR crop of a "traffic light" detection box.
Output is "red", "yellow", "green", or "unknown".
"""

import cv2
import numpy as np

# OpenCV HSV: H is 0-180, S and V are 0-255.
# Red wraps the hue circle, so it needs two ranges.
_RED_RANGES = [
    ((0, 120, 100), (10, 255, 255)),
    ((170, 120, 100), (180, 255, 255)),
]
_YELLOW_RANGE = ((15, 120, 120), (35, 255, 255))
_GREEN_RANGE = ((40, 80, 80), (90, 255, 255))

# Below this fraction of the crop matching any color, return "unknown".
_MIN_COLOR_FRACTION = 0.02


def classify_light(crop_bgr) -> str:
    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

    red_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in _RED_RANGES:
        red_mask = cv2.bitwise_or(
            red_mask, cv2.inRange(hsv, np.array(lo), np.array(hi))
        )
    yellow_mask = cv2.inRange(hsv, np.array(_YELLOW_RANGE[0]), np.array(_YELLOW_RANGE[1]))
    green_mask = cv2.inRange(hsv, np.array(_GREEN_RANGE[0]), np.array(_GREEN_RANGE[1]))

    counts = {
        "red": int(red_mask.sum() // 255),
        "yellow": int(yellow_mask.sum() // 255),
        "green": int(green_mask.sum() // 255),
    }

    total = crop_bgr.shape[0] * crop_bgr.shape[1]
    best = max(counts, key=counts.get)
    if total == 0 or counts[best] / total < _MIN_COLOR_FRACTION:
        return "unknown"
    return best


def crop_box(frame, box_xyxy):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    x1i = max(0, int(round(x1)))
    y1i = max(0, int(round(y1)))
    x2i = min(w, int(round(x2)))
    y2i = min(h, int(round(y2)))
    if x2i <= x1i or y2i <= y1i:
        return np.empty((0, 0, 3), dtype=np.uint8)
    return frame[y1i:y2i, x1i:x2i]
