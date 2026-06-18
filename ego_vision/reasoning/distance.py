"""Sample per-detection distance from a metric depth map."""

import numpy as np

from ego_vision.config.settings import DEPTH_SCALE
from ego_vision.reasoning.ego_zone import contact_point


def sample_distance(depth_map, box_xyxy, patch_half=4):
    """Median of a small patch around the box's bottom-center, scaled by DEPTH_SCALE.

    Bottom-center is the object's footprint; sampling the bbox center would
    often hit air above a pedestrian or the roof of a car.
    """
    h, w = depth_map.shape[:2]
    cx, cy = contact_point(box_xyxy)
    cx_i = int(round(cx))
    cy_i = int(round(cy))

    if cx_i < 0 or cx_i >= w or cy_i < 0 or cy_i >= h:
        return None

    x0 = max(0, cx_i - patch_half)
    x1 = min(w, cx_i + patch_half + 1)
    y0 = max(0, cy_i - patch_half)
    y1 = min(h, cy_i + patch_half + 1)
    patch = depth_map[y0:y1, x0:x1]
    if patch.size == 0:
        return None

    return float(np.median(patch)) * DEPTH_SCALE
