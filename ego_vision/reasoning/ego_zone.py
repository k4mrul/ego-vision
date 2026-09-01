"""Static symmetric trapezoid for the road ahead.

Built from 4 normalized scalars (see settings.EGO_ZONE_*).
"""

import cv2
import numpy as np

from ego_vision.config.settings import (
    EGO_NEAR_MARGIN_PX,
    EGO_ZONE_BOTTOM_WIDTH,
    EGO_ZONE_BOTTOM_Y,
    EGO_ZONE_CENTER,
    EGO_ZONE_TOP_WIDTH,
    EGO_ZONE_TOP_Y,
)


def make_ego_zone(
    width,
    height,
    *,
    center=EGO_ZONE_CENTER,
    bottom_width=EGO_ZONE_BOTTOM_WIDTH,
    top_width=EGO_ZONE_TOP_WIDTH,
    top_y=EGO_ZONE_TOP_Y,
    bottom_y=EGO_ZONE_BOTTOM_Y,
):
    bw = bottom_width / 2.0
    tw = top_width / 2.0
    pts_norm = [
        (center - bw, bottom_y),
        (center + bw, bottom_y),
        (center + tw, top_y),
        (center - tw, top_y),
    ]
    pts = [(int(round(x * width)), int(round(y * height))) for x, y in pts_norm]
    return np.array(pts, dtype=np.int32)


def contact_point(box_xyxy):
    """Bottom-center of the box (pedestrian's feet, vehicle's wheels)."""
    x1, _, x2, y2 = box_xyxy
    return ((x1 + x2) / 2.0, float(y2))


def in_zone(box_xyxy, zone):
    return cv2.pointPolygonTest(zone, contact_point(box_xyxy), False) >= 0


def near_zone(box_xyxy, zone, margin_px=EGO_NEAR_MARGIN_PX):
    """Outside the zone but within margin_px of its boundary."""
    d = cv2.pointPolygonTest(zone, contact_point(box_xyxy), True)
    return d < 0 and -d <= margin_px
