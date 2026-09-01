"""config"""

DEFAULT_MODEL = "yolo11m.pt"
DEFAULT_CONF = 0.30
DEFAULT_IOU = 0.50

VEHICLE_CLASSES = frozenset({
    "car", "truck", "bus", "motorcycle", "bicycle", "train",
})

# Vulnerable Road Users VRUs like people or animals. 
VRU_CLASSES = frozenset({
    "person", "dog", "cat", "horse"
})

CONTROL_CLASSES = frozenset({
    "traffic light", "stop sign",
})

# Detectable on COCO but removed from detection for the HUD to reduce clutter.
IGNORED_CLASSES = frozenset({
    "fire hydrant", "parking meter", "bench",
})

ROAD_CLASSES = VEHICLE_CLASSES | VRU_CLASSES | CONTROL_CLASSES

GROUP_VEHICLE = "vehicle"
GROUP_VRU = "vru"
GROUP_CONTROL = "control"


def class_group(class_name: str) -> str | None:
    if class_name in VEHICLE_CLASSES:
        return GROUP_VEHICLE
    if class_name in VRU_CLASSES:
        return GROUP_VRU
    if class_name in CONTROL_CLASSES:
        return GROUP_CONTROL
    return None


# BGR (OpenCV order).
GROUP_COLORS: dict[str, tuple[int, int, int]] = {
    GROUP_VEHICLE: (0, 200, 255),    # amber, most common class group
    GROUP_VRU:     (0, 80, 255),     # red-orange, highest attention
    GROUP_CONTROL: (0, 255, 120),    # green, signals/signs
}

# ---------------------------------------------------------------------------
# Reasoning / decision layer
# ---------------------------------------------------------------------------

# Ego zone is a symmetric trapezoid in front of the car,
# built from four normalized scalars.
#
#                center
#                  |
#       top_y  +---+---+         (top edge, far away on road)
#              |       |
#              |       |
#  bottom_y +---+---+            (bottom edge, at the bonnet / street start)
#              |<----->|
#            bottom_width
EGO_ZONE_CENTER       = 0.50   # horizontal center as fraction of width
EGO_ZONE_BOTTOM_WIDTH = 0.32   # bottom edge width as fraction of frame width
EGO_ZONE_TOP_WIDTH    = 0.06   # top edge width (narrow = far away)
EGO_ZONE_TOP_Y        = 0.60   # top edge Y (smaller = reaches higher toward horizon)
EGO_ZONE_BOTTOM_Y     = 0.73   # bottom edge Y (at the bonnet edge where street starts)

# Outside-but-within this many pixels of the zone counts as "near".
EGO_NEAR_MARGIN_PX = 80

# Lead-vehicle thresholds in meters.
LEAD_STOP_DISTANCE_M  = 10.0   # <= this and in ego zone -> STOP
LEAD_YIELD_DISTANCE_M = 40.0   # <= this and in ego zone -> YIELD

# Hysteresis: a new action must hold for N consecutive proposals to commit.
HYSTERESIS_N = 5

# ---------------------------------------------------------------------------
# Depth estimation
# ---------------------------------------------------------------------------
# Metric (real-meter) outdoor model. Distances are in meters, max ~80m.
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
DEPTH_MAX_M = 80.0

# Per-camera calibration factor for depth readings. The model is trained on
# KITTI camera intrinsics, so values may need a multiplicative correction on
# different cameras. Tune by eyeballing a known-distance object: take what
# the model reports, divide by the actual distance. Applied to every depth
# sample downstream.
#   1.0  = no correction (use when first measuring)
#   0.35 = typical when the raw readings come out roughly 3x too far
DEPTH_SCALE = 0.35

# BEV lateral conversion factor: (norm_x - 0.5) * BEV_LATERAL_SCALE * distance_m
# gives lateral position in meters. Default 1.15 assumes ~60 deg horizontal
# FOV (typical dashcam). Tune per camera if dots land off-center.
BEV_LATERAL_SCALE = 1.15
BEV_MAX_RANGE_M = 50.0

# Overlay scaling: every viz function computes
#   s = frame.shape[1] / OVERLAY_REFERENCE_WIDTH
# and multiplies all pixel/font constants by s. Increasing this value
# shrinks every overlay proportionally; decreasing it grows them.
OVERLAY_REFERENCE_WIDTH = 2400


ACTION_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "STOP":            (0,   0, 200),
    "BRAKE":           (0,  60, 220),
    "YIELD":           (0, 200, 220),
    "SLOW DOWN":       (0, 200, 220),
    "GO":              (0, 180,  60),
}