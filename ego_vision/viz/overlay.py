"""
Draw detection boxes colored by class group or light state.

Pixel and font constants scale with frame width relative to REFERENCE_W.
"""

import cv2

from ego_vision.config.settings import GROUP_COLORS, LIGHT_COLORS
from ego_vision.viz.common import scale


def draw_detections(frame, detections, show_conf=True, show_distance=True):
    s = scale(frame)
    box_thick = max(1, int(2 * s))
    font_scale = 0.5 * s
    font_thick = max(1, int(1 * s))
    label_pad_x = max(1, int(2 * s))
    label_pad_y = max(1, int(2 * s))

    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.box_xyxy)

        if det.class_name == "traffic light" and det.light_state is not None:
            color = LIGHT_COLORS.get(det.light_state, LIGHT_COLORS["unknown"])
        else:
            color = GROUP_COLORS.get(det.class_group, (255, 255, 255))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thick)

        parts = [det.class_name]
        if det.class_name == "traffic light" and det.light_state is not None:
            parts.append(det.light_state.upper())
        if show_distance and det.distance_m is not None:
            parts.append(f"{det.distance_m:.1f}m")
        if show_conf:
            parts.append(f"{det.confidence:.2f}")
        label = " ".join(parts)

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        label_y_top = max(0, y1 - th - baseline - label_pad_y)
        cv2.rectangle(
            frame,
            (x1, label_y_top),
            (x1 + tw + 2 * label_pad_x, label_y_top + th + baseline + label_pad_y),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + label_pad_x, label_y_top + th),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            font_thick,
            cv2.LINE_AA,
        )
    return frame
