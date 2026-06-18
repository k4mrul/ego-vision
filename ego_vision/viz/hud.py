"""HUD overlays: ego zone + action card + debug strip.

All pixel and font constants are tuned to look right at a REFERENCE_W-wide
frame; everything scales linearly from there so the layout reads the same at
720p, 1080p, 1440p, or 4K.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from ego_vision.config.settings import ACTION_COLORS_BGR
from ego_vision.viz.common import _translucent_panel, scale

FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class HudState:
    action: str
    reason: str
    frame_idx: int = 0
    proc_fps: float = 0.0
    recent_actions: list = field(default_factory=list)
    light_ahead: str | None = None
    speed_kmh: float | None = None
    # System status panel
    track_count: int = 0
    device: str = ""


def draw_ego_zone(frame: np.ndarray, zone: np.ndarray) -> np.ndarray:
    s = scale(frame)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [zone], (0, 255, 255))
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    cv2.polylines(frame, [zone], True, (0, 255, 255), max(1, int(2 * s)))
    return frame


def draw_debug_strip(frame: np.ndarray, hud: HudState) -> np.ndarray:
    s = scale(frame)
    h = frame.shape[0]
    parts = []
    if hud.speed_kmh is not None:
        parts.append(f"speed:{hud.speed_kmh:5.1f} km/h")
    if hud.light_ahead:
        parts.append(f"light:{hud.light_ahead}")
    if not parts:
        return frame
    text = "  |  ".join(parts)

    font_scale = 0.7 * s
    thickness = max(1, int(2 * s))
    pad = max(2, int(10 * s))
    margin = max(2, int(8 * s))
    (tw, th), bl = cv2.getTextSize(text, FONT, font_scale, thickness)
    y2 = h - margin
    y1 = y2 - th - bl - 2 * pad

    _translucent_panel(
        frame, (margin, y1), (margin + tw + 2 * pad, y2),
        alpha=0.7, border=True
    )

    cv2.putText(
        frame, text, (margin + pad, y2 - pad - bl),
        FONT, font_scale, (240, 240, 240), thickness, cv2.LINE_AA,
    )
    return frame


def draw_status_panel(frame, hud):
    """Top-right widget: system status dots + counts."""
    s = scale(frame)
    w = frame.shape[1]
    panel_w = int(250 * s)
    panel_h = int(288 * s)
    margin = int(16 * s)
    x0 = w - panel_w - margin
    y0 = margin

    pad_x = int(12 * s)
    header_y = y0 + int(26 * s)
    div1_y = y0 + int(36 * s)
    row_h = int(24 * s)
    row_h_text = int(22 * s)
    dot_r = max(2, int(5 * s))

    # Semi-transparent dark background with thin border.
    _translucent_panel(
        frame, (x0, y0), (x0 + panel_w, y0 + panel_h), 
        alpha=0.7, border=True
    )

    # Header.
    cv2.putText(frame, "SYSTEMS", (x0 + pad_x, header_y),
                FONT, 0.6 * s, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.line(frame, (x0 + pad_x, div1_y), (x0 + panel_w - pad_x, div1_y), (60, 60, 60), 1)

    # Status rows.
    items = [
        ("Detection", True),
        ("Tracking", True),
        ("Depth", True),
        ("Engine", True),
    ]
    GREEN = (0, 200, 0)
    GRAY = (90, 90, 90)

    y = y0 + int(64 * s)
    dot_x = x0 + int(22 * s)
    label_x = x0 + int(36 * s)
    status_x = x0 + panel_w - int(46 * s)

    for label, on in items:
        color = GREEN if on else GRAY
        cv2.circle(frame, (dot_x, y - int(4 * s)), dot_r, color, -1)
        cv2.putText(frame, label, (label_x, y),
                    FONT, 0.5 * s, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, "OK" if on else "OFF", (status_x, y),
                    FONT, 0.5 * s, color, 1, cv2.LINE_AA)
        y += row_h

    # Data rows: dividers act as separator markers in the same list.
    data_rows = [
        "divider",
        ("Device", hud.device),
        "divider",
        ("Tracks", hud.track_count),
        ("Frame",  hud.frame_idx),
        ("FPS",    f"{hud.proc_fps:5.1f}"),
    ]
    for row in data_rows:
        if row == "divider":
            cv2.line(frame, (x0 + pad_x, y - int(4 * s)),
                     (x0 + panel_w - pad_x, y - int(4 * s)), (60, 60, 60), 1)
            y += row_h_text
            continue
        label, value = row
        cv2.putText(frame, f"{label:9}{value}", (x0 + pad_x, y),
                    FONT, 0.5 * s, (220, 220, 220), 1, cv2.LINE_AA)
        y += row_h_text

    return frame


def draw_hud(frame: np.ndarray, hud: HudState) -> np.ndarray:
    """Bottom-center card with current action and reason."""
    s = scale(frame)
    color = ACTION_COLORS_BGR.get(hud.action, (200, 200, 200))
    h, w = frame.shape[:2]

    pad = int(14 * s)
    bottom_gap = int(40 * s)
    action_scale = 1.0 * s
    reason_scale = 0.6 * s
    action_thick = max(1, int(3 * s))

    (atw, ath), abl = cv2.getTextSize(hud.action, FONT, action_scale, action_thick)
    (rtw, rth), rbl = cv2.getTextSize(hud.reason, FONT, reason_scale, 1)
    box_w = max(atw, rtw) + 2 * pad
    box_h = ath + abl + rth + rbl + 3 * pad
    x = (w - box_w) // 2
    y = h - box_h - bottom_gap

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), color, -1)
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    cv2.putText(
        frame, hud.action, (x + pad, y + pad + ath),
        FONT, action_scale, (255, 255, 255), action_thick, cv2.LINE_AA,
    )
    cv2.putText(
        frame, hud.reason, (x + pad, y + pad + ath + abl + pad + rth),
        FONT, reason_scale, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return frame
