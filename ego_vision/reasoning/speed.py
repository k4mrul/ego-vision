"""Ego vehicle speed estimation.

Two signals combined under EMA smoothing:

1. Primary: distance change to tracked static objects (traffic lights,
   stop signs). They don't move; the rate their measured distance shrinks
   IS our speed. Median across visible static objects is naturally robust.

2. Fallback: depth at a fixed road-surface pixel. Used only when no
   static objects have a stable track. Brittle if the sample pixel gets
   covered by a vehicle, so we sanity-reject very small readings.

EMA smooths across the handoff so the displayed speed doesn't jump when the
source switches.
"""

from collections import deque

import numpy as np

from ego_vision.config.settings import DEPTH_SCALE, GROUP_CONTROL


class SpeedEstimator:
    def __init__(
        self,
        fps,
        history=5,
        ema_alpha=0.30,
        road_sample_x_frac=0.50,
        road_sample_y_frac=0.85,
        gc_after_frames=30,
        sanity_max_mps=50.0,
        road_min_distance_m=2.0,
    ):
        self.fps = max(1e-6, fps)
        self.history = max(2, history)
        self.ema_alpha = ema_alpha
        self.road_sample_x_frac = road_sample_x_frac
        self.road_sample_y_frac = road_sample_y_frac
        self.gc_after = gc_after_frames
        self.sanity_max_mps = sanity_max_mps
        self.road_min_distance_m = road_min_distance_m

        self._track_buf = {}
        self._track_last_seen = {}
        self._road_buf = deque(maxlen=self.history)
        self._frame_idx = 0
        self._ema = None

    def update(self, detections, depth_map) -> float:
        """Returns smoothed ego speed in m/s, or None until first valid sample."""
        self._frame_idx += 1

        candidates = []
        for d in detections:
            if d.class_group != GROUP_CONTROL:
                continue
            if d.track_id is None or d.distance_m is None:
                continue
            v = self._track_speed(d.track_id, d.distance_m)
            if v is not None:
                candidates.append(v)

        if candidates:
            speed_raw = float(np.median(candidates))
        elif depth_map is not None:
            speed_raw = self._road_speed(depth_map)
        else:
            speed_raw = None

        if speed_raw is not None and abs(speed_raw) > self.sanity_max_mps:
            speed_raw = None

        self._gc()

        if speed_raw is None:
            return self._ema
        if self._ema is None:
            self._ema = speed_raw
        else:
            self._ema = self.ema_alpha * speed_raw + (1 - self.ema_alpha) * self._ema
        return self._ema

    def _regression_speed(self, buf) :
        """Fit a line to (frame_idx, distance) and return ego speed from its slope."""
        if len(buf) < 2:
            return None
        frames = np.array([f for f, _ in buf], dtype=np.float64)
        dists = np.array([d for _, d in buf], dtype=np.float64)
        if frames[-1] - frames[0] <= 0:
            return None
        slope, _ = np.polyfit(frames, dists, 1)
        return float(-slope * self.fps)

    def _track_speed(self, tid, d_curr):
        buf = self._track_buf.setdefault(tid, deque(maxlen=self.history))
        buf.append((self._frame_idx, d_curr))
        self._track_last_seen[tid] = self._frame_idx
        return self._regression_speed(buf)

    def _road_speed(self, depth_map):
        h, w = depth_map.shape[:2]
        x = int(round(self.road_sample_x_frac * w))
        y = int(round(self.road_sample_y_frac * h))
        if x < 0 or x >= w or y < 0 or y >= h:
            return None
        ph = 4
        x0, x1 = max(0, x - ph), min(w, x + ph + 1)
        y0, y1 = max(0, y - ph), min(h, y + ph + 1)
        patch = depth_map[y0:y1, x0:x1]
        if patch.size == 0:
            return None

        # Apply same DEPTH_SCALE calibration as sample_distance.
        d_curr = float(np.median(patch)) * DEPTH_SCALE
        # If the road sample is suspiciously close, a car drove into the pixel.
        if d_curr < self.road_min_distance_m:
            return None
        self._road_buf.append((self._frame_idx, d_curr))
        return self._regression_speed(self._road_buf)

    def _gc(self):
        stale = [
            tid for tid, last in self._track_last_seen.items()
            if self._frame_idx - last > self.gc_after
        ]
        for tid in stale:
            self._track_buf.pop(tid, None)
            self._track_last_seen.pop(tid, None)
