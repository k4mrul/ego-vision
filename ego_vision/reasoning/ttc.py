"""Time-to-collision estimator.

Per-track distance history. TTC = distance / closing_speed when approaching;
otherwise None.
"""

from collections import deque

from ego_vision.perception.detector import Detection


class TtcEstimator:
    def __init__(self, fps, history=5, gc_after_frames=30):
        self.fps = max(1e-6, fps)
        self.history = max(2, history)
        self.gc_after = gc_after_frames
        self._tracks = {}
        self._last_seen = {}
        self._frame_idx = 0

    def update(self, detections) -> list[Detection]:
        self._frame_idx += 1
        for d in detections:
            ttc = self._ingest(d)
            if ttc is not None:
                d.ttc_sec = ttc

        stale = [
            tid for tid, last in self._last_seen.items()
            if self._frame_idx - last > self.gc_after
        ]
        for tid in stale:
            self._tracks.pop(tid, None)
            self._last_seen.pop(tid, None)
        return detections

    def _ingest(self, d):
        if d.track_id is None or d.distance_m is None:
            return None
        tid = d.track_id
        buf = self._tracks.setdefault(tid, deque(maxlen=self.history))
        buf.append((self._frame_idx, d.distance_m))
        self._last_seen[tid] = self._frame_idx
        if len(buf) < 2:
            return None
        (f0, d0), (f1, d1) = buf[0], buf[-1]
        gap = f1 - f0
        if gap <= 0:
            return None
        closing = (d0 - d1) * self.fps / gap
        if closing <= 0:
            return None
        return d1 / closing
