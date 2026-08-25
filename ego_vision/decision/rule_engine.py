"""Priority rule engine with hysteresis.

Priority order (first match wins):
  1. STOP: any in-zone object within LEAD_STOP_DISTANCE_M
  2. SLOW DOWN: any in-zone object within LEAD_YIELD_DISTANCE_M
  3. GO: otherwise

Hysteresis: a new action must hold for N consecutive proposals before it
replaces the displayed one.
"""

from collections import deque

from ego_vision.config.settings import (
    HYSTERESIS_N,
    LEAD_STOP_DISTANCE_M,
    LEAD_YIELD_DISTANCE_M,
)


class RuleEngine:
    def __init__(self, hysteresis_n=HYSTERESIS_N):
        self.n = max(1, hysteresis_n)
        self._pending = deque(maxlen=self.n)
        self.current = ("GO", "starting")
        self.history = []

    def decide(self, scene) -> tuple[str, str]:
        proposed = self._propose(scene)
        self._pending.append(proposed)
        if (
            len(self._pending) == self.n
            and all(p[0] == proposed[0] for p in self._pending)
            and proposed[0] != self.current[0]
        ):
            self.current = proposed
            self.history.append(proposed[0])
            if len(self.history) > 20:
                self.history.pop(0)
        elif proposed[0] == self.current[0]:
            self.current = proposed
        return self.current

    @staticmethod
    def _propose(scene) -> tuple[str, str]:
        if scene.min_in_zone_distance_m is not None:
            d = scene.min_in_zone_distance_m
            cls = scene.closest_in_zone.class_name if scene.closest_in_zone else "object"
            if d <= LEAD_STOP_DISTANCE_M:
                return ("STOP", f"{cls} at {d:.1f}m in path")
            if d <= LEAD_YIELD_DISTANCE_M:
                return ("SLOW DOWN", f"{cls} at {d:.1f}m ahead")
        return ("GO", "Path clear")
