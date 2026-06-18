"""Priority rule engine with hysteresis.

Priority order (first match wins):
  1. EMERGENCY BRAKE: TTC below the hard threshold
  2. STOP: red light, stop sign, VRU in zone, or close lead vehicle
  3. SLOW DOWN: yellow light, VRU near zone, or moderate-distance lead
  4. GO: otherwise

Hysteresis: a new action must hold for N consecutive proposals before it
replaces the displayed one.
"""

from collections import deque

from ego_vision.config.settings import (
    EMERGENCY_TTC_SEC,
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
        # Emergency Brake.
        if (
            scene.min_in_zone_ttc_sec is not None
            and scene.min_in_zone_ttc_sec < EMERGENCY_TTC_SEC
        ):
            return ("EMERGENCY BRAKE", f"Time to Collision {scene.min_in_zone_ttc_sec:.1f}s")

        # Stop.
        if scene.vru_in_zone:
            return ("STOP", f"{scene.vru_in_zone[0].class_name} in path")
        if scene.stop_signs:
            return ("STOP", "Stop sign ahead")
        if scene.light_ahead == "red":
            return ("STOP", "Red light")
        if (
            scene.lead_vehicle is not None
            and scene.lead_distance_m is not None
            and scene.lead_distance_m <= LEAD_STOP_DISTANCE_M
        ):
            return ("STOP", f"Lead vehicle {scene.lead_distance_m:.1f}m")

        # Slow down.
        if scene.light_ahead == "yellow":
            return ("SLOW DOWN", "Yellow light")
        if scene.vru_near_zone:
            return ("SLOW DOWN", f"{scene.vru_near_zone[0].class_name} approaching")
        if (
            scene.lead_vehicle is not None
            and scene.lead_distance_m is not None
            and scene.lead_distance_m <= LEAD_YIELD_DISTANCE_M
        ):
            return ("SLOW DOWN", f"Lead vehicle {scene.lead_distance_m:.1f}m")

        # Go.
        return ("GO", "Path clear")
