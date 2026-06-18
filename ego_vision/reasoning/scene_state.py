"""Per-frame scene record: the input to the rule engine."""

from dataclasses import dataclass

from ego_vision.config.settings import VEHICLE_CLASSES, VRU_CLASSES
from ego_vision.reasoning.ego_zone import in_zone, near_zone


@dataclass
class SceneState:
    detections: list
    ego_zone: object
    vru_in_zone: list
    vru_near_zone: list
    stop_signs: list
    light_ahead: str | None
    lead_vehicle: object
    lead_distance_m: float | None
    min_in_zone_ttc_sec: float | None


def _box_area(d) -> float:
    x1, y1, x2, y2 = d.box_xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def assemble(detections, frame_shape, ego_zone) -> SceneState:
    vru_in = []
    vru_near = []
    vehicles_in = []
    stop_signs = []
    lights = []

    for d in detections:
        if d.class_name in VRU_CLASSES:
            if in_zone(d.box_xyxy, ego_zone):
                vru_in.append(d)
            elif near_zone(d.box_xyxy, ego_zone):
                vru_near.append(d)
        elif d.class_name in VEHICLE_CLASSES:
            if in_zone(d.box_xyxy, ego_zone):
                vehicles_in.append(d)
        elif d.class_name == "stop sign":
            stop_signs.append(d)
        elif d.class_name == "traffic light" and d.light_state in ("red", "yellow", "green"):
            lights.append(d)

    light_ahead = None
    if lights:
        biggest = max(lights, key=_box_area)
        light_ahead = biggest.light_state

    lead = None
    lead_distance_m = None
    vehicles_with_d = [v for v in vehicles_in if v.distance_m is not None]
    if vehicles_with_d:
        lead = min(vehicles_with_d, key=lambda v: v.distance_m)
        lead_distance_m = lead.distance_m

    in_zone_dets = vru_in + vehicles_in
    ttcs = [d.ttc_sec for d in in_zone_dets if d.ttc_sec is not None]
    min_ttc = min(ttcs) if ttcs else None

    return SceneState(
        detections=detections,
        ego_zone=ego_zone,
        vru_in_zone=vru_in,
        vru_near_zone=vru_near,
        stop_signs=stop_signs,
        light_ahead=light_ahead,
        lead_vehicle=lead,
        lead_distance_m=lead_distance_m,
        min_in_zone_ttc_sec=min_ttc,
    )
