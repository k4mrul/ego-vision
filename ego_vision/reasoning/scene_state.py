"""Per-frame scene record: the input to the rule engine."""

from dataclasses import dataclass

from ego_vision.reasoning.ego_zone import in_zone


@dataclass
class SceneState:
    detections: list
    ego_zone: object
    min_in_zone_distance_m: float | None
    closest_in_zone: object | None


def assemble(detections, frame_shape, ego_zone) -> SceneState:
    candidates = [
        d for d in detections
        if d.distance_m is not None and in_zone(d.box_xyxy, ego_zone)
    ]

    closest = None
    min_d = None
    if candidates:
        closest = min(candidates, key=lambda d: d.distance_m)
        min_d = closest.distance_m

    return SceneState(
        detections=detections,
        ego_zone=ego_zone,
        min_in_zone_distance_m=min_d,
        closest_in_zone=closest,
    )
