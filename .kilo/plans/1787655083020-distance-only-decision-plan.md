# Simplify to distance-only action decision

## Goal

Remove lane detection, traffic-light color classification, and time-to-collision.
Keep YOLO detection + ByteTrack tracking + Depth Anything V2 distance. The
decision becomes a single question: is any tracked object inside the ego-zone
trapezoid close enough to require an action?

Decision = minimum distance among tracked objects whose contact point falls
inside the ego zone:
- `min_d <= LEAD_STOP_DISTANCE_M` -> **STOP**
- `min_d <= LEAD_YIELD_DISTANCE_M` -> **SLOW DOWN**
- otherwise -> **GO**

If no tracked object is in the zone with a valid distance, action is **GO**.

## Out of scope

- Lane detection model integration (already noted as "still being worked on" in README; keep `lane_detector.py` out of imports but leave the file as-is, or delete — choose delete per step 5).
- Speed estimation (`reasoning/speed.py` already disabled).
- BEV mini-map continues to render all tracked objects for spatial context.
- Hysteresis on the rule engine (kept; just simpler input).

## Files to change

### 1. `ego_vision/pipeline.py` — drop lane/light/TTC wiring
- Remove imports:
  - `from ego_vision.perception.light_classifier import classify_light, crop_box`
  - `from ego_vision.reasoning.ttc import TtcEstimator`
- Remove instantiation of `TtcEstimator`.
- Remove the inner loop that calls `classify_light(crop_box(...))` and assigns `det.light_state`.
- Remove the call `ttc_estimator.update(last_dets)`.
- Keep everything else: detector, depth, ego zone, scene assemble, rule engine, viz.
- Add a one-liner log on startup so the user sees the simplified path:
  `print("Decision mode: distance-only (ego zone)")`.

### 2. `ego_vision/reasoning/scene_state.py` — distance-only scene
- Remove imports of `VRU_CLASSES`, `VEHICLE_CLASSES`, `in_zone`, `near_zone` usage that splits by class group.
- Replace `SceneState` dataclass fields with the minimum needed:
  - `detections: list`
  - `ego_zone: object`
  - `min_in_zone_distance_m: float | None`
  - `closest_in_zone: object | None`
- Rewrite `assemble(detections, frame_shape, ego_zone)` to:
  - iterate detections with `distance_m is not None`
  - keep ones where `in_zone(d.box_xyxy, ego_zone)` is true
  - pick the one with the smallest `distance_m`
  - return `SceneState(...)` with that min and the det itself (for reason text)
- Delete `vru_in_zone`, `vru_near_zone`, `stop_signs`, `light_ahead`, `lead_vehicle`, `lead_distance_m`, `min_in_zone_ttc_sec` from the dataclass.
- Keep `in_zone` import; drop `near_zone` use (still used by no one after this change — leave the function in `ego_zone.py` since the file's `near_zone` is only referenced from scene_state).

### 3. `ego_vision/decision/rule_engine.py` — distance-only rules
- Update module docstring to reflect new priority (first match wins):
  1. STOP — any in-zone object within `LEAD_STOP_DISTANCE_M`
  2. SLOW DOWN — any in-zone object within `LEAD_YIELD_DISTANCE_M`
  3. GO — otherwise
- Remove imports of `EMERGENCY_TTC_SEC` (no longer referenced).
- Rewrite `_propose`:
  ```
  if scene.min_in_zone_distance_m is not None:
      d = scene.min_in_zone_distance_m
      cls = scene.closest_in_zone.class_name if scene.closest_in_zone else "object"
      if d <= LEAD_STOP_DISTANCE_M:
          return ("STOP", f"{cls} at {d:.1f}m in path")
      if d <= LEAD_YIELD_DISTANCE_M:
          return ("SLOW DOWN", f"{cls} at {d:.1f}m ahead")
  return ("GO", "Path clear")
  ```
- Keep `RuleEngine` class + hysteresis behavior intact.

### 4. `ego_vision/config/settings.py` — small cleanups
- Remove unused constants if no remaining references:
  - `EMERGENCY_TTC_SEC` (delete; TTC is gone).
- Keep `LEAD_STOP_DISTANCE_M`, `LEAD_YIELD_DISTANCE_M`, the `EGO_ZONE_*` settings, `EGO_NEAR_MARGIN_PX` (still used by HUD/ego-zone drawing; verify), `HYSTERESIS_N`, depth/BEV/overlay constants.
- Grep before deleting to confirm zero remaining references.

### 5. Delete unused modules
- Delete `ego_vision/perception/light_classifier.py`.
- Delete `ego_vision/perception/lane_detector.py`.
- Delete `ego_vision/reasoning/ttc.py`.
- Delete `ego_vision/reasoning/speed.py` (already disabled in pipeline; matches the new "distance-only" stance).
- Remove stale `__pycache__/` entries: not needed (Python regenerates), but mention in case the user wants a clean tree.

### 6. `ego_vision/viz/overlay.py` — light-state label cleanup
- Drop the `LIGHT_COLORS` reference and the "TRAFFIC LIGHT red/yellow/green" label suffix.
- The label becomes: `class_name + " " + distance` (+ optional conf).
- Color logic simplifies to `GROUP_COLORS.get(det.class_group, (255, 255, 255))`.
- Imports: drop `LIGHT_COLORS` import.

### 7. `ego_vision/viz/hud.py` — drop TTC / light-related HUD fields
- `HudState` currently exposes `light_ahead`. Drop that field and any rendering that uses it (`draw_hud`, `draw_status_panel`, `draw_debug_strip`).
- Drop the "TTC" line in the debug strip if present.
- Keep the system status panel but drop the traffic-light subsystem row.

### 8. `README.md` — sync docs
- Rewrite the "How it works" table:
  | Step | What it does | How |
  |---|---|---|
  | Find objects | Detect road users (vehicles, people, animals, signs, lights) | YOLO11 |
  | Track objects | Track across frames | ByteTrack |
  | Measure distance | Meters per tracked object | Depth Anything V2 |
  | Decide | STOP / SLOW DOWN / GO from the closest in-zone distance | Single rule on min distance |
  | Draw | Boxes, IDs, distances, action overlay | OpenCV |
- Rewrite "How it decides" to the new 3-tier distance-only table.
- Drop references to traffic-light color, lane detection, TTC, BEV-as-decision-input (BEV still drawn).
- Remove the `EGO_NEAR_MARGIN_PX` row from the settings table if `near_zone` is no longer called.
- Update the project-layout tree to remove `light_classifier.py`, `lane_detector.py`, `ttc.py`, `speed.py`.
- Update "To Do" lane-detection bullet wording (still aspirational, but note lane integration is now de-prioritized in favor of distance-only).

## Validation

1. `python -c "import ego_vision.pipeline"` — import sanity.
2. `python -m ego_vision.pipeline -i data/video.mp4 --show` — full run, confirm:
   - No `classify_light` or TTC log lines.
   - HUD shows STOP / SLOW DOWN / GO with messages like "car at 6.4m in path".
   - No crash on a video that has no detections (action stays GO).
3. Add a tiny synthetic test (optional): a `pytest`-style test that builds a `SceneState` with `min_in_zone_distance_m = 5.0` and asserts `RuleEngine().decide(scene)[0] == "STOP"`. Place at `tests/test_decision.py`.
4. Re-run on a previously-tested clip and confirm the action still flips GO -> SLOW DOWN -> STOP -> GO when a vehicle approaches.

## Risks

- Removing `light_ahead` from `HudState` will break any external consumer (none in this repo, but worth grepping for `light_ahead`).
- `EGO_NEAR_MARGIN_PX` becomes unused once `near_zone` is no longer called; leaving the constant is safe but it's dead code.
- `class_group`, `GROUP_COLORS`, and `CONTROL_CLASSES` are still used by overlay for box coloring; keep them.
- `in_zone` from `ego_zone.py` stays (still needed by `scene_state`).

## Open questions

None — user confirmed: distance-only in ego zone, drop TTC, keep ego-zone overlay.
