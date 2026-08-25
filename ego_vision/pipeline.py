import argparse
import sys
import time
from pathlib import Path

import cv2

from ego_vision.config.settings import (
    DEFAULT_CONF,
    DEFAULT_IOU,
    DEFAULT_MODEL,
    HYSTERESIS_N,
)
from ego_vision.decision.rule_engine import RuleEngine
from ego_vision.ingest.video_reader import VideoReader
from ego_vision.perception.detector import YoloDetector
from ego_vision.perception.depth_estimator import DepthEstimator
from ego_vision.perception.inference_resizer import InferenceResizer
from ego_vision.reasoning.distance import sample_distance
from ego_vision.reasoning.ego_zone import make_ego_zone
from ego_vision.reasoning.scene_state import assemble
from ego_vision.viz.hud import (
    HudState,
    draw_debug_strip,
    draw_ego_zone,
    draw_hud,
    draw_status_panel,
)
from ego_vision.viz.bev_map import draw_bev
from ego_vision.viz.overlay import draw_detections

WINDOW_NAME = "Ego Vision"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detect + track + rule engine + HUD")

    # Input / output
    p.add_argument("--input", "-i", required=True, type=Path, help="Input video file")
    p.add_argument("--output", "-o", type=Path, default=None, help="Annotated output video (optional)")
    p.add_argument("--show", action="store_true", help="Display annotated frames in a window")
    p.add_argument("--resize-width", type=int, default=1280,
                   help="Downscale input to this width for inference (preserves aspect ratio). "
                        "Display/save stay at native. Default 1280. Pass 0 to keep native for inference too")

    # Detection and tracking
    p.add_argument("--model", default=DEFAULT_MODEL, help="YOLO weights (default is yolo11m.pt)")
    p.add_argument("--conf", type=float, default=DEFAULT_CONF)
    p.add_argument("--iou", type=float, default=DEFAULT_IOU)
    p.add_argument("--device", default='cuda:0', help="e.g. 'cpu', '0', 'cuda:0'")
    p.add_argument("--every", type=int, default=1, help="Run detection every Nth frame")

    # Depth
    p.add_argument("--depth-every", type=int, default=3,
                   help="Run depth every Nth frame; reuse last map between samples (default 3)")

    # reasoning
    p.add_argument("--hysteresis", type=int, default=HYSTERESIS_N,
                   help="Frames an action must hold before switching")

    # Visualization
    p.add_argument("--no-ego-zone", action="store_true", help="Hide the ego zone overlay")
    p.add_argument("--no-debug", action="store_true", help="Hide the bottom debug strip")
    p.add_argument("--no-conf", action="store_true", help="Hide confidence scores on detection labels")

    return p.parse_args(argv)


def run(args):
    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 2
    if not args.show and args.output is None:
        print("Nothing to do: pass --show, --output, or both.", file=sys.stderr)
        return 2

    detector = YoloDetector(
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        track=True,
    )
    depth_estimator = DepthEstimator()
    engine = RuleEngine(hysteresis_n=args.hysteresis)

    reader = VideoReader(args.input)
    meta = reader.meta
    resizer = InferenceResizer(meta.width, meta.height, args.resize_width)
    ego_zone = make_ego_zone(meta.width, meta.height)
    print("Decision mode: distance-only (ego zone)")

    writer: cv2.VideoWriter | None = None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.output), fourcc, meta.fps, (meta.width, meta.height))
        if not writer.isOpened():
            reader.release()
            print(f"Could not open writer for {args.output}", file=sys.stderr)
            return 3

    if args.show:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    t0 = time.time()
    last_dets = []
    last_depth_map = None
    paused = False
    try:
        for idx, frame in reader.frames():
            start_one_frame = time.time()
            inference_frame = resizer.to_inference(frame)

            if idx % args.every == 0:
                start_det = time.time()
                last_dets = detector.detect(inference_frame)
                print(f"Detection took {time.time() - start_det:.2f} sec", flush=True)
                for d in last_dets:
                    d.box_xyxy = resizer.scale_box_to_native(d.box_xyxy)

            if idx % args.depth_every == 0 or last_depth_map is None:
                start_depth = time.time()
                last_depth_map = resizer.scale_depth_to_native(
                    depth_estimator.estimate_bgr(inference_frame)
                )
                print(f"Depth estimation took {time.time() - start_depth:.2f} sec")
            for d in last_dets:
                d.distance_m = sample_distance(last_depth_map, d.box_xyxy)

            scene = assemble(last_dets, frame.shape, ego_zone)
            action, reason = engine.decide(scene)

            # Order matters: boxes, then zone (so zone tints over boxes),
            # then HUD card (so HUD never sits under a box).
            draw_detections(frame, last_dets, show_conf=not args.no_conf)
            if not args.no_ego_zone:
                draw_ego_zone(frame, ego_zone)

            elapsed = time.time() - t0
            proc_fps = (idx + 1) / elapsed if elapsed > 0 else 0.0
            hud = HudState(
                action=action,
                reason=reason,
                frame_idx=idx,
                proc_fps=proc_fps,
                recent_actions=engine.history,
                track_count=sum(1 for d in last_dets if d.track_id is not None),
                device=args.device,
            )
            frame = draw_hud(frame, hud)
            frame = draw_status_panel(frame, hud)
            frame = draw_bev(frame, last_dets)
            if not args.no_debug:
                frame = draw_debug_strip(frame, hud)

            if writer is not None:
                writer.write(frame)

            if args.show:
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(0 if paused else 1) & 0xFF
                if key in (ord("q"), 27):
                    print("Quit requested.")
                    break
                if key == ord(" "):
                    paused = not paused

            if idx % 30 == 0:
                print(f"frame {idx:>6}  proc {proc_fps:5.1f} fps  "
                      f"action={action}  reason={reason}", flush=True)
            print(f"Total time for one frame: {time.time() - start_one_frame:.2f} sec", flush=True)
    finally:
        reader.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    if args.output is not None:
        print(f"Wrote {args.output}")
    return 0


def main() -> None:
    sys.exit(run(parse_args()))


if __name__ == "__main__":
    main()
