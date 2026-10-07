"""
Wrong-Way & Lane Violation Detection Pipeline
=============================================
Detect vehicles (YOLOv8 + ByteTrack)
→ map to 4-point LaneZone polygon (P1..P4) with legal direction vector
→ evaluate directional displacement & forbidden entry sides
→ draw directional bounding boxes (Green: IN, Yellow: OUT, Red: VIOLATION)
→ draw anchor points and lane guide points with entry-to-exit direction arrow
→ capture evidence snapshots & log to CSV / SQLite
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional

import cv2

from src.lane_zone import LaneZone
from src.motion_tracker import MotionTracker
from src.tracker import build_detector_and_tracker
from src.utils import (
    FPSCounter,
    FrameIngester,
    draw_hud,
    draw_tracked_vehicle,
    load_config,
    resize_for_display,
    resize_to_fit,
    setup_logging,
)
from src.violation_detector import ViolationDetector
from src.violation_reporter import ViolationReporter

logger = logging.getLogger(__name__)


class LaneViolationPipeline:
    """Dedicated Wrong-Way & Lane Violation detection pipeline."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.lane_zone: Optional[LaneZone] = None

        motion_cfg = cfg.get("motion", {})
        self.motion_tracker = MotionTracker(
            history_size=int(motion_cfg.get("history_size", 20)),
            stale_after_frames=int(motion_cfg.get("stale_after_frames", 90)),
        )

        self.violation_detector = ViolationDetector(cfg)
        self.reporter = ViolationReporter(cfg)
        self.fps_counter = FPSCounter(
            window_size=int(cfg.get("display", {}).get("fps_window", 30))
        )

        self.vehicle_detector = None
        self.vehicle_tracker = None

    def _update_tracking_runtime_params(self, processing_fps: float) -> None:
        bt_cfg = self.cfg["bytetrack"]
        bt_cfg["frame_rate"] = max(1, int(round(processing_fps)))

        if str(bt_cfg.get("lost_track_buffer", "auto")) == "auto":
            bt_cfg["lost_track_buffer"] = max(30, int(round(processing_fps * 2.0)))

        logger.info(
            "[Pipeline] ByteTrack params | frame_rate=%d | lost_track_buffer=%d",
            bt_cfg["frame_rate"],
            bt_cfg["lost_track_buffer"],
        )

    def _reason_tag(self, reason: str) -> str:
        if reason == "opposite_direction":
            return "WRONG-WAY"
        if reason == "forbidden_entry_side":
            return "ENTRY-VIOL"
        if reason == "speeding":
            return "SPEEDING"
        return "VIOLATION"

    def run(self, max_frames: Optional[int] = None) -> None:
        display_cfg = self.cfg.get("display", {})
        show_window = bool(display_cfg.get("show_window", True))
        window_name = str(display_cfg.get("window_name", "Lane Violation Detection"))
        display_scale = float(display_cfg.get("display_scale", 0.7))
        fixed_window_size = bool(display_cfg.get("fixed_window_size", True))
        window_width = int(display_cfg.get("window_width", 1280))
        window_height = int(display_cfg.get("window_height", 720))
        line_thickness = int(display_cfg.get("line_thickness", 2))

        normal_color = tuple(display_cfg.get("bbox_color_normal", [0, 255, 0]))
        out_of_lane_color = tuple(display_cfg.get("bbox_color_out_of_lane", [255, 180, 0]))
        violation_color = tuple(display_cfg.get("bbox_color_violation", [0, 0, 255]))
        lane_color = tuple(display_cfg.get("lane_polygon_color", [0, 255, 255]))

        with FrameIngester(self.cfg) as ingester:
            self.lane_zone = LaneZone.from_config(
                self.cfg,
                frame_size=(ingester.frame_width, ingester.frame_height),
            )
            assert self.lane_zone is not None
            self._update_tracking_runtime_params(ingester.processing_fps)
            self.vehicle_detector, self.vehicle_tracker = build_detector_and_tracker(self.cfg)

            self.reporter.open(
                width=ingester.frame_width,
                height=ingester.frame_height,
                fps=max(1.0, ingester.processing_fps),
            )

            if show_window:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                if fixed_window_size:
                    cv2.resizeWindow(window_name, window_width, window_height)

            for frame_idx, timestamp_sec, frame in ingester:
                if max_frames is not None and frame_idx >= max_frames:
                    logger.info("[Pipeline] Reached max_frames=%d — stopping.", max_frames)
                    break

                detections = self.vehicle_detector.detect(frame)
                tracked, lost_ids = self.vehicle_tracker.update(detections)

                annotated = frame.copy()
                self.lane_zone.draw(
                    annotated,
                    polygon_color=lane_color,
                    thickness=line_thickness,
                    draw_points=bool(display_cfg.get("draw_lane_points", True)),
                )

                active_ids = set()

                for vehicle in tracked:
                    track_id = vehicle.track_id
                    active_ids.add(track_id)

                    bbox = vehicle.bbox
                    lane_anchor = ((bbox[0] + bbox[2]) // 2, bbox[3])
                    inside_lane = self.lane_zone.contains(lane_anchor)

                    self.motion_tracker.update(
                        track_id=track_id,
                        centroid=lane_anchor,
                        frame_idx=frame_idx,
                        inside_lane=inside_lane,
                    )

                    event = None
                    if inside_lane:
                        event = self.violation_detector.evaluate(
                            track_id=track_id,
                            bbox=bbox,
                            centroid=lane_anchor,
                            frame_idx=frame_idx,
                            lane_zone=self.lane_zone,
                            motion_tracker=self.motion_tracker,
                            fps=ingester.processing_fps,
                        )

                    existing_event = self.violation_detector.get_event(track_id)
                    if existing_event is not None:
                        color = violation_color
                        subtitle = self._reason_tag(existing_event.reason)
                    elif inside_lane:
                        color = normal_color
                        progress = self.lane_zone.progress_ratio(lane_anchor)
                        subtitle = f"IN:{progress:+.2f}"
                    else:
                        color = out_of_lane_color
                        subtitle = "OUT"

                    draw_tracked_vehicle(
                        annotated,
                        track_id=track_id,
                        bbox=bbox,
                        color=color,
                        thickness=line_thickness,
                        subtitle=subtitle,
                    )
                    cv2.circle(annotated, lane_anchor, 3, color, -1, lineType=cv2.LINE_AA)

                    if event is not None:
                        self.reporter.report_violation(event, annotated, timestamp_sec)

                for lost_id in lost_ids:
                    self.motion_tracker.remove(lost_id)

                self.motion_tracker.cleanup(active_ids=active_ids, current_frame_idx=frame_idx)

                fps_value = self.fps_counter.tick()
                draw_hud(
                    annotated,
                    fps=fps_value,
                    frame_idx=frame_idx,
                    active_tracks=len(tracked),
                    total_violations=self.violation_detector.total_violations(),
                )

                self.reporter.write_frame(annotated)

                if show_window:
                    if fixed_window_size:
                        preview = resize_to_fit(annotated, window_width, window_height)
                    else:
                        preview = resize_for_display(annotated, display_scale)
                    cv2.imshow(window_name, preview)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        logger.info("[Pipeline] 'q' pressed. Stopping.")
                        break

            if show_window:
                cv2.destroyAllWindows()

            self.reporter.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lane violation detection pipeline (wrong-way detection with 4-point LaneZone)",
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=None,
        help="Video source path, webcam index (0/1/2), or RTSP URL",
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    parser.add_argument("--no-display", action="store_true", help="Disable preview window")
    parser.add_argument("--save-video", action="store_true", help="Force save annotated video")
    parser.add_argument("--no-save-video", action="store_true", help="Disable annotated video saving")
    parser.add_argument("--decimation", type=int, default=None, help="Override frame decimation")
    parser.add_argument("--fps", type=float, default=None, help="Override source FPS")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after processing N frames")
    parser.add_argument("--csv", type=str, default=None, help="Override CSV output path")
    parser.add_argument("--annotated-video", type=str, default=None, help="Override annotated video path")
    return parser.parse_args()


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> None:
    if args.source is not None:
        cfg["input"]["source"] = args.source

    if args.no_display:
        cfg["display"]["show_window"] = False

    if args.save_video:
        cfg["output"]["save_annotated_video"] = True

    if args.no_save_video:
        cfg["output"]["save_annotated_video"] = False

    if args.decimation is not None:
        cfg["input"]["frame_decimation"] = max(1, int(args.decimation))

    if args.fps is not None and args.fps > 0:
        cfg["input"]["force_fps"] = float(args.fps)

    if args.csv is not None:
        cfg["paths"]["csv_log"] = args.csv

    if args.annotated_video is not None:
        cfg["paths"]["annotated_video"] = args.annotated_video


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    apply_cli_overrides(cfg, args)

    setup_logging(cfg)
    logger_main = logging.getLogger(__name__)

    logger_main.info("Starting Lane Violation Pipeline (Wrong-Way Detection)")
    logger_main.info("Input source: %s", cfg["input"]["source"])

    pipeline = LaneViolationPipeline(cfg)
    pipeline.run(max_frames=args.max_frames)

    logger_main.info("Lane violation pipeline finished successfully.")


if __name__ == "__main__":
    main()
