"""
Standalone Speed Violation Detection Pipeline
=============================================
Focuses specifically on Perspective Homography IPM Speed Estimation
and Speeding Violation Detection.

Usage:
    python speed_violation.py [video_path] [--config config.yaml] [--no-display]
"""

from __future__ import annotations

import argparse
import csv
import datetime
import logging
import sqlite3
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from src.speed_estimator import PerspectiveSpeedEstimator
from src.tracker import TrackedVehicle, build_detector_and_tracker
from src.utils import (
    FPSCounter,
    FrameIngester,
    draw_hud,
    draw_tracked_vehicle,
    load_config,
    resize_to_fit,
    setup_logging,
    trigger_beep,
)

logger = logging.getLogger(__name__)


class SpeedViolationPipeline:
    """Dedicated standalone pipeline for vehicle speed measurement and speeding violation detection."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg

        speed_cfg = cfg.get("speed", {})
        self._speed_limit_kmh = float(speed_cfg.get("speed_limit_kmh", 60.0))
        self._speed_tolerance_kmh = float(speed_cfg.get("speed_tolerance_kmh", 5.0))
        self._min_track_frames = int(speed_cfg.get("min_track_frames", 8))
        self._min_sustained_frames = int(speed_cfg.get("min_sustained_frames", 6))
        self._min_measurement_y = float(speed_cfg.get("min_measurement_y", 260.0))
        source_poly = speed_cfg.get(
            "source_polygon",
            [[750, 180], [1350, 180], [1850, 980], [450, 980]],
        )
        self._source_polygon = np.array(source_poly, dtype=np.int32)
        ground_w = float(speed_cfg.get("ground_width_meters", 8.0))
        ground_l = float(speed_cfg.get("ground_length_meters", 30.0))
        window_sec = float(speed_cfg.get("window_duration_sec", 0.5))

        self.speed_estimator = PerspectiveSpeedEstimator(
            source_polygon=source_poly,
            ground_width_meters=ground_w,
            ground_length_meters=ground_l,
            speed_limit_kmh=self._speed_limit_kmh,
            speed_tolerance_kmh=self._speed_tolerance_kmh,
            min_track_frames=self._min_track_frames,
            min_sustained_frames=self._min_sustained_frames,
            min_measurement_y=self._min_measurement_y,
            window_duration_sec=window_sec,
        )
        self.fps_counter = FPSCounter()

        paths_cfg = cfg.get("paths", {})
        self._output_dir = Path(paths_cfg.get("output_dir", "outputs"))
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._snapshot_dir = self._output_dir / "speed_violations"
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)

        self._csv_path = Path(paths_cfg.get("speed_csv", "outputs/speed_violations.csv"))
        self._db_path = Path(paths_cfg.get("db_log", "outputs/detections.db"))
        self._video_path = Path(paths_cfg.get("annotated_video", "outputs/annotated_speed.mp4"))

        self._csv_file = None
        self._csv_writer = None
        self._db_conn = None
        self._video_writer = None

        self._confirmed_violators: set[int] = set()

    def _open_outputs(self, width: int, height: int, fps: float) -> None:
        write_header = not self._csv_path.exists() or self._csv_path.stat().st_size == 0
        self._csv_file = open(self._csv_path, mode="a", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(
            self._csv_file,
            fieldnames=[
                "pc_date",
                "pc_time",
                "timestamp_sec",
                "frame_idx",
                "track_id",
                "speed_kmh",
                "peak_speed_kmh",
                "speed_limit_kmh",
                "excess_kmh",
                "snapshot_path",
            ],
        )
        if write_header:
            self._csv_writer.writeheader()
            self._csv_file.flush()

        self._db_conn = sqlite3.connect(str(self._db_path))
        self._db_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS detections (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                pc_date        TEXT,
                pc_time        TEXT,
                timestamp_sec  REAL,
                frame_idx      INTEGER,
                track_id       INTEGER,
                plate_bn       TEXT,
                plate_en       TEXT,
                plate_conf     REAL,
                speed_kmh      REAL,
                peak_speed_kmh REAL,
                speeding       INTEGER,
                lane_violation TEXT,
                violation_type TEXT,
                x1             INTEGER,
                y1             INTEGER,
                x2             INTEGER,
                y2             INTEGER,
                plate_crop_path TEXT,
                snapshot_path  TEXT
            )
        """
        )
        self._db_conn.commit()

        save_video = bool(self.cfg.get("output", {}).get("save_annotated_video", True))
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._video_writer = cv2.VideoWriter(str(self._video_path), fourcc, fps, (width, height))

    def _close_outputs(self) -> None:
        if self._video_writer:
            self._video_writer.release()
        if self._csv_file:
            self._csv_file.close()
        if self._db_conn:
            self._db_conn.close()

    def _trigger_alert(self, speed_kmh: float) -> None:
        trigger_beep(1200, 200)

    def run(self, max_frames: Optional[int] = None) -> None:
        display_cfg = self.cfg.get("display", {})
        show_window = bool(display_cfg.get("show_window", True))
        window_name = "Speed Violation Detection"
        line_thickness = int(display_cfg.get("line_thickness", 2))

        normal_color = tuple(display_cfg.get("bbox_color_normal", [0, 255, 0]))
        violation_color = tuple(display_cfg.get("bbox_color_violation", [0, 0, 255]))

        with FrameIngester(self.cfg) as ingester:
            self.speed_estimator = PerspectiveSpeedEstimator.from_config(
                self.cfg, frame_size=(ingester.frame_width, ingester.frame_height)
            )
            self._source_polygon = np.array(self.speed_estimator.source_polygon, dtype=np.int32)

            detector, tracker = build_detector_and_tracker(self.cfg)
            self._open_outputs(ingester.frame_width, ingester.frame_height, max(1.0, ingester.processing_fps))

            if show_window:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 1280, 720)

            total_violations = 0
            frame_idx = 0

            for frame_idx, timestamp_sec, frame in ingester:
                if max_frames is not None and frame_idx >= max_frames:
                    break

                detections = detector.detect(frame)
                tracked, lost_ids = tracker.update(detections)
                annotated = frame.copy()

                # Draw calibrated road speed polygon
                if hasattr(self, "_source_polygon") and self._source_polygon is not None:
                    cv2.polylines(annotated, [self._source_polygon], isClosed=True, color=(255, 120, 0), thickness=2)

                for vehicle in tracked:
                    tid = vehicle.track_id
                    bbox = vehicle.bbox
                    anchor = ((bbox[0] + bbox[2]) / 2.0, float(bbox[3]))

                    # Update speed estimator
                    self.speed_estimator.update_track(tid, anchor, timestamp_sec, bbox=bbox)
                    eff_speed = self.speed_estimator.get_effective_speed(tid)

                    is_speeding = self.speed_estimator.is_confirmed_speeding(tid)
                    if is_speeding and tid not in self._confirmed_violators:
                        self._confirmed_violators.add(tid)
                        total_violations += 1
                        self._trigger_alert(eff_speed)

                        # Capture snapshot
                        now_dt = datetime.datetime.now()
                        snap_name = f"speed_viol_track_{tid}_frame_{frame_idx}.png"
                        snap_path = self._snapshot_dir / snap_name

                        x1, y1, x2, y2 = bbox
                        pad = 20
                        h, w = frame.shape[:2]
                        crop = frame[max(0, y1 - pad) : min(h, y2 + pad), max(0, x1 - pad) : min(w, x2 + pad)].copy()
                        cv2.imwrite(str(snap_path), crop)

                        if self._csv_writer and self._csv_file:
                            self._csv_writer.writerow(
                                {
                                    "pc_date": now_dt.strftime("%Y-%m-%d"),
                                    "pc_time": now_dt.strftime("%H:%M:%S"),
                                    "timestamp_sec": f"{timestamp_sec:.2f}",
                                    "frame_idx": frame_idx,
                                    "track_id": tid,
                                    "speed_kmh": f"{eff_speed:.1f}",
                                    "peak_speed_kmh": f"{self.speed_estimator.get_peak_speed(tid):.1f}",
                                    "speed_limit_kmh": f"{self._speed_limit_kmh:.0f}",
                                    "excess_kmh": f"{max(0.0, eff_speed - self._speed_limit_kmh):.1f}",
                                    "snapshot_path": str(snap_path),
                                }
                            )
                            self._csv_file.flush()

                        if self._db_conn:
                            try:
                                self._db_conn.execute(
                                    """
                                    INSERT INTO detections (
                                        pc_date, pc_time, timestamp_sec, frame_idx, track_id,
                                        plate_bn, plate_en, plate_conf, speed_kmh, peak_speed_kmh,
                                        speeding, lane_violation, violation_type,
                                        plate_crop_path, snapshot_path
                                    )
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                    (
                                        now_dt.strftime("%Y-%m-%d"),
                                        now_dt.strftime("%H:%M:%S"),
                                        timestamp_sec,
                                        frame_idx,
                                        tid,
                                        "",
                                        "",
                                        0.0,
                                        eff_speed,
                                        self.speed_estimator.get_peak_speed(tid),
                                        1,
                                        "",
                                        "SPEEDING",
                                        "",
                                        str(snap_path),
                                    ),
                                )
                                self._db_conn.commit()
                            except Exception as e:
                                logger.warning("Could not log speed violation to DB: %s", e)

                    color = violation_color if is_speeding else normal_color
                    status_lbl = f"{eff_speed:.0f} km/h | 🚨SPEEDING" if is_speeding else (f"{eff_speed:.0f} km/h" if eff_speed > 0 else "")
                    draw_tracked_vehicle(annotated, tid, bbox, color, line_thickness, status_lbl)

                for lost_id in lost_ids:
                    self.speed_estimator.remove_track(lost_id)

                fps_val = self.fps_counter.tick()
                draw_hud(
                    annotated,
                    fps_val,
                    frame_idx,
                    len(tracked),
                    total_violations,
                    extra={"Speed Limit": f"{self._speed_limit_kmh:.0f} km/h"},
                )

                if self._video_writer:
                    self._video_writer.write(annotated)

                if show_window:
                    display_frame = resize_to_fit(annotated, 1280, 720)
                    cv2.imshow(window_name, display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break

            if show_window:
                cv2.destroyAllWindows()
            self._close_outputs()

            logger.info("Speed violation run complete. Total violations: %d", total_violations)


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone Speed Violation Detection Pipeline")
    parser.add_argument("source", nargs="?", default=None, help="Video file or stream")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--no-display", action="store_true", help="Disable preview window")
    parser.add_argument("--max-frames", type=int, default=None, help="Process max N frames")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.source:
        cfg["input"]["source"] = args.source
    if args.no_display:
        cfg["display"]["show_window"] = False

    setup_logging(cfg)
    pipeline = SpeedViolationPipeline(cfg)
    pipeline.run(max_frames=args.max_frames)


if __name__ == "__main__":
    main()
