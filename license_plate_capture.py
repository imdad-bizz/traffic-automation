"""
Standalone License Plate Capture & Recognition Pipeline
========================================================
Focuses specifically on vehicle detection, Bengali/English number plate
localization, cropping, enhancement, and isolated plate snap harvesting.

Usage:
    python license_plate_capture.py [video_path] [--config config.yaml] [--no-display]
"""

from __future__ import annotations

import argparse
import csv
import datetime
import logging
import sqlite3
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.plate_detector import (
    BanglaPlateOCR,
    LicensePlateDetector,
    PlatePreprocessor,
    TrackPlateManager,
)
from src.tracker import TrackedVehicle, build_detector_and_tracker
from src.utils import (
    FPSCounter,
    FrameIngester,
    draw_hud,
    draw_tracked_vehicle,
    load_config,
    resize_to_fit,
    setup_logging,
)

logger = logging.getLogger(__name__)


class LicensePlateCapturePipeline:
    """Standalone pipeline for vehicle license plate detection, cropping, and snap capturing."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg

        plate_cfg = cfg.get("plate", {})
        plate_model = str(plate_cfg.get("model_path", "models/plate_detector.pt"))
        plate_det_conf = float(plate_cfg.get("detection_conf", 0.20))
        plate_read_interval = int(plate_cfg.get("read_interval_frames", 3))
        plate_min_conf = float(plate_cfg.get("min_confidence", 0.18))
        use_gpu = bool(plate_cfg.get("use_gpu", False))

        easyocr_models = cfg.get("paths", {}).get("easyocr_models_dir", "models/EasyOCR/models")
        easyocr_user = cfg.get("paths", {}).get("easyocr_user_network_dir", "models/EasyOCR/user_network")

        self.plate_detector = LicensePlateDetector(
            model_path=plate_model,
            confidence_threshold=plate_det_conf,
        )
        self.plate_ocr = BanglaPlateOCR(
            custom_model_dir=easyocr_models,
            user_network_dir=easyocr_user,
            use_gpu=use_gpu,
        )
        self.plate_manager = TrackPlateManager(
            plate_detector=self.plate_detector,
            plate_ocr=self.plate_ocr,
            ocr_interval_frames=plate_read_interval,
            min_ocr_conf=plate_min_conf,
        )
        self.fps_counter = FPSCounter()

        paths_cfg = cfg.get("paths", {})
        self._output_dir = Path(paths_cfg.get("output_dir", "outputs"))
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._crops_dir = self._output_dir / "plate_crops"
        self._crops_dir.mkdir(parents=True, exist_ok=True)
        self._snapshots_dir = self._output_dir / "snapshots"
        self._snapshots_dir.mkdir(parents=True, exist_ok=True)

        self._csv_path = Path(paths_cfg.get("plate_csv", "outputs/plate_captures.csv"))
        self._db_path = Path(paths_cfg.get("db_log", "outputs/detections.db"))
        self._csv_file = None
        self._csv_writer = None
        self._db_conn = None

    def _open_outputs(self) -> None:
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
                "plate_bn",
                "plate_en",
                "plate_conf",
                "plate_crop_path",
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

    def _close_outputs(self) -> None:
        if self._csv_file:
            self._csv_file.close()
        if self._db_conn:
            self._db_conn.close()

    def run(self, max_frames: Optional[int] = None) -> None:
        display_cfg = self.cfg.get("display", {})
        show_window = bool(display_cfg.get("show_window", True))
        window_name = "License Plate Detection & Capture"
        line_thickness = int(display_cfg.get("line_thickness", 2))
        normal_color = (0, 255, 0)

        with FrameIngester(self.cfg) as ingester:
            detector, tracker = build_detector_and_tracker(self.cfg)
            self._open_outputs()

            if show_window:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 1280, 720)

            total_captured = 0

            for frame_idx, timestamp_sec, frame in ingester:
                if max_frames is not None and frame_idx >= max_frames:
                    break

                detections = detector.detect(frame)
                tracked, lost_ids = tracker.update(detections)
                annotated = frame.copy()

                for vehicle in tracked:
                    tid = vehicle.track_id
                    bbox = vehicle.bbox

                    det_res = self.plate_manager.update_track(
                        track_id=tid,
                        frame=frame,
                        vehicle_bbox=bbox,
                        frame_idx=frame_idx,
                        timestamp_sec=timestamp_sec,
                    )

                    rec = self.plate_manager.get_record(tid)
                    p_bn = rec.best_plate_text if rec else ""
                    p_en = rec.best_plate_text_en if rec else ""

                    # If plate was cropped in this frame, draw and save
                    if det_res and det_res.cropped_plate is not None:
                        # Draw plate bbox inside vehicle
                        px1, py1, px2, py2 = det_res.plate_bbox
                        cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 255, 255), 2)

                    tag = f"[{p_bn}]" if p_bn else (f"[{p_en}]" if p_en else "SCANNING")
                    draw_tracked_vehicle(annotated, tid, bbox, normal_color, line_thickness, tag)

                # Persist lost vehicles with best plate crop and snapshot
                for lost_id in lost_ids:
                    rec = self.plate_manager.get_record(lost_id)
                    p_crop = self.plate_manager.get_clearest_plate_crop(lost_id)

                    if rec and rec.frames_tracked >= 5:
                        now_dt = datetime.datetime.now()
                        plate_crop_path = ""
                        snap_path = ""

                        if p_crop is not None and p_crop.size > 0:
                            p_crop_name = f"track_{lost_id}_plate.png"
                            p_dest = self._crops_dir / p_crop_name
                            enhanced = PlatePreprocessor.enhance_for_display(p_crop)
                            cv2.imwrite(str(p_dest), enhanced)
                            plate_crop_path = str(p_dest)

                        if rec.best_vehicle_snapshot is not None:
                            s_name = f"track_{lost_id}_snapshot.png"
                            s_dest = self._snapshots_dir / s_name
                            cv2.imwrite(str(s_dest), rec.best_vehicle_snapshot)
                            snap_path = str(s_dest)

                        if self._csv_writer and self._csv_file:
                            self._csv_writer.writerow(
                                {
                                    "pc_date": now_dt.strftime("%Y-%m-%d"),
                                    "pc_time": now_dt.strftime("%H:%M:%S"),
                                    "timestamp_sec": f"{rec.timestamp_sec:.2f}",
                                    "frame_idx": rec.frame_idx,
                                    "track_id": lost_id,
                                    "plate_bn": rec.best_plate_text,
                                    "plate_en": rec.best_plate_text_en,
                                    "plate_conf": f"{rec.best_ocr_conf:.2f}",
                                    "plate_crop_path": plate_crop_path,
                                    "snapshot_path": snap_path,
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
                                        rec.timestamp_sec,
                                        rec.frame_idx,
                                        lost_id,
                                        rec.best_plate_text,
                                        rec.best_plate_text_en,
                                        rec.best_ocr_conf,
                                        0.0,
                                        0.0,
                                        0,
                                        "",
                                        "NORMAL",
                                        plate_crop_path,
                                        snap_path,
                                    ),
                                )
                                self._db_conn.commit()
                            except Exception as e:
                                logger.warning("Could not log plate detection to DB: %s", e)

                        total_captured += 1

                    self.plate_manager.remove_track(lost_id)

                fps_val = self.fps_counter.tick()
                draw_hud(
                    annotated,
                    fps_val,
                    frame_idx,
                    ingester.total_frames,
                    total_captured,
                    extra={"Plates Harvested": str(total_captured)},
                )

                if show_window:
                    display_frame = resize_to_fit(annotated, 1280, 720)
                    cv2.imshow(window_name, display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break

            if self.plate_manager is not None:
                self.plate_manager.shutdown(wait_seconds=1.5)

            if show_window:
                cv2.destroyAllWindows()
            self._close_outputs()

            logger.info("Plate capture run complete. Total plates logged: %d", total_captured)


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone License Plate Capture Pipeline")
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
    pipeline = LicensePlateCapturePipeline(cfg)
    pipeline.run(max_frames=args.max_frames)


if __name__ == "__main__":
    main()
