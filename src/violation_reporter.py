from __future__ import annotations

import csv
import datetime
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import winsound
except ImportError:
    winsound = None

from src.plate_detector import LicensePlateDetector, BanglaPlateOCR, PlatePreprocessor

logger = logging.getLogger(__name__)


class ViolationReporter:
    """Saves violation events to CSV, SQLite, snapshots and optionally video."""

    CSV_HEADERS = [
        "pc_date",
        "pc_time",
        "timestamp_sec",
        "frame_idx",
        "track_id",
        "reason",
        "direction_score",
        "progress_delta",
        "speed_kmh",
        "plate_text",
        "x1",
        "y1",
        "x2",
        "y2",
        "snapshot_path",
    ]

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        paths_cfg = cfg["paths"]
        out_cfg = cfg["output"]

        self._save_video = bool(out_cfg.get("save_annotated_video", True))
        self._output_dir = Path(paths_cfg["output_dir"])
        self._snapshot_dir = self._output_dir / "wrong_lane" / "snapshots"
        self._csv_path = Path(paths_cfg["csv_log"])
        self._db_path = Path(paths_cfg.get("db_log", "outputs/violations.db"))
        self._video_path = Path(paths_cfg["annotated_video"])

        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._video_path.parent.mkdir(parents=True, exist_ok=True)

        self._csv_file = None
        self._csv_writer: Optional[csv.DictWriter] = None
        self._video_writer: Optional[cv2.VideoWriter] = None
        self._db_conn: Optional[sqlite3.Connection] = None
        self._saved_tracks: set[int] = set()

        # License Plate Detector & Bangla OCR
        self._plate_detector: Optional[LicensePlateDetector] = None
        self._plate_ocr: Optional[BanglaPlateOCR] = None
        self._init_alpr()

    def _init_alpr(self) -> None:
        plate_model_path = self.cfg.get("plate", {}).get("model_path", "models/plate_detector.pt")
        if Path(plate_model_path).exists():
            try:
                logger.info("[ViolationReporter] Initializing LicensePlateDetector (%s)...", plate_model_path)
                self._plate_detector = LicensePlateDetector(model_path=plate_model_path)

                easyocr_models = self.cfg.get("paths", {}).get("easyocr_models_dir", "models/EasyOCR/models")
                easyocr_user = self.cfg.get("paths", {}).get("easyocr_user_network_dir", "models/EasyOCR/user_network")
                self._plate_ocr = BanglaPlateOCR(
                    custom_model_dir=easyocr_models,
                    user_network_dir=easyocr_user,
                    use_gpu=bool(self.cfg.get("plate", {}).get("use_gpu", False)),
                )
                logger.info("[ViolationReporter] ALPR system ready.")
            except Exception as e:
                logger.warning("[ViolationReporter] Could not load plate detector: %s", e)

    def open(self, width: int, height: int, fps: float) -> None:
        write_header = not self._csv_path.exists() or self._csv_path.stat().st_size == 0
        self._csv_file = open(self._csv_path, "a", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self.CSV_HEADERS)
        if write_header:
            self._csv_writer.writeheader()
            self._csv_file.flush()

        self._db_conn = sqlite3.connect(str(self._db_path))
        self._create_tables()

        if self._save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._video_writer = cv2.VideoWriter(str(self._video_path), fourcc, fps, (width, height))
            if not self._video_writer.isOpened():
                logger.warning("[ViolationReporter] Could not open annotated video writer: %s", self._video_path)
                self._video_writer = None

    def _create_tables(self) -> None:
        if self._db_conn:
            cursor = self._db_conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_sec REAL,
                    frame_idx INTEGER,
                    track_id INTEGER,
                    reason TEXT,
                    direction_score REAL,
                    progress_delta REAL,
                    speed_kmh REAL,
                    plate_text TEXT,
                    snapshot_path TEXT
                )
            """
            )
            self._db_conn.commit()

    def write_frame(self, frame) -> None:
        if self._video_writer is not None:
            self._video_writer.write(frame)

    def _crop_vehicle(self, frame, bbox: tuple[int, int, int, int], padding: int = 14):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)
        x2 = min(w, int(x2) + padding)
        y2 = min(h, int(y2) + padding)

        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2].copy()

    def report_violation(self, event, frame, timestamp_sec: float) -> str | None:
        if event.track_id in self._saved_tracks:
            return None

        snapshot_image = self._crop_vehicle(frame, event.bbox, padding=16)
        if snapshot_image is None:
            snapshot_image = frame.copy()

        # Detect license plate in the vehicle snapshot
        plate_text = ""
        best_plate_crop = None
        best_plate_conf = 0.0

        if self._plate_detector is not None and self._plate_ocr is not None:
            plates = self._plate_detector.detect_in_vehicle_crop(snapshot_image, (0, 0, snapshot_image.shape[1], snapshot_image.shape[0]))
            for p in plates:
                if p.cropped_plate is not None and p.cropped_plate.size > 0:
                    text, conf = self._plate_ocr.recognize(p.cropped_plate)
                    if conf > best_plate_conf:
                        best_plate_conf = conf
                        plate_text = text
                        best_plate_crop = p.cropped_plate

        # If a plate crop was detected, inset it on the top-right of the vehicle snapshot
        if best_plate_crop is not None and best_plate_crop.size > 0:
            sh, sw = snapshot_image.shape[:2]
            target_pw = min(sw // 2, 220)
            if target_pw > 40:
                scale = target_pw / float(best_plate_crop.shape[1])
                target_ph = max(20, int(best_plate_crop.shape[0] * scale))
                zoomed_p = cv2.resize(best_plate_crop, (target_pw, target_ph), interpolation=cv2.INTER_CUBIC)
                cv2.rectangle(zoomed_p, (0, 0), (target_pw - 1, target_ph - 1), (0, 255, 255), 2)

                px = max(0, sw - target_pw - 6)
                py = 6
                if py + target_ph < sh and px + target_pw < sw:
                    snapshot_image[py : py + target_ph, px : px + target_pw] = zoomed_p

        # Text banner
        reason_text = event.reason.replace("_", "-").upper()
        speed_kmh = getattr(event, "speed_kmh", 0.0)

        info_lines = [
            f"ID: {event.track_id}",
            f"VIOLATION: {reason_text}",
        ]
        if plate_text:
            info_lines.append(f"PLATE: {plate_text}")
        if speed_kmh > 0:
            info_lines.append(f"SPEED: {speed_kmh:.1f} km/h")

        y_offset = 24
        for line in info_lines:
            color = (0, 0, 255) if "VIOLATION" in line else (0, 255, 0)
            (tw, th), bl = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(snapshot_image, (4, y_offset - th - 3), (8 + tw, y_offset + bl), (0, 0, 0), -1)
            cv2.putText(
                snapshot_image,
                line,
                (6, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
            y_offset += 24

        snapshot_name = f"track_{event.track_id}_frame_{event.frame_idx}.png"
        snapshot_path = self._snapshot_dir / snapshot_name
        cv2.imwrite(str(snapshot_path), snapshot_image)

        # Beep alert sound
        if winsound is not None:
            def _play_beep():
                try:
                    winsound.Beep(1200, 250)
                except Exception:
                    pass
            threading.Thread(target=_play_beep, daemon=True).start()

        now_dt = datetime.datetime.now()
        pc_date = now_dt.strftime("%Y-%m-%d")
        pc_time = now_dt.strftime("%H:%M:%S")

        assert self._csv_writer is not None
        assert self._csv_file is not None

        x1, y1, x2, y2 = event.bbox
        self._csv_writer.writerow(
            {
                "pc_date": pc_date,
                "pc_time": pc_time,
                "timestamp_sec": f"{timestamp_sec:.3f}",
                "frame_idx": event.frame_idx,
                "track_id": event.track_id,
                "reason": event.reason,
                "direction_score": f"{event.direction_score:.4f}",
                "progress_delta": f"{event.progress_delta:.2f}",
                "speed_kmh": f"{speed_kmh:.2f}",
                "plate_text": plate_text,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "snapshot_path": str(snapshot_path),
            }
        )
        self._csv_file.flush()

        if self._db_conn:
            cursor = self._db_conn.cursor()
            # Ensure pc_date / pc_time columns exist
            cur = self._db_conn.execute("PRAGMA table_info(violations)")
            v_cols = {col[1] for col in cur.fetchall()}
            if "pc_date" not in v_cols:
                try:
                    self._db_conn.execute("ALTER TABLE violations ADD COLUMN pc_date TEXT")
                    self._db_conn.execute("ALTER TABLE violations ADD COLUMN pc_time TEXT")
                except Exception:
                    pass
            cursor.execute(
                """
                INSERT INTO violations (pc_date, pc_time, timestamp_sec, frame_idx, track_id, reason, direction_score, progress_delta, speed_kmh, plate_text, snapshot_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    pc_date,
                    pc_time,
                    timestamp_sec,
                    event.frame_idx,
                    event.track_id,
                    event.reason,
                    event.direction_score,
                    event.progress_delta,
                    speed_kmh,
                    plate_text,
                    str(snapshot_path),
                ),
            )
            self._db_conn.commit()

        self._saved_tracks.add(event.track_id)

        logger.info(
            "[ViolationReporter] violation track=%d reason=%s plate='%s' speed=%.1f saved=%s",
            event.track_id,
            event.reason,
            plate_text,
            speed_kmh,
            snapshot_path,
        )
        return str(snapshot_path)

    def close(self) -> None:
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None

        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

        if self._db_conn is not None:
            self._db_conn.close()
            self._db_conn = None

    def __enter__(self) -> "ViolationReporter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
