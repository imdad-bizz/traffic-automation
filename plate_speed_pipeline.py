"""
Plate Detection & Speed Estimation Pipeline (Bangladeshi ALPR)
==============================================================
Detect vehicles (YOLOv8 + ByteTrack)
→ detect license plates (YOLO plate detector)
→ recognize Bangladeshi plates (specialized Bengali EasyOCR + BRTA parser)
→ estimate calibrated speed from perspective homography IPM
→ enriched sustained speeding validation (filters horizon jitter & momentary spikes)
→ non-blocking audio beep alert & on-screen notification banner on violation
→ crystal-clear enhanced license plate snapshots & vehicle evidence
→ PC clock date & time recording
→ export to SQLite, CSV, and executive Excel with embedded thumbnails & optional GPT-4o Vision.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import logging
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import cv2
import numpy as np

try:
    import winsound
except ImportError:
    winsound = None

from src.lane_zone import LaneZone
from src.motion_tracker import MotionTracker
from src.plate_detector import (
    BanglaPlateOCR,
    LicensePlateDetector,
    PlateDetectionResult,
    PlatePreprocessor,
    TrackPlateManager,
)
from src.speed_estimator import PerspectiveSpeedEstimator
from src.tracker import TrackedVehicle, build_detector_and_tracker
from src.utils import (
    FPSCounter,
    FrameIngester,
    draw_bilingual_text,
    draw_tracked_vehicle,
    resize_for_display,
    resize_to_fit,
    to_english_digits,
    trigger_beep,
)
from src.violation_detector import ViolationDetector, ViolationEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-track state for persistence and validation
# ---------------------------------------------------------------------------
@dataclass
class TrackState:
    last_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    last_speed_kmh: float = 0.0
    effective_speed_kmh: float = 0.0
    peak_speed_kmh: float = 0.0
    is_confirmed_speeding: bool = False
    is_confirmed_lane_violation: bool = False
    lane_violation_reason: str = ""
    lane_status: str = "OUT"
    saved: bool = False
    frames_seen: int = 0
    max_vehicle_area: int = 0
    best_frame_idx: int = 0
    best_timestamp_sec: float = 0.0
    saved_snap_path: str = ""
    saved_crop_path: str = ""


# ---------------------------------------------------------------------------
# Pipeline Class
# ---------------------------------------------------------------------------
class PlateSpeedPipeline:
    def __init__(self, cfg: dict, enable_lane: bool = True) -> None:
        self.cfg = cfg
        self.enable_lane = enable_lane

        # Lane Zone & Motion Tracker (Wrong-Way & Lane Violation)
        self.lane_zone: Optional[LaneZone] = None
        motion_cfg = cfg.get("motion", {})
        self.motion_tracker = MotionTracker(
            history_size=int(motion_cfg.get("history_size", 20)),
            stale_after_frames=int(motion_cfg.get("stale_after_frames", 90)),
        )
        self.violation_detector = ViolationDetector(cfg)

        # Speed estimation config
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

        # Audio and visual alert settings
        alerts_cfg = cfg.get("alerts", {})
        self._alerts_enabled = bool(alerts_cfg.get("beep_enabled", True))
        self._beep_freq = int(alerts_cfg.get("beep_frequency_hz", 1200))
        self._beep_dur = int(alerts_cfg.get("beep_duration_ms", 250))
        self._banner_duration_sec = float(alerts_cfg.get("banner_duration_sec", 2.5))

        self._beeped_tracks: Set[int] = set()
        self._active_alert_text: str = ""
        self._active_alert_expiry: float = 0.0

        # Persistent sets for exact violation tracking
        self._confirmed_speed_violators: Set[int] = set()
        self._confirmed_lane_violators: Set[int] = set()
        self._violating_tracks: Set[int] = set()

        # Plate detection & OCR config
        plate_cfg = cfg.get("plate", {})
        self._plate_model_path = str(plate_cfg.get("model_path", "models/plate_detector.pt"))
        self._plate_det_conf = float(plate_cfg.get("detection_conf", 0.20))
        self._plate_read_interval = int(plate_cfg.get("read_interval_frames", 3))
        self._plate_min_conf = float(plate_cfg.get("min_confidence", 0.18))
        self._use_gpu = bool(plate_cfg.get("use_gpu", False))

        self.fps_counter = FPSCounter(
            window_size=int(cfg.get("display", {}).get("fps_window", 30))
        )
        self.vehicle_detector = None
        self.vehicle_tracker = None

        self.plate_detector: Optional[LicensePlateDetector] = None
        self.plate_ocr: Optional[BanglaPlateOCR] = None
        self.plate_manager: Optional[TrackPlateManager] = None

        self._tracks: Dict[int, TrackState] = {}
        self._latest_detected_plate: Optional[np.ndarray] = None
        self._latest_detected_text_bn: str = ""
        self._latest_detected_text_en: str = ""
        self._latest_detected_conf: float = 0.0

        # Persistence paths
        paths_cfg = cfg["paths"]
        self._output_dir = Path(paths_cfg.get("output_dir", "outputs"))
        self._snapshot_dir = self._output_dir / "plate_snapshots"
        self._crops_dir = self._output_dir / "plate_crops"
        self._csv_path = Path(paths_cfg.get("csv_log", "outputs/detections.csv"))
        self._db_path = Path(paths_cfg.get("db_log", "outputs/detections.db"))
        self._video_path = Path(paths_cfg.get("annotated_video", "outputs/annotated_plate_speed.mp4"))

        for d in (self._output_dir, self._snapshot_dir, self._crops_dir, self._csv_path.parent, self._db_path.parent):
            d.mkdir(parents=True, exist_ok=True)

        self._csv_file = None
        self._csv_writer: Optional[csv.DictWriter] = None
        self._video_writer: Optional[cv2.VideoWriter] = None
        self._db_conn: Optional[sqlite3.Connection] = None

    def _init_alpr(self) -> None:
        logger.info("[Pipeline] Initializing License Plate Detector (%s)...", self._plate_model_path)
        self.plate_detector = LicensePlateDetector(
            model_path=self._plate_model_path,
            confidence_threshold=self._plate_det_conf,
        )

        logger.info("[Pipeline] Initializing BanglaPlateOCR with custom bn_license_tps network...")
        easyocr_models = self.cfg.get("paths", {}).get("easyocr_models_dir", "models/EasyOCR/models")
        easyocr_user = self.cfg.get("paths", {}).get("easyocr_user_network_dir", "models/EasyOCR/user_network")

        self.plate_ocr = BanglaPlateOCR(
            custom_model_dir=easyocr_models,
            user_network_dir=easyocr_user,
            use_gpu=self._use_gpu,
        )

        self.plate_manager = TrackPlateManager(
            plate_detector=self.plate_detector,
            plate_ocr=self.plate_ocr,
            ocr_interval_frames=self._plate_read_interval,
            min_ocr_conf=self._plate_min_conf,
        )

    def _trigger_alert_beep(self, track_id: int, speed: float = 0.0, violation_type: str = "SPEED") -> None:
        """Plays a non-blocking alert beep sound and displays an on-screen alert banner."""
        alert_key = f"{track_id}_{violation_type}"
        if alert_key in self._beeped_tracks:
            return
        self._beeped_tracks.add(alert_key)

        # Trigger on-screen banner
        now_t = time.time()
        if "WRONG" in violation_type or "LANE" in violation_type or "ENTRY" in violation_type or "opposite" in violation_type:
            lbl = "WRONG-WAY (Opposite Direction)" if "opposite" in violation_type or "WRONG" in violation_type else "FORBIDDEN ENTRY SIDE"
            self._active_alert_text = f"🚨 ALERT: LANE VIOLATION! Track #{track_id} ({lbl})"
            logger.warning("🚨 [ALERT] LANE VIOLATION! Track #%d flagged for %s", track_id, lbl)
        else:
            self._active_alert_text = (
                f"🚨 ALERT: SPEED VIOLATION! Track #{track_id} clocked at {speed:.1f} km/h (Limit: {self._speed_limit_kmh:.0f})"
            )
            logger.warning(
                "🚨 [ALERT] SPEED VIOLATION! Track #%d clocked at %.1f km/h (Limit: %.0f km/h + %.0f tolerance)",
                track_id,
                speed,
                self._speed_limit_kmh,
                self._speed_tolerance_kmh,
            )
        self._active_alert_expiry = now_t + self._banner_duration_sec

        # Non-blocking debounced audio beep
        if self._alerts_enabled:
            trigger_beep(self._beep_freq, self._beep_dur)

    def _open_outputs(self, width: int, height: int, fps: float) -> None:
        # CSV
        write_header = not self._csv_path.exists() or self._csv_path.stat().st_size == 0
        self._csv_file = open(self._csv_path, "a", newline="", encoding="utf-8")
        headers = [
            "pc_date",
            "pc_time",
            "timestamp_sec",
            "frame_idx",
            "track_id",
            "plate_bn",
            "plate_en",
            "plate_conf",
            "speed_kmh",
            "peak_speed_kmh",
            "speeding",
            "lane_violation",
            "violation_type",
            "x1",
            "y1",
            "x2",
            "y2",
            "plate_crop_path",
            "snapshot_path",
        ]
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=headers)
        if write_header:
            self._csv_writer.writeheader()
            self._csv_file.flush()

        # SQLite
        self._db_conn = sqlite3.connect(str(self._db_path))
        self._db_conn.execute("""
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
                plate_crop_path TEXT,
                snapshot_path  TEXT
            )
        """)
        # Safe schema migration
        cur = self._db_conn.execute("PRAGMA table_info(detections)")
        existing_cols = {row[1] for row in cur.fetchall()}
        for col, col_type in [
            ("pc_date", "TEXT"),
            ("pc_time", "TEXT"),
            ("plate_crop_path", "TEXT"),
            ("peak_speed_kmh", "REAL DEFAULT 0.0"),
            ("lane_violation", "TEXT DEFAULT ''"),
            ("violation_type", "TEXT DEFAULT 'None'"),
        ]:
            if col not in existing_cols:
                try:
                    self._db_conn.execute(f"ALTER TABLE detections ADD COLUMN {col} {col_type}")
                except Exception:
                    pass
        self._db_conn.commit()

        # Video writer
        save_video = bool(self.cfg.get("output", {}).get("save_annotated_video", True))
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._video_writer = cv2.VideoWriter(
                str(self._video_path), fourcc, fps, (width, height)
            )
            if not self._video_writer.isOpened():
                logger.warning("[Pipeline] Cannot open video writer: %s", self._video_path)
                self._video_writer = None

    def _close_outputs(self) -> None:
        if self._video_writer:
            self._video_writer.release()
        if self._csv_file:
            self._csv_file.close()
        if self._db_conn:
            self._db_conn.close()

    def _save_record(
        self,
        frame: np.ndarray,
        track_id: int,
        bbox: Tuple[int, int, int, int],
        frame_idx: int,
        timestamp_sec: float,
        plate_bn: str,
        plate_en: str,
        plate_conf: float,
        speed_kmh: float,
        peak_speed_kmh: float,
        is_speeding: bool,
        plate_crop: Optional[np.ndarray] = None,
        lane_violation: str = "",
    ) -> str:
        # Capture current PC system clock time
        now_dt = datetime.datetime.now()
        pc_date = now_dt.strftime("%Y-%m-%d")
        pc_time = now_dt.strftime("%H:%M:%S")

        speeding_val = 1 if is_speeding else 0

        # Snapshot crop
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        pad = 20
        crop = frame[max(0, y1 - pad) : min(h, y2 + pad), max(0, x1 - pad) : min(w, x2 + pad)].copy()

        # Save isolated enhanced plate crop for maximum clarity
        plate_crop_path = ""
        if plate_crop is not None and plate_crop.size > 0:
            p_name = f"track_{track_id}_plate.png"
            p_dest = self._crops_dir / p_name
            # Generate enhanced version with CLAHE + super-resolution
            enhanced_plate = PlatePreprocessor.enhance_for_display(plate_crop)
            cv2.imwrite(str(p_dest), enhanced_plate)
            plate_crop_path = str(p_dest)

        if crop.size > 0:
            composite_h, composite_w = crop.shape[:2]

            # Inset zoomed enhanced license plate in top-right
            if plate_crop is not None and plate_crop.size > 0:
                target_pw = min(composite_w // 2, 280)
                if target_pw > 50:
                    scale = target_pw / float(plate_crop.shape[1])
                    target_ph = max(26, int(plate_crop.shape[0] * scale))
                    zoomed_plate = cv2.resize(plate_crop, (target_pw, target_ph), interpolation=cv2.INTER_CUBIC)
                    cv2.rectangle(zoomed_plate, (0, 0), (target_pw - 1, target_ph - 1), (0, 255, 255), 2)
                    px_start = max(0, composite_w - target_pw - 8)
                    py_start = 8
                    if py_start + target_ph < composite_h and px_start + target_pw < composite_w:
                        crop[py_start : py_start + target_ph, px_start : px_start + target_pw] = zoomed_plate

            # Info badge lines
            info_lines = [
                f"TRACK ID: {track_id}",
                f"TIME: {pc_date} {pc_time}",
            ]
            if plate_bn:
                info_lines.append(f"PLATE: {plate_bn}")
            if plate_en:
                info_lines.append(f"TRANSLIT: {plate_en} (conf {plate_conf:.2f})")
            if speed_kmh > 0:
                info_lines.append(f"SPEED: {speed_kmh:.1f} km/h (Limit: {self._speed_limit_kmh:.0f})")
            if lane_violation:
                lbl = "WRONG-WAY (Opposite Direction)" if lane_violation == "opposite_direction" or "wrong" in lane_violation.lower() else f"ENTRY-VIOL ({lane_violation})"
                info_lines.append(f"*** LANE VIOLATION: {lbl} ***")
            if is_speeding:
                info_lines.append("*** SPEED LIMIT VIOLATION ***")

            y_offset = 14
            for line in info_lines:
                color = (0, 0, 255) if "VIOLATION" in line else (0, 255, 0)
                draw_bilingual_text(crop, line, (10, y_offset), font_size=18, color=color, bg_color=(0, 0, 0))
                y_offset += 24

        snap_name = f"track_{track_id}_frame_{frame_idx}.png"
        snap_path = self._snapshot_dir / snap_name
        cv2.imwrite(str(snap_path), crop)

        if is_speeding and lane_violation:
            v_type_str = "Speed + Lane Violation"
        elif is_speeding:
            v_type_str = "Speed Violation"
        elif lane_violation:
            v_type_str = "Wrong-Way / Lane Violation"
        else:
            v_type_str = "Compliant"

        # CSV logging
        if self._csv_writer and self._csv_file:
            self._csv_writer.writerow(
                {
                    "pc_date": pc_date,
                    "pc_time": pc_time,
                    "timestamp_sec": f"{timestamp_sec:.3f}",
                    "frame_idx": frame_idx,
                    "track_id": track_id,
                    "plate_bn": plate_bn,
                    "plate_en": plate_en,
                    "plate_conf": f"{plate_conf:.3f}",
                    "speed_kmh": f"{speed_kmh:.1f}",
                    "peak_speed_kmh": f"{peak_speed_kmh:.1f}",
                    "speeding": speeding_val,
                    "lane_violation": lane_violation,
                    "violation_type": v_type_str,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "plate_crop_path": plate_crop_path,
                    "snapshot_path": str(snap_path),
                }
            )
            self._csv_file.flush()

        # SQLite logging
        if self._db_conn:
            self._db_conn.execute(
                """
                INSERT INTO detections (pc_date, pc_time, timestamp_sec, frame_idx, track_id, plate_bn, plate_en, plate_conf, speed_kmh, peak_speed_kmh, speeding, lane_violation, violation_type, plate_crop_path, snapshot_path)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
                (
                    pc_date,
                    pc_time,
                    timestamp_sec,
                    frame_idx,
                    track_id,
                    plate_bn,
                    plate_en,
                    plate_conf,
                    speed_kmh,
                    peak_speed_kmh,
                    speeding_val,
                    lane_violation,
                    v_type_str,
                    plate_crop_path,
                    str(snap_path),
                ),
            )
            self._db_conn.commit()

        tag = f"🚨 {v_type_str.upper()}" if (is_speeding or lane_violation) else "✅ COMPLIANT"
        logger.info(
            "[Pipeline] %s track=%d plate='%s' (%s) speed=%.1f km/h [PC Time: %s %s] → %s",
            tag,
            track_id,
            plate_bn,
            plate_en,
            speed_kmh,
            pc_date,
            pc_time,
            snap_path.name,
        )
        return str(snap_path)

    def _draw_hud(
        self,
        frame: np.ndarray,
        fps: float,
        frame_idx: int,
        total_frames: int,
        active_tracks: int,
        total_saved: int,
        total_plates: int,
        total_violations: int,
        speed_violations: int = 0,
        lane_violations: int = 0,
    ) -> None:
        h, w = frame.shape[:2]

        # Top banner
        header_h = 46
        cv2.rectangle(frame, (0, 0), (w, header_h), (16, 24, 32), -1)

        pct = f"({(frame_idx / max(1, total_frames)) * 100:.1f}%)" if total_frames > 0 else ""
        if speed_violations > 0 or lane_violations > 0:
            viol_detail = f"{total_violations} (Lane:{lane_violations}, Speed:{speed_violations})"
        else:
            viol_detail = f"{total_violations}"

        hud = (
            f"FPS: {fps:.1f} | Frame: {frame_idx}/{total_frames} {pct} | "
            f"Active: {active_tracks} | Plates: {total_plates} | Violations: {viol_detail} | Logged: {total_saved}"
        )
        hud_color = (0, 80, 255) if total_violations > 0 else (0, 255, 0)
        cv2.putText(
            frame,
            hud,
            (14, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            hud_color,
            2,
            cv2.LINE_AA,
        )

        # On-screen visual notification alert banner
        if time.time() < self._active_alert_expiry and self._active_alert_text:
            alert_bar_h = 38
            cv2.rectangle(frame, (0, header_h), (w, header_h + alert_bar_h), (0, 0, 200), -1)
            cv2.rectangle(frame, (0, header_h), (w, header_h + alert_bar_h), (0, 255, 255), 2)
            cv2.putText(
                frame,
                self._active_alert_text,
                (20, header_h + 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Picture-in-Picture for latest recognized plate
        if self._latest_detected_plate is not None and self._latest_detected_plate.size > 0:
            pip_w, pip_h = 260, 95
            pip_x = w - pip_w - 14
            pip_y = header_h + 16

            cv2.rectangle(
                frame,
                (pip_x - 4, pip_y - 4),
                (pip_x + pip_w + 4, pip_y + pip_h + 38),
                (12, 12, 12),
                -1,
            )
            cv2.rectangle(
                frame,
                (pip_x - 4, pip_y - 4),
                (pip_x + pip_w + 4, pip_y + pip_h + 38),
                (0, 255, 255),
                2,
            )

            resized_p = cv2.resize(
                self._latest_detected_plate, (pip_w, pip_h), interpolation=cv2.INTER_CUBIC
            )
            frame[pip_y : pip_y + pip_h, pip_x : pip_x + pip_w] = resized_p

            title = f"{self._latest_detected_text_bn or 'PLATE DETECTED'}"
            if self._latest_detected_text_en:
                title += f" ({self._latest_detected_text_en})"

            draw_bilingual_text(
                frame,
                title,
                (pip_x, pip_y + pip_h + 6),
                font_size=17,
                color=(0, 255, 255),
                bg_color=(12, 12, 12),
            )

    def run(self, max_frames: Optional[int] = None) -> None:
        self._init_alpr()

        display_cfg = self.cfg.get("display", {})
        show_window = bool(display_cfg.get("show_window", True))
        window_name = str(display_cfg.get("window_name", "Bangladeshi ALPR & Speed Monitoring"))
        fixed_window_size = bool(display_cfg.get("fixed_window_size", True))
        window_width = int(display_cfg.get("window_width", 1280))
        window_height = int(display_cfg.get("window_height", 720))
        display_scale = float(display_cfg.get("display_scale", 0.7))
        line_thickness = int(display_cfg.get("line_thickness", 2))

        normal_color = tuple(display_cfg.get("bbox_color_normal", [0, 255, 0]))
        out_of_lane_color = tuple(display_cfg.get("bbox_color_out_of_lane", [255, 180, 0]))
        violation_color = tuple(display_cfg.get("bbox_color_violation", [0, 0, 255]))
        lane_color = tuple(display_cfg.get("lane_polygon_color", [0, 255, 255]))
        draw_lane_points = bool(display_cfg.get("draw_lane_points", True))

        with FrameIngester(self.cfg) as ingester:
            fps = ingester.processing_fps
            total_source_frames = int(ingester._capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if self.enable_lane and "lane_zone" in self.cfg:
                self.lane_zone = LaneZone.from_config(
                    self.cfg, frame_size=(ingester.frame_width, ingester.frame_height)
                )
            self.speed_estimator = PerspectiveSpeedEstimator.from_config(
                self.cfg, frame_size=(ingester.frame_width, ingester.frame_height)
            )
            self.vehicle_detector, self.vehicle_tracker = build_detector_and_tracker(self.cfg)
            self._open_outputs(ingester.frame_width, ingester.frame_height, max(1.0, fps))

            if show_window:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                if fixed_window_size:
                    cv2.resizeWindow(window_name, window_width, window_height)

            total_saved = 0
            total_plates_read = 0
            total_violations = 0
            last_progress_log = 0.0

            logger.info(
                "[Pipeline] Processing video: total_source_frames=%d, processing_fps=%.1f",
                total_source_frames,
                fps,
            )

            for frame_idx, timestamp_sec, frame in ingester:
                if max_frames is not None and frame_idx >= max_frames:
                    logger.info("[Pipeline] Reached max_frames=%d — stopping.", max_frames)
                    break

                detections = self.vehicle_detector.detect(frame)
                tracked, lost_ids = self.vehicle_tracker.update(detections)
                annotated = frame.copy()

                # Draw 4-point Lane Zone polygon, guide points P1..P4 and directional arrow
                if self.lane_zone is not None:
                    self.lane_zone.draw(
                        annotated,
                        polygon_color=lane_color,
                        thickness=line_thickness,
                        draw_points=draw_lane_points,
                    )

                active_ids: Set[int] = set()
                for vehicle in tracked:
                    tid = vehicle.track_id
                    active_ids.add(tid)
                    bbox = vehicle.bbox
                    cx, cy = vehicle.centroid
                    vx1, vy1, vx2, vy2 = bbox
                    v_area = (vx2 - vx1) * (vy2 - vy1)

                    if tid not in self._tracks:
                        self._tracks[tid] = TrackState()
                    state = self._tracks[tid]
                    state.last_bbox = bbox
                    state.frames_seen += 1

                    if v_area > state.max_vehicle_area:
                        state.max_vehicle_area = v_area
                        state.best_frame_idx = frame_idx
                        state.best_timestamp_sec = timestamp_sec

                    # Enriched Speed Estimation via Perspective Homography
                    speed = self.speed_estimator.update_track(
                        track_id=tid,
                        centroid=(cx, cy),
                        timestamp_sec=timestamp_sec,
                        bbox=bbox,
                    )
                    state.last_speed_kmh = speed
                    eff_speed = self.speed_estimator.get_effective_speed(tid)
                    state.effective_speed_kmh = eff_speed
                    peak_speed = self.speed_estimator.get_peak_speed(tid)
                    state.peak_speed_kmh = peak_speed

                    # Check confirmed speeding violation
                    is_speeding = self.speed_estimator.is_confirmed_speeding(tid)
                    if is_speeding and not state.is_confirmed_speeding:
                        state.is_confirmed_speeding = True
                        self._confirmed_speed_violators.add(tid)
                        self._violating_tracks.add(tid)
                        self._trigger_alert_beep(tid, eff_speed, violation_type="SPEED")

                    # Lane Anchor, Motion Tracking & Direction Violation
                    lane_anchor = ((vx1 + vx2) // 2, vy2)
                    inside_lane = self.lane_zone.contains(lane_anchor) if self.lane_zone is not None else False

                    if self.lane_zone is not None:
                        self.motion_tracker.update(
                            track_id=tid,
                            centroid=lane_anchor,
                            frame_idx=frame_idx,
                            inside_lane=inside_lane,
                        )
                        if inside_lane:
                            lane_event = self.violation_detector.evaluate(
                                track_id=tid,
                                bbox=bbox,
                                centroid=lane_anchor,
                                frame_idx=frame_idx,
                                lane_zone=self.lane_zone,
                                motion_tracker=self.motion_tracker,
                                fps=fps,
                            )
                            if lane_event is not None and not state.is_confirmed_lane_violation:
                                state.is_confirmed_lane_violation = True
                                state.lane_violation_reason = lane_event.reason
                                self._confirmed_lane_violators.add(tid)
                                self._violating_tracks.add(tid)
                                if lane_event.reason == "opposite_direction":
                                    r_lbl = "WRONG-WAY"
                                elif lane_event.reason == "forbidden_entry_side":
                                    r_lbl = "ENTRY-VIOL"
                                elif lane_event.reason == "speeding":
                                    r_lbl = "SPEED"
                                else:
                                    r_lbl = "VIOLATION"
                                self._trigger_alert_beep(tid, eff_speed, violation_type=r_lbl)

                        existing_lane_event = self.violation_detector.get_event(tid)
                        if existing_lane_event is not None and not state.is_confirmed_lane_violation:
                            state.is_confirmed_lane_violation = True
                            state.lane_violation_reason = existing_lane_event.reason
                            self._confirmed_lane_violators.add(tid)
                            self._violating_tracks.add(tid)

                    # Plate Detection & Recognition
                    det_result = self.plate_manager.update_track(
                        track_id=tid,
                        frame=frame,
                        vehicle_bbox=bbox,
                        frame_idx=frame_idx,
                        timestamp_sec=timestamp_sec,
                        speed_kmh=eff_speed,
                    )

                    plate_record = self.plate_manager.get_record(tid)
                    plate_bn = plate_record.best_plate_text if plate_record else ""
                    plate_en = plate_record.best_plate_text_en if plate_record else ""

                    if det_result is not None and det_result.cropped_plate is not None:
                        self._latest_detected_plate = det_result.cropped_plate.copy()
                        self._latest_detected_text_bn = det_result.plate_text or plate_bn
                        self._latest_detected_text_en = det_result.plate_text_en or plate_en
                        self._latest_detected_conf = det_result.ocr_confidence
                        if det_result.plate_text:
                            total_plates_read += 1

                        # Draw box around detected plate on vehicle
                        px1, py1, px2, py2 = det_result.bbox
                        cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 255, 255), 2)
                        cv2.putText(
                            annotated,
                            "PLATE",
                            (px1, max(14, py1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (0, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )

                    # Determine lane tag and status
                    if state.is_confirmed_lane_violation:
                        lane_tag = "🚨WRONG-WAY" if state.lane_violation_reason == "opposite_direction" else "🚨ENTRY-VIOL"
                    elif inside_lane and self.lane_zone is not None:
                        progress = self.lane_zone.progress_ratio(lane_anchor)
                        lane_tag = f"IN:{progress:+.2f}"
                    else:
                        lane_tag = "OUT"
                    state.lane_status = lane_tag

                    # Determine color:
                    # Red if speeding OR lane violation; Green if inside lane; Orange if outside lane
                    is_speed_viol = state.is_confirmed_speeding or (eff_speed > (self._speed_limit_kmh + self._speed_tolerance_kmh))
                    is_any_viol = is_speed_viol or state.is_confirmed_lane_violation

                    if is_any_viol:
                        color = violation_color
                    elif inside_lane:
                        color = normal_color
                    else:
                        color = out_of_lane_color

                    parts = []
                    if plate_bn:
                        parts.append(f"[{plate_bn}]")
                    elif plate_en:
                        parts.append(f"[{plate_en}]")

                    if eff_speed > 0:
                        parts.append(f"{eff_speed:.0f}km/h")

                    if is_speed_viol:
                        parts.append("🚨SPEEDING")

                    parts.append(lane_tag)

                    subtitle = " | ".join(parts) if parts else ""
                    draw_tracked_vehicle(
                        annotated, tid, bbox, color, line_thickness, subtitle
                    )
                    # Draw bottom anchor circle on lane
                    cv2.circle(annotated, lane_anchor, 3, color, -1, lineType=cv2.LINE_AA)

                # Handle lost tracks (vehicle exited scene) -> persist definitive record
                for lost_id in lost_ids:
                    if self.lane_zone is not None:
                        self.motion_tracker.remove(lost_id)

                    state = self._tracks.get(lost_id)
                    plate_rec = self.plate_manager.get_record(lost_id)
                    eff_s = self.speed_estimator.get_effective_speed(lost_id)
                    peak_s = self.speed_estimator.get_peak_speed(lost_id)
                    is_viol = state.is_confirmed_speeding if state else False

                    if state and not state.saved and state.frames_seen >= self._min_track_frames:
                        p_bn = plate_rec.best_plate_text if plate_rec else ""
                        p_en = plate_rec.best_plate_text_en if plate_rec else ""
                        p_conf = plate_rec.best_ocr_conf if plate_rec else 0.0
                        p_crop = self.plate_manager.get_clearest_plate_crop(lost_id)

                        snap_frame = (
                            plate_rec.best_vehicle_snapshot
                            if (plate_rec and plate_rec.best_vehicle_snapshot is not None)
                            else frame
                        )
                        f_idx = plate_rec.frame_idx if plate_rec else frame_idx
                        t_sec = plate_rec.timestamp_sec if plate_rec else timestamp_sec

                        self._save_record(
                            frame=snap_frame,
                            track_id=lost_id,
                            bbox=state.last_bbox,
                            frame_idx=f_idx,
                            timestamp_sec=t_sec,
                            plate_bn=p_bn,
                            plate_en=p_en,
                            plate_conf=p_conf,
                            speed_kmh=eff_s,
                            peak_speed_kmh=peak_s,
                            is_speeding=is_viol,
                            plate_crop=p_crop,
                            lane_violation=state.lane_violation_reason if state.is_confirmed_lane_violation else "",
                        )
                        state.saved = True
                        total_saved += 1

                    self._tracks.pop(lost_id, None)
                    self.plate_manager.remove_track(lost_id)
                    self.speed_estimator.remove_track(lost_id)

                if self.lane_zone is not None:
                    self.motion_tracker.cleanup(active_ids=active_ids, current_frame_idx=frame_idx)

                # Persist active tracks that reach optimal close-up point (y2 >= 850 or leaving foreground)
                for tid in active_ids:
                    state = self._tracks.get(tid)
                    plate_rec = self.plate_manager.get_record(tid)
                    if not state or state.saved or state.frames_seen < (self._min_track_frames * 2):
                        continue

                    # If vehicle has passed foreground (closest point to camera)
                    y2 = state.last_bbox[3]
                    is_close_enough = y2 >= 850
                    eff_s = state.effective_speed_kmh
                    is_viol = state.is_confirmed_speeding or state.is_confirmed_lane_violation

                    if is_close_enough or (is_viol and state.frames_seen >= 15):
                        p_bn = plate_rec.best_plate_text if plate_rec else ""
                        p_en = plate_rec.best_plate_text_en if plate_rec else ""
                        p_conf = plate_rec.best_ocr_conf if plate_rec else 0.0
                        p_crop = self.plate_manager.get_clearest_plate_crop(tid)

                        snap_frame = (
                            plate_rec.best_vehicle_snapshot
                            if (plate_rec and plate_rec.best_vehicle_snapshot is not None)
                            else frame
                        )

                        self._save_record(
                            frame=snap_frame,
                            track_id=tid,
                            bbox=state.last_bbox,
                            frame_idx=frame_idx,
                            timestamp_sec=timestamp_sec,
                            plate_bn=p_bn,
                            plate_en=p_en,
                            plate_conf=p_conf,
                            speed_kmh=eff_s,
                            peak_speed_kmh=state.peak_speed_kmh,
                            is_speeding=state.is_confirmed_speeding,
                            plate_crop=p_crop,
                            lane_violation=state.lane_violation_reason if state.is_confirmed_lane_violation else "",
                        )
                        state.saved = True
                        total_saved += 1


                fps_value = self.fps_counter.tick()
                total_violations = len(self._violating_tracks)
                speed_violations = len(self._confirmed_speed_violators)
                lane_violations = len(self._confirmed_lane_violators)

                self._draw_hud(
                    annotated,
                    fps_value,
                    frame_idx,
                    total_source_frames,
                    len(tracked),
                    total_saved,
                    total_plates_read,
                    total_violations=total_violations,
                    speed_violations=speed_violations,
                    lane_violations=lane_violations,
                )

                if self._video_writer:
                    self._video_writer.write(annotated)

                # Terminal progress update
                now_t = time.time()
                if now_t - last_progress_log >= 3.0:
                    last_progress_log = now_t
                    pct_str = f"{(frame_idx / max(1, total_source_frames)) * 100:.1f}%" if total_source_frames > 0 else "N/A"
                    logger.info(
                        "[Progress] Frame %d/%d (%s) | FPS: %.1f | Active: %d | Violations: %d (Lane:%d, Speed:%d) | Logged: %d",
                        frame_idx,
                        total_source_frames,
                        pct_str,
                        fps_value,
                        len(tracked),
                        total_violations,
                        lane_violations,
                        speed_violations,
                        total_saved,
                    )

                if show_window:
                    if fixed_window_size:
                        preview = resize_to_fit(annotated, window_width, window_height)
                    else:
                        preview = resize_for_display(annotated, display_scale)
                    cv2.imshow(window_name, preview)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        logger.info("[Pipeline] 'q' pressed — stopping.")
                        break

            # Flush and shutdown background OCR worker so pending plate reads complete
            if self.plate_manager is not None:
                logger.info("[Pipeline] Finalizing background ALPR recognition tasks...")
                self.plate_manager.shutdown(wait_seconds=1.5)

            # Save any remaining active tracks at end of footage
            for tid, state in list(self._tracks.items()):
                if not state.saved and state.frames_seen >= self._min_track_frames:
                    plate_rec = self.plate_manager.get_record(tid)
                    p_bn = plate_rec.best_plate_text if plate_rec else ""
                    p_en = plate_rec.best_plate_text_en if plate_rec else ""
                    p_conf = plate_rec.best_ocr_conf if plate_rec else 0.0
                    p_crop = self.plate_manager.get_clearest_plate_crop(tid)
                    eff_s = self.speed_estimator.get_effective_speed(tid)
                    peak_s = self.speed_estimator.get_peak_speed(tid)
                    is_viol = state.is_confirmed_speeding

                    snap_frame = (
                        plate_rec.best_vehicle_snapshot
                        if (plate_rec and plate_rec.best_vehicle_snapshot is not None)
                        else frame
                    )

                    self._save_record(
                        frame=snap_frame,
                        track_id=tid,
                        bbox=state.last_bbox,
                        frame_idx=frame_idx,
                        timestamp_sec=timestamp_sec,
                        plate_bn=p_bn,
                        plate_en=p_en,
                        plate_conf=p_conf,
                        speed_kmh=eff_s,
                        peak_speed_kmh=peak_s,
                        is_speeding=is_viol,
                        plate_crop=p_crop,
                        lane_violation=state.lane_violation_reason if state.is_confirmed_lane_violation else "",
                    )
                    state.saved = True
                    total_saved += 1

            if show_window:
                cv2.destroyAllWindows()
            self._close_outputs()

            logger.info("=======================================================")
            logger.info("Pipeline Complete!")
            logger.info("Total Frames Processed: %d", frame_idx)
            logger.info(
                "Total Violations Detected: %d (Lane: %d, Speed: %d)",
                len(self._violating_tracks),
                len(self._confirmed_lane_violators),
                len(self._confirmed_speed_violators),
            )
            logger.info("Total Events Logged: %d", total_saved)
            logger.info("Annotated Video: %s", self._video_path)
            logger.info("CSV Database: %s", self._csv_path)
            logger.info("=======================================================")

            # Phase 3: High-Accuracy Deep ALPR, Optional GPT-4o Vision & Excel Export
            try:
                from process_crops import process_all_crops

                excel_path = str(self._output_dir / "traffic_violations_alpr.xlsx")
                process_all_crops(
                    crops_dir=str(self._crops_dir),
                    db_path=str(self._db_path),
                    excel_path=excel_path,
                    speed_limit_kmh=self._speed_limit_kmh,
                    speed_tolerance_kmh=self._speed_tolerance_kmh,
                    use_gpt=True,
                    gpt_api_key=self.cfg.get("ai", {}).get("openai_api_key") or None,
                )
            except Exception as e:
                logger.warning("[Pipeline] Could not auto-export Excel: %s", e)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bangladeshi License Plate ALPR & Speed Estimation Pipeline",
    )
    parser.add_argument("source", nargs="?", default=None, help="Video source path or webcam")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    parser.add_argument("--no-display", action="store_true", help="Disable preview window")
    parser.add_argument("--save-video", action="store_true", help="Force save annotated video")
    parser.add_argument("--no-save-video", action="store_true", help="Disable annotated video saving")
    parser.add_argument("--decimation", type=int, default=None, help="Override frame decimation")
    parser.add_argument("--fps", type=float, default=None, help="Override source FPS")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")
    parser.add_argument("--no-lane", action="store_true", help="Disable lane zone checks")
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


def main() -> None:
    from src.utils import load_config, setup_logging

    args = parse_args()
    cfg = load_config(args.config)
    apply_cli_overrides(cfg, args)
    setup_logging(cfg)

    logger.info("Starting Dedicated Plate & Speed Pipeline")
    pipeline = PlateSpeedPipeline(cfg, enable_lane=not args.no_lane)
    pipeline.run(max_frames=args.max_frames)


if __name__ == "__main__":
    main()
