from __future__ import annotations

import argparse
import csv
import datetime
import logging
import threading
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    import winsound
except ImportError:
    winsound = None

from src.tracker import build_detector_and_tracker
from src.utils import (
    FPSCounter,
    FrameIngester,
    draw_hud,
    draw_tracked_vehicle,
    draw_bilingual_text,
    load_config,
    resize_for_display,
    resize_to_fit,
    setup_logging,
)
from src.plate_detector import LicensePlateDetector, BanglaPlateOCR

logger = logging.getLogger(__name__)

Point = tuple[int, int]

# Light detection ROI polygon (non-rectangular), as provided.
LIGHT_BOX_POINTS: list[Point] = [
    (764, 41),
    (780, 41),
    (780, 65),
    (820, 63),
    (820, 86),
    (764, 93),
]

# Traffic signal stop line endpoints, as provided.
SIGNAL_LINE_POINTS: tuple[Point, Point] = ((1150, 390), (540, 700))
SIGNAL_STRIP_HALF_WIDTH_PX = 34

SIGNAL_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
    "green": (0, 200, 0),
}


@dataclass(frozen=True)
class TrafficViolationEvent:
    timestamp_sec: float
    frame_idx: int
    track_id: int
    signal_color: str
    bbox: tuple[int, int, int, int]


@dataclass
class TrackLineState:
    last_anchor: Point
    last_side: int
    was_in_strip: bool = False


class LightSignalDetector:
    """Detect traffic signal color inside the configured light ROI."""

    def __init__(
        self,
        polygon_points: list[Point],
        smoothing_window: int = 7,
        min_active_ratio: float = 0.015,
    ) -> None:
        self._polygon = np.array(polygon_points, dtype=np.int32)
        self._history: deque[str] = deque(maxlen=max(1, int(smoothing_window)))
        self._min_active_ratio = max(0.001, float(min_active_ratio))

        self._mask: np.ndarray | None = None
        self._mask_shape: tuple[int, int] | None = None
        self._min_active_pixels = 1

    def _ensure_mask(self, frame_shape: tuple[int, ...]) -> None:
        h, w = frame_shape[:2]
        shape_2d = (h, w)
        if self._mask is not None and self._mask_shape == shape_2d:
            return

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [self._polygon], 255)

        roi_pixels = int(cv2.countNonZero(mask))
        self._min_active_pixels = max(1, int(round(roi_pixels * self._min_active_ratio)))

        self._mask = mask
        self._mask_shape = shape_2d

        logger.info(
            "[LightSignalDetector] ROI pixels=%d | min_active_pixels=%d",
            roi_pixels,
            self._min_active_pixels,
        )

    def _masked_count(self, binary_mask: np.ndarray) -> int:
        assert self._mask is not None
        masked = cv2.bitwise_and(binary_mask, binary_mask, mask=self._mask)
        return int(cv2.countNonZero(masked))

    def detect(self, frame: np.ndarray) -> tuple[str, dict[str, int]]:
        self._ensure_mask(frame.shape)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        red_mask_1 = cv2.inRange(hsv, np.array([0, 70, 120]), np.array([12, 255, 255]))
        red_mask_2 = cv2.inRange(hsv, np.array([165, 70, 120]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)

        yellow_mask = cv2.inRange(hsv, np.array([18, 70, 120]), np.array([40, 255, 255]))
        green_mask = cv2.inRange(hsv, np.array([40, 60, 80]), np.array([90, 255, 255]))

        counts = {
            "red": self._masked_count(red_mask),
            "yellow": self._masked_count(yellow_mask),
            "green": self._masked_count(green_mask),
        }

        dominant_color, dominant_count = max(counts.items(), key=lambda item: item[1])

        # Keep the latest valid detections and use majority vote to suppress flicker.
        if dominant_count >= self._min_active_pixels:
            self._history.append(dominant_color)
        elif not self._history:
            self._history.append(dominant_color)

        if self._history:
            stable_color = Counter(self._history).most_common(1)[0][0]
            return stable_color, counts

        return dominant_color, counts

    def draw_roi(self, frame: np.ndarray) -> None:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self._polygon], (0, 0, 0))
        cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)

        cv2.polylines(frame, [self._polygon], isClosed=True, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)

        for idx, (x, y) in enumerate(self._polygon.tolist(), start=1):
            cv2.circle(frame, (x, y), 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.putText(
                frame,
                f"P{idx}",
                (x + 4, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )


class TrafficViolationReporter:
    CSV_HEADERS = [
        "pc_date",
        "pc_time",
        "timestamp_sec",
        "frame_idx",
        "track_id",
        "signal_color",
        "plate_text",
        "x1",
        "y1",
        "x2",
        "y2",
        "snapshot_path",
    ]

    def __init__(self, csv_path: Path, snapshot_dir: Path, cfg: dict | None = None) -> None:
        self._csv_path = csv_path
        self._snapshot_dir = snapshot_dir
        self.cfg = cfg or {}

        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)

        self._csv_file = None
        self._writer: csv.DictWriter | None = None
        self._saved_track_ids: set[int] = set()

        self._plate_detector = None
        self._plate_ocr = None
        try:
            plate_model = self.cfg.get("plate", {}).get("model_path", "models/plate_detector.pt")
            if Path(plate_model).exists():
                self._plate_detector = LicensePlateDetector(model_path=plate_model)
                easyocr_models = self.cfg.get("paths", {}).get("easyocr_models_dir", "models/EasyOCR/models")
                easyocr_user = self.cfg.get("paths", {}).get("easyocr_user_network_dir", "models/EasyOCR/user_network")
                self._plate_ocr = BanglaPlateOCR(
                    custom_model_dir=easyocr_models,
                    user_network_dir=easyocr_user,
                    use_gpu=bool(self.cfg.get("plate", {}).get("use_gpu", False)),
                )
        except Exception as e:
            logger.warning("[TrafficViolationReporter] Could not init ALPR: %s", e)

    def open(self) -> None:
        write_header = not self._csv_path.exists() or self._csv_path.stat().st_size == 0
        self._csv_file = open(self._csv_path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=self.CSV_HEADERS)

        if write_header:
            self._writer.writeheader()
            self._csv_file.flush()

    def _crop_vehicle(self, frame: np.ndarray, bbox: tuple[int, int, int, int], padding: int = 12) -> np.ndarray | None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)
        x2 = min(w, int(x2) + padding)
        y2 = min(h, int(y2) + padding)

        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2].copy()

    def report(self, event: TrafficViolationEvent, frame: np.ndarray) -> str | None:
        if event.track_id in self._saved_track_ids:
            return None

        if self._writer is None or self._csv_file is None:
            raise RuntimeError("TrafficViolationReporter is not open")

        snapshot = self._crop_vehicle(frame, event.bbox, padding=16)
        if snapshot is None:
            snapshot = frame.copy()

        plate_text = ""
        best_plate_crop = None
        best_plate_conf = 0.0

        if self._plate_detector is not None and self._plate_ocr is not None:
            plates = self._plate_detector.detect_in_vehicle_crop(
                snapshot, (0, 0, snapshot.shape[1], snapshot.shape[0])
            )
            for p in plates:
                if p.cropped_plate is not None and p.cropped_plate.size > 0:
                    text, conf = self._plate_ocr.recognize(p.cropped_plate)
                    if conf > best_plate_conf:
                        best_plate_conf = conf
                        plate_text = text
                        best_plate_crop = p.cropped_plate

        if best_plate_crop is not None and best_plate_crop.size > 0:
            sh, sw = snapshot.shape[:2]
            target_pw = min(sw // 2, 220)
            if target_pw > 40:
                scale = target_pw / float(best_plate_crop.shape[1])
                target_ph = max(20, int(best_plate_crop.shape[0] * scale))
                zoomed_p = cv2.resize(best_plate_crop, (target_pw, target_ph), interpolation=cv2.INTER_CUBIC)
                cv2.rectangle(zoomed_p, (0, 0), (target_pw - 1, target_ph - 1), (0, 255, 255), 2)
                px = max(0, sw - target_pw - 6)
                py = 6
                if py + target_ph < sh and px + target_pw < sw:
                    snapshot[py : py + target_ph, px : px + target_pw] = zoomed_p

        info_lines = [
            f"ID:{event.track_id}",
            "RED-LIGHT VIOLATION",
        ]
        if plate_text:
            info_lines.append(f"PLATE: {plate_text}")

        y_offset = 12
        for line in info_lines:
            color = (0, 0, 255) if "VIOLATION" in line else (0, 255, 0)
            draw_bilingual_text(snapshot, line, (8, y_offset), font_size=18, color=color, bg_color=(0, 0, 0))
            y_offset += 24

        snapshot_name = f"track_{event.track_id}_frame_{event.frame_idx}.png"
        snapshot_path = self._snapshot_dir / snapshot_name
        cv2.imwrite(str(snapshot_path), snapshot)

        # Non-blocking audio beep alert
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

        x1, y1, x2, y2 = event.bbox
        self._writer.writerow(
            {
                "pc_date": pc_date,
                "pc_time": pc_time,
                "timestamp_sec": f"{event.timestamp_sec:.3f}",
                "frame_idx": event.frame_idx,
                "track_id": event.track_id,
                "signal_color": event.signal_color,
                "plate_text": plate_text,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "snapshot_path": str(snapshot_path),
            }
        )
        self._csv_file.flush()
        self._saved_track_ids.add(event.track_id)

        logger.info(
            "[TrafficViolationReporter] Violation saved | track_id=%d | frame=%d | snapshot=%s",
            event.track_id,
            event.frame_idx,
            snapshot_path,
        )
        return str(snapshot_path)

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._writer = None


def _line_side(point: Point, line_a: Point, line_b: Point, epsilon: float = 2.0) -> int:
    cross = float((line_b[0] - line_a[0]) * (point[1] - line_a[1]) - (line_b[1] - line_a[1]) * (point[0] - line_a[0]))
    if cross > epsilon:
        return 1
    if cross < -epsilon:
        return -1
    return 0


def _orientation(a: Point, b: Point, c: Point) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _is_on_segment(a: Point, b: Point, p: Point, epsilon: float = 1.0) -> bool:
    return (
        min(a[0], b[0]) - epsilon <= p[0] <= max(a[0], b[0]) + epsilon
        and min(a[1], b[1]) - epsilon <= p[1] <= max(a[1], b[1]) + epsilon
    )


def _segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    o1 = _orientation(p1, p2, q1)
    o2 = _orientation(p1, p2, q2)
    o3 = _orientation(q1, q2, p1)
    o4 = _orientation(q1, q2, p2)

    s1 = 0 if abs(o1) < 1e-6 else (1 if o1 > 0 else -1)
    s2 = 0 if abs(o2) < 1e-6 else (1 if o2 > 0 else -1)
    s3 = 0 if abs(o3) < 1e-6 else (1 if o3 > 0 else -1)
    s4 = 0 if abs(o4) < 1e-6 else (1 if o4 > 0 else -1)

    if s1 != s2 and s3 != s4:
        return True

    if s1 == 0 and _is_on_segment(p1, p2, q1):
        return True
    if s2 == 0 and _is_on_segment(p1, p2, q2):
        return True
    if s3 == 0 and _is_on_segment(q1, q2, p1):
        return True
    if s4 == 0 and _is_on_segment(q1, q2, p2):
        return True

    return False


def _crossed_signal_line(
    previous_anchor: Point,
    current_anchor: Point,
    line_a: Point,
    line_b: Point,
    previous_side: int,
    current_side: int,
) -> bool:
    if previous_anchor == current_anchor:
        return False

    if previous_side == 0 and current_side == 0:
        return _segments_intersect(previous_anchor, current_anchor, line_a, line_b)

    if previous_side == 0 or current_side == 0:
        return _segments_intersect(previous_anchor, current_anchor, line_a, line_b)

    if previous_side == current_side:
        return False

    return _segments_intersect(previous_anchor, current_anchor, line_a, line_b)


def _build_signal_strip_polygon(line_a: Point, line_b: Point, half_width: int) -> list[Point]:
    ax, ay = line_a
    bx, by = line_b

    vec = np.array([bx - ax, by - ay], dtype=np.float32)
    length = float(np.linalg.norm(vec))
    if length <= 1e-6:
        raise ValueError("Signal line has zero length; cannot build strip")

    unit = vec / length
    normal = np.array([-unit[1], unit[0]], dtype=np.float32)
    offset = normal * float(max(1, half_width))

    a = np.array([ax, ay], dtype=np.float32)
    b = np.array([bx, by], dtype=np.float32)

    p1 = tuple(np.round(a + offset).astype(int))
    p2 = tuple(np.round(b + offset).astype(int))
    p3 = tuple(np.round(b - offset).astype(int))
    p4 = tuple(np.round(a - offset).astype(int))

    return [p1, p2, p3, p4]


def _bbox_corners(bbox: tuple[int, int, int, int]) -> list[Point]:
    x1, y1, x2, y2 = bbox
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def _point_in_bbox(point: Point, bbox: tuple[int, int, int, int]) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def _bbox_touches_strip(bbox: tuple[int, int, int, int], strip_polygon: list[Point]) -> bool:
    strip_np = np.array(strip_polygon, dtype=np.int32)
    corners = _bbox_corners(bbox)

    for corner in corners:
        if cv2.pointPolygonTest(strip_np, (float(corner[0]), float(corner[1])), False) >= 0:
            return True

    for point in strip_polygon:
        if _point_in_bbox(point, bbox):
            return True

    bbox_edges = [
        (corners[0], corners[1]),
        (corners[1], corners[2]),
        (corners[2], corners[3]),
        (corners[3], corners[0]),
    ]
    strip_edges = [
        (strip_polygon[0], strip_polygon[1]),
        (strip_polygon[1], strip_polygon[2]),
        (strip_polygon[2], strip_polygon[3]),
        (strip_polygon[3], strip_polygon[0]),
    ]

    for edge_a_start, edge_a_end in bbox_edges:
        for edge_b_start, edge_b_end in strip_edges:
            if _segments_intersect(edge_a_start, edge_a_end, edge_b_start, edge_b_end):
                return True

    return False


class TrafficLightViolationPipeline:
    def __init__(self, cfg: dict, args: argparse.Namespace) -> None:
        self.cfg = cfg
        self.args = args
        self._apply_vehicle_recall_tuning()

        self._line_a, self._line_b = SIGNAL_LINE_POINTS
        self._signal_strip_half_width = max(6, int(getattr(args, "strip_half_width", SIGNAL_STRIP_HALF_WIDTH_PX)))
        self._signal_strip_polygon = _build_signal_strip_polygon(
            self._line_a,
            self._line_b,
            self._signal_strip_half_width,
        )
        self._signal_strip_polygon_np = np.array(self._signal_strip_polygon, dtype=np.int32)
        self._light_detector = LightSignalDetector(LIGHT_BOX_POINTS)

        self._reporter = TrafficViolationReporter(
            csv_path=Path(args.csv),
            snapshot_dir=Path(args.snapshots_dir),
            cfg=cfg,
        )

        self._save_video = bool(cfg["output"].get("save_annotated_video", True))
        self._annotated_video_path = Path(args.annotated_video)
        self._annotated_video_path.parent.mkdir(parents=True, exist_ok=True)
        self._video_writer: cv2.VideoWriter | None = None

        self._fps_counter = FPSCounter(window_size=int(cfg["display"].get("fps_window", 30)))
        self._track_states: dict[int, TrackLineState] = {}
        self._violating_track_ids: set[int] = set()

        self._vehicle_detector = None
        self._vehicle_tracker = None

    def _apply_vehicle_recall_tuning(self) -> None:
        """Boost detector/tracker recall so trucks are less likely to be missed."""
        det_cfg = self.cfg["vehicle_detection"]
        bt_cfg = self.cfg["bytetrack"]
        input_cfg = self.cfg["input"]

        # Detector-side recall improvements.
        det_cfg["confidence_threshold"] = min(float(det_cfg.get("confidence_threshold", 0.35)), 0.22)
        det_cfg["iou_threshold"] = max(float(det_cfg.get("iou_threshold", 0.45)), 0.55)
        det_cfg["input_size"] = max(int(det_cfg.get("input_size", 640)), 960)

        target_classes = {int(c) for c in det_cfg.get("target_classes", [])}
        target_classes.update({2, 3, 5, 7})  # car, motorcycle, bus, truck
        det_cfg["target_classes"] = sorted(target_classes)

        # Tracker-side recall improvements in crowded scenes.
        bt_cfg["track_activation_threshold"] = min(
            float(bt_cfg.get("track_activation_threshold", 0.35)),
            0.25,
        )
        bt_cfg["minimum_matching_threshold"] = min(
            float(bt_cfg.get("minimum_matching_threshold", 0.75)),
            0.65,
        )

        # Process slightly more frames when decimation is automatic.
        if str(input_cfg.get("frame_decimation", "auto")).lower() == "auto":
            input_cfg["target_process_fps"] = max(float(input_cfg.get("target_process_fps", 12.0)), 15.0)

        logger.info(
            "[Pipeline] Recall tuning | conf=%.2f iou=%.2f imgsz=%d classes=%s track_activation=%.2f match=%.2f target_proc_fps=%.1f",
            det_cfg["confidence_threshold"],
            det_cfg["iou_threshold"],
            det_cfg["input_size"],
            det_cfg["target_classes"],
            bt_cfg["track_activation_threshold"],
            bt_cfg["minimum_matching_threshold"],
            float(input_cfg.get("target_process_fps", 0.0)),
        )

    def _update_tracking_runtime_params(self, processing_fps: float) -> None:
        bt_cfg = self.cfg["bytetrack"]
        bt_cfg["frame_rate"] = max(1, int(round(processing_fps)))

        if str(bt_cfg.get("lost_track_buffer", "auto")).lower() == "auto":
            bt_cfg["lost_track_buffer"] = max(30, int(round(processing_fps * 2.0)))

        logger.info(
            "[Pipeline] ByteTrack params | frame_rate=%d | lost_track_buffer=%d",
            bt_cfg["frame_rate"],
            bt_cfg["lost_track_buffer"],
        )

    def _open_video_writer(self, width: int, height: int, fps: float) -> None:
        if not self._save_video:
            return

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._video_writer = cv2.VideoWriter(str(self._annotated_video_path), fourcc, fps, (width, height))

        if not self._video_writer.isOpened():
            logger.warning("[Pipeline] Could not open annotated video writer: %s", self._annotated_video_path)
            self._video_writer = None

    def _close_video_writer(self) -> None:
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None

    def _draw_signal_strip(self, frame: np.ndarray, signal_color: str) -> tuple[int, int, int]:
        line_color = SIGNAL_COLORS_BGR.get(signal_color, (160, 160, 160))
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self._signal_strip_polygon_np], line_color)
        cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)

        cv2.polylines(frame, [self._signal_strip_polygon_np], isClosed=True, color=line_color, thickness=2, lineType=cv2.LINE_AA)
        cv2.line(frame, self._line_a, self._line_b, line_color, 2, cv2.LINE_AA)
        cv2.circle(frame, self._line_a, 5, line_color, -1, cv2.LINE_AA)
        cv2.circle(frame, self._line_b, 5, line_color, -1, cv2.LINE_AA)
        return line_color

    def _draw_signal_panel(self, frame: np.ndarray, signal_color: str, counts: dict[str, int], color: tuple[int, int, int]) -> None:
        text = (
            f"Signal:{signal_color.upper()} | "
            f"R:{counts['red']} Y:{counts['yellow']} G:{counts['green']}"
        )
        cv2.putText(
            frame,
            text,
            (12, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

    def run(self) -> None:
        display_cfg = self.cfg["display"]
        show_window = bool(display_cfg.get("show_window", True))
        window_name = str(display_cfg.get("window_name", "Traffic Light Violation Detection"))
        display_scale = float(display_cfg.get("display_scale", 0.7))
        fixed_window_size = bool(display_cfg.get("fixed_window_size", True))
        window_width = int(display_cfg.get("window_width", 1280))
        window_height = int(display_cfg.get("window_height", 720))
        line_thickness = int(display_cfg.get("line_thickness", 2))

        bbox_normal_color = tuple(display_cfg.get("bbox_color_normal", [0, 255, 0]))
        bbox_violation_color = tuple(display_cfg.get("bbox_color_violation", [0, 0, 255]))

        with FrameIngester(self.cfg) as ingester:
            self._update_tracking_runtime_params(ingester.processing_fps)
            self._vehicle_detector, self._vehicle_tracker = build_detector_and_tracker(self.cfg)

            self._reporter.open()
            self._open_video_writer(
                width=ingester.frame_width,
                height=ingester.frame_height,
                fps=max(1.0, ingester.processing_fps),
            )

            if show_window:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                if fixed_window_size:
                    cv2.resizeWindow(window_name, window_width, window_height)

            for frame_idx, timestamp_sec, frame in ingester:
                raw_frame = frame.copy()

                detections = self._vehicle_detector.detect(frame)
                tracked, lost_ids = self._vehicle_tracker.update(detections)

                signal_color, counts = self._light_detector.detect(frame)

                annotated = frame.copy()
                signal_line_color = self._draw_signal_strip(annotated, signal_color)
                self._draw_signal_panel(annotated, signal_color, counts, signal_line_color)

                for vehicle in tracked:
                    track_id = vehicle.track_id
                    bbox = vehicle.bbox
                    anchor = ((bbox[0] + bbox[2]) // 2, bbox[3])

                    current_side = _line_side(anchor, self._line_a, self._line_b)
                    touches_strip = _bbox_touches_strip(bbox, self._signal_strip_polygon)
                    state = self._track_states.get(track_id)

                    just_flagged = False
                    if track_id not in self._violating_track_ids and signal_color == "red":
                        crossed = False
                        entered_strip = touches_strip

                        if state is not None:
                            crossed = _crossed_signal_line(
                                previous_anchor=state.last_anchor,
                                current_anchor=anchor,
                                line_a=self._line_a,
                                line_b=self._line_b,
                                previous_side=state.last_side,
                                current_side=current_side,
                            )
                            entered_strip = touches_strip and not state.was_in_strip

                        if touches_strip and (entered_strip or crossed):
                            event = TrafficViolationEvent(
                                timestamp_sec=timestamp_sec,
                                frame_idx=frame_idx,
                                track_id=track_id,
                                signal_color=signal_color,
                                bbox=bbox,
                            )
                            self._reporter.report(event, raw_frame)
                            self._violating_track_ids.add(track_id)
                            just_flagged = True

                    self._track_states[track_id] = TrackLineState(
                        last_anchor=anchor,
                        last_side=current_side,
                        was_in_strip=touches_strip,
                    )

                    is_violator = track_id in self._violating_track_ids
                    box_color = bbox_violation_color if is_violator else bbox_normal_color
                    if is_violator:
                        subtitle = "RED-VIOL"
                    elif touches_strip:
                        subtitle = f"{signal_color.upper()} STRIP"
                    else:
                        subtitle = signal_color.upper()

                    draw_tracked_vehicle(
                        annotated,
                        track_id=track_id,
                        bbox=bbox,
                        color=box_color,
                        thickness=line_thickness,
                        subtitle=subtitle,
                    )
                    cv2.circle(annotated, anchor, 3, box_color, -1, lineType=cv2.LINE_AA)

                    if just_flagged:
                        cv2.putText(
                            annotated,
                            "RED-LIGHT VIOLATION",
                            (bbox[0], max(20, bbox[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )

                for lost_id in lost_ids:
                    self._track_states.pop(lost_id, None)

                fps_value = self._fps_counter.tick()
                draw_hud(
                    annotated,
                    fps=fps_value,
                    frame_idx=frame_idx,
                    active_tracks=len(tracked),
                    total_violations=len(self._violating_track_ids),
                )

                if self._video_writer is not None:
                    self._video_writer.write(annotated)

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

            self._close_video_writer()
            self._reporter.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Traffic light violation detection pipeline")
    parser.add_argument(
        "source",
        nargs="?",
        default="light_cut.mp4",
        help="Video source path, webcam index (0/1/2), or RTSP URL",
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    parser.add_argument("--no-display", action="store_true", help="Disable preview window")
    parser.add_argument("--save-video", action="store_true", help="Force save annotated video")
    parser.add_argument("--no-save-video", action="store_true", help="Disable annotated video saving")
    parser.add_argument("--decimation", type=int, default=None, help="Override frame decimation (1=every frame)")
    parser.add_argument("--fps", type=float, default=None, help="Override source FPS")
    parser.add_argument(
        "--strip-half-width",
        type=int,
        default=SIGNAL_STRIP_HALF_WIDTH_PX,
        help="Half-width in pixels for the stop-strip violation zone",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="outputs/traffic_light_violations.csv",
        help="Path for traffic-light violation CSV",
    )
    parser.add_argument(
        "--snapshots-dir",
        type=str,
        default="outputs/traffic_light_snapshots",
        help="Directory where violation snapshots are saved",
    )
    parser.add_argument(
        "--annotated-video",
        type=str,
        default="outputs/annotated_traffic_light_violation.mp4",
        help="Output path for annotated video",
    )
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
    args = parse_args()
    cfg = load_config(args.config)
    apply_cli_overrides(cfg, args)

    setup_logging(cfg)

    logger.info("Starting traffic light violation pipeline")
    logger.info("Input source: %s", cfg["input"]["source"])
    logger.info("Light ROI points: %s", LIGHT_BOX_POINTS)
    logger.info("Signal line points: %s -> %s", SIGNAL_LINE_POINTS[0], SIGNAL_LINE_POINTS[1])
    logger.info("Signal strip half-width: %d px", max(6, int(args.strip_half_width)))

    pipeline = TrafficLightViolationPipeline(cfg, args)
    pipeline.run()

    logger.info("Traffic light pipeline finished")


if __name__ == "__main__":
    main()
