from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - dependency guard
    raise RuntimeError("ultralytics is required. Install from requirements.txt") from exc

try:
    import supervision as sv
except ImportError as exc:  # pragma: no cover - dependency guard
    raise RuntimeError("supervision is required. Install from requirements.txt") from exc


@dataclass
class TrackedVehicle:
    track_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def centroid(self) -> tuple[int, int]:
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2


class VehicleDetector:
    """YOLO vehicle detector using the proven c_ALPR detection settings pattern."""

    def __init__(self, cfg: dict) -> None:
        model_path = cfg["paths"]["vehicle_model"]
        det_cfg = cfg["vehicle_detection"]

        logger.info("[VehicleDetector] Loading model: %s", model_path)
        self._model = YOLO(model_path)
        self._model.fuse()

        self._conf = float(det_cfg["confidence_threshold"])
        self._iou = float(det_cfg["iou_threshold"])
        self._classes = list(det_cfg["target_classes"])
        self._imgsz = int(det_cfg["input_size"])

    def detect(self, frame: np.ndarray) -> list[dict]:
        result = self._model(
            frame,
            conf=self._conf,
            iou=self._iou,
            classes=self._classes,
            imgsz=self._imgsz,
            verbose=False,
        )[0]

        boxes = result.boxes
        detections: list[dict] = []

        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": float(box.conf[0]),
                    "class_id": int(box.cls[0]),
                }
            )

        return detections


class VehicleTracker:
    """ByteTrack wrapper that returns active tracks and lost IDs each frame."""

    def __init__(self, cfg: dict) -> None:
        bt_cfg = cfg["bytetrack"]
        frame_rate = max(1, int(bt_cfg.get("frame_rate", 30)))

        raw_lost_buffer = bt_cfg.get("lost_track_buffer", 30)
        if isinstance(raw_lost_buffer, str) and raw_lost_buffer.lower() == "auto":
            lost_track_buffer = max(30, int(round(frame_rate * 2.0)))
        else:
            lost_track_buffer = max(1, int(raw_lost_buffer))

        self._tracker = sv.ByteTrack(
            track_activation_threshold=float(bt_cfg["track_activation_threshold"]),
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=float(bt_cfg["minimum_matching_threshold"]),
            frame_rate=frame_rate,
        )
        self._active_ids: set[int] = set()

    def update(self, detections: list[dict]) -> tuple[list[TrackedVehicle], set[int]]:
        if not detections:
            sv_dets = sv.Detections.empty()
        else:
            xyxy = np.array(
                [[d["x1"], d["y1"], d["x2"], d["y2"]] for d in detections],
                dtype=np.float32,
            )
            confidence = np.array([d["confidence"] for d in detections], dtype=np.float32)
            class_id = np.array([d["class_id"] for d in detections], dtype=int)
            sv_dets = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)

        tracked_sv = self._tracker.update_with_detections(sv_dets)

        tracked: list[TrackedVehicle] = []
        current_ids: set[int] = set()

        if tracked_sv.tracker_id is not None:
            for idx in range(len(tracked_sv)):
                x1, y1, x2, y2 = map(int, tracked_sv.xyxy[idx])
                if x2 <= x1 or y2 <= y1:
                    continue

                track_id = int(tracked_sv.tracker_id[idx])
                conf = float(tracked_sv.confidence[idx]) if tracked_sv.confidence is not None else 0.0
                cls = int(tracked_sv.class_id[idx]) if tracked_sv.class_id is not None else -1

                tracked.append(
                    TrackedVehicle(
                        track_id=track_id,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        confidence=conf,
                        class_id=cls,
                    )
                )
                current_ids.add(track_id)

        lost_ids = self._active_ids - current_ids
        self._active_ids = current_ids

        return tracked, lost_ids

    def reset(self) -> set[int]:
        flushed = set(self._active_ids)
        self._tracker.reset()
        self._active_ids.clear()
        return flushed


def build_detector_and_tracker(cfg: dict) -> tuple[VehicleDetector, VehicleTracker]:
    return VehicleDetector(cfg), VehicleTracker(cfg)
