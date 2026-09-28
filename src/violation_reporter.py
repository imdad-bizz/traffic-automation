from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional

import cv2

logger = logging.getLogger(__name__)


class ViolationReporter:
    """Persist violation evidence as snapshot PNG, annotated video, and CSV."""

    CSV_HEADERS = [
        "timestamp_sec",
        "frame_idx",
        "track_id",
        "reason",
        "direction_score",
        "progress_delta",
        "x1",
        "y1",
        "x2",
        "y2",
        "snapshot_path",
    ]

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg

        output_cfg = cfg["output"]
        paths_cfg = cfg["paths"]

        self._save_video = bool(output_cfg.get("save_annotated_video", True))
        self._output_dir = Path(paths_cfg["output_dir"])
        self._snapshot_dir = self._output_dir / "wrong_lane" / "snapshots"
        self._csv_path = Path(paths_cfg["csv_log"])
        self._video_path = Path(paths_cfg["annotated_video"])

        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._video_path.parent.mkdir(parents=True, exist_ok=True)

        self._csv_file = None
        self._csv_writer: Optional[csv.DictWriter] = None
        self._video_writer: Optional[cv2.VideoWriter] = None
        self._saved_tracks: set[int] = set()

    def open(self, width: int, height: int, fps: float) -> None:
        write_header = not self._csv_path.exists() or self._csv_path.stat().st_size == 0
        self._csv_file = open(self._csv_path, "a", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self.CSV_HEADERS)

        if write_header:
            self._csv_writer.writeheader()
            self._csv_file.flush()

        if self._save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._video_writer = cv2.VideoWriter(str(self._video_path), fourcc, fps, (width, height))
            if not self._video_writer.isOpened():
                logger.warning("[ViolationReporter] Could not open annotated video writer: %s", self._video_path)
                self._video_writer = None

    def write_frame(self, frame) -> None:
        if self._video_writer is not None:
            self._video_writer.write(frame)

    def _crop_vehicle(self, frame, bbox: tuple[int, int, int, int], padding: int = 10):
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

        snapshot_image = self._crop_vehicle(frame, event.bbox, padding=12)
        if snapshot_image is None:
            snapshot_image = frame.copy()

        reason_text = event.reason.replace("_", "-").upper()
        cv2.putText(
            snapshot_image,
            f"ID:{event.track_id}",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            snapshot_image,
            reason_text,
            (8, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        snapshot_name = f"track_{event.track_id}_frame_{event.frame_idx}.png"
        snapshot_path = self._snapshot_dir / snapshot_name
        cv2.imwrite(str(snapshot_path), snapshot_image)

        assert self._csv_writer is not None
        assert self._csv_file is not None

        x1, y1, x2, y2 = event.bbox
        self._csv_writer.writerow(
            {
                "timestamp_sec": f"{timestamp_sec:.3f}",
                "frame_idx": event.frame_idx,
                "track_id": event.track_id,
                "reason": event.reason,
                "direction_score": f"{event.direction_score:.4f}",
                "progress_delta": f"{event.progress_delta:.2f}",
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "snapshot_path": str(snapshot_path),
            }
        )
        self._csv_file.flush()
        self._saved_tracks.add(event.track_id)

        logger.info(
            "[ViolationReporter] violation track=%d reason=%s saved=%s",
            event.track_id,
            event.reason,
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

    def __enter__(self) -> "ViolationReporter":
        return self

    def __exit__(self, *args) -> None:
        self.close()
