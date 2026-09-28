from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Generator, Optional

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    required_sections = [
        "paths",
        "input",
        "vehicle_detection",
        "bytetrack",
        "lane_zone",
        "motion",
        "violation",
        "display",
        "output",
        "logging",
    ]
    missing = [section for section in required_sections if section not in cfg]
    if missing:
        raise KeyError(f"Missing config sections: {missing}")

    return cfg


def setup_logging(cfg: dict) -> None:
    log_cfg = cfg.get("logging", {})
    level_name = str(log_cfg.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    if log_cfg.get("log_to_console", True):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

    log_file = log_cfg.get("log_file")
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


class FrameIngester:
    """Read frames from source and apply decimation to control processing FPS."""

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._source = cfg["input"]["source"]
        self._target_process_fps = float(cfg["input"].get("target_process_fps", 10.0))
        self._forced_fps = float(cfg["input"].get("force_fps", 0.0))
        self._decimation_cfg = cfg["input"].get("frame_decimation", "auto")

        self._capture: Optional[cv2.VideoCapture] = None
        self._frame_index = 0
        self._processed_count = 0
        self._source_fps = 30.0
        self._decimation = 1
        self._processing_fps = 30.0

        self._open()

    @property
    def source_fps(self) -> float:
        return self._source_fps

    @property
    def processing_fps(self) -> float:
        return self._processing_fps

    @property
    def decimation(self) -> int:
        return self._decimation

    @property
    def frame_width(self) -> int:
        if self._capture and self._capture.isOpened():
            return int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        return 0

    @property
    def frame_height(self) -> int:
        if self._capture and self._capture.isOpened():
            return int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return 0

    def _open(self) -> None:
        source = self._source
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        self._capture = cv2.VideoCapture(source)
        if not self._capture.isOpened():
            raise RuntimeError(f"Could not open video source: {self._source}")

        measured_fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if self._forced_fps > 0:
            self._source_fps = self._forced_fps
        elif measured_fps > 0:
            self._source_fps = measured_fps
        else:
            self._source_fps = 30.0

        if isinstance(self._decimation_cfg, str) and self._decimation_cfg.lower() == "auto":
            self._decimation = max(1, int(round(self._source_fps / self._target_process_fps)))
        else:
            self._decimation = max(1, int(self._decimation_cfg))

        self._processing_fps = self._source_fps / self._decimation

        logger.info(
            "[FrameIngester] source=%s | source_fps=%.2f | decimation=1/%d | processing_fps=%.2f",
            self._source,
            self._source_fps,
            self._decimation,
            self._processing_fps,
        )

    def __iter__(self) -> Generator[tuple[int, float, np.ndarray], None, None]:
        while True:
            assert self._capture is not None
            ok, frame = self._capture.read()
            if not ok:
                break

            frame_idx = self._frame_index
            self._frame_index += 1

            if frame_idx % self._decimation != 0:
                continue

            self._processed_count += 1
            timestamp_sec = frame_idx / self._source_fps
            yield frame_idx, timestamp_sec, frame

    def release(self) -> None:
        if self._capture and self._capture.isOpened():
            self._capture.release()

    def __enter__(self) -> "FrameIngester":
        return self

    def __exit__(self, *args) -> None:
        self.release()


class FPSCounter:
    def __init__(self, window_size: int = 30) -> None:
        self._window_size = window_size
        self._times: list[float] = []

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now)

        if len(self._times) > self._window_size + 1:
            self._times.pop(0)

        if len(self._times) < 2:
            return 0.0

        elapsed = self._times[-1] - self._times[0]
        if elapsed <= 0:
            return 0.0

        return (len(self._times) - 1) / elapsed


def _as_bgr(color: list[int] | tuple[int, int, int]) -> tuple[int, int, int]:
    if len(color) != 3:
        return (0, 255, 0)
    return int(color[0]), int(color[1]), int(color[2])


def draw_tracked_vehicle(
    frame: np.ndarray,
    track_id: int,
    bbox: tuple[int, int, int, int],
    color: list[int] | tuple[int, int, int],
    thickness: int,
    subtitle: str | None = None,
) -> None:
    x1, y1, x2, y2 = bbox
    draw_color = _as_bgr(color)

    cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, thickness)

    label = f"ID:{track_id}"
    if subtitle:
        label = f"{label} {subtitle}"

    (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    top = max(0, y1 - lh - baseline - 6)
    cv2.rectangle(frame, (x1, top), (x1 + lw + 6, y1), draw_color, -1)
    cv2.putText(
        frame,
        label,
        (x1 + 3, y1 - baseline - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


def draw_hud(
    frame: np.ndarray,
    fps: float,
    frame_idx: int,
    active_tracks: int,
    total_violations: int,
) -> None:
    hud = (
        f"FPS:{fps:.1f} | Frame:{frame_idx} | Active:{active_tracks} | "
        f"Violations:{total_violations}"
    )
    cv2.putText(
        frame,
        hud,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def resize_for_display(frame: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-6:
        return frame

    width = max(1, int(frame.shape[1] * scale))
    height = max(1, int(frame.shape[0] * scale))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def resize_to_fit(frame: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    """Resize a frame to fit within a target window while preserving aspect ratio."""
    if max_width <= 0 or max_height <= 0:
        return frame

    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        return frame

    scale = min(max_width / width, max_height / height, 1.0)
    if abs(scale - 1.0) < 1e-6:
        return frame

    out_w = max(1, int(round(width * scale)))
    out_h = max(1, int(round(height * scale)))
    return cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
