from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Generator, Optional

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)

_last_beep_time: float = 0.0
_beep_lock = threading.Lock()


def trigger_beep(frequency_hz: int = 1200, duration_ms: int = 250, min_interval_sec: float = 0.4) -> None:
    """Trigger a non-blocking audio beep with debounced interval to prevent audio stutter."""
    global _last_beep_time
    now = time.time()
    with _beep_lock:
        if now - _last_beep_time < min_interval_sec:
            return
        _last_beep_time = now

    def _play_sound():
        try:
            import winsound
            winsound.Beep(int(frequency_hz), int(duration_ms))
        except Exception:
            pass

    threading.Thread(target=_play_sound, daemon=True).start()


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

    @property
    def total_frames(self) -> int:
        if self._capture and self._capture.isOpened():
            return int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
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

    is_unicode = any(ord(c) > 127 for c in label)
    if is_unicode:
        draw_bilingual_text(
            frame,
            label,
            (x1, max(0, y1 - 26)),
            font_size=18,
            color=(0, 0, 0),
            bg_color=draw_color,
        )
    else:
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
    extra: Optional[dict[str, str]] = None,
) -> None:
    parts = [
        f"FPS:{fps:.1f}",
        f"Frame:{frame_idx}",
        f"Active:{active_tracks}",
        f"Violations:{total_violations}",
    ]
    if extra:
        for k, v in extra.items():
            parts.append(f"{k}:{v}")
    hud = " | ".join(parts)
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


BN_TO_EN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def to_english_digits(text: str) -> str:
    """Translates Bengali numeral characters to English digits (০-৯ -> 0-9)."""
    return text.translate(BN_TO_EN_DIGITS)


_font_cache = {}


def draw_bilingual_text(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    font_size: int = 20,
    color: tuple[int, int, int] = (0, 255, 0),
    bg_color: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Draws text on an OpenCV frame with full Unicode/Bengali support using Pillow.
    
    If text only contains ASCII characters and no background is required, uses cv2.putText
    for speed. If Bengali characters are present, uses Windows Nirmala font.
    """
    if not text:
        return frame

    # Check for non-ASCII characters
    is_unicode = any(ord(c) > 127 for c in text)

    if not is_unicode and bg_color is None:
        cv2.putText(
            frame,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size / 30.0,
            _as_bgr(color),
            2,
            cv2.LINE_AA,
        )
        return frame

    try:
        from PIL import Image, ImageDraw, ImageFont

        if font_size not in _font_cache:
            font_path = "C:/Windows/Fonts/Nirmala.ttc"
            if Path(font_path).exists():
                _font_cache[font_size] = ImageFont.truetype(font_path, font_size)
            else:
                _font_cache[font_size] = ImageFont.load_default()
        font = _font_cache[font_size]

        # Convert OpenCV BGR to Pillow RGB
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        draw = ImageDraw.Draw(pil_img)

        x, y = position
        b, g, r = _as_bgr(color)
        text_color_rgb = (r, g, b)

        if bg_color is not None:
            bb, bg, br = _as_bgr(bg_color)
            bg_rgb = (br, bg, bb)
            bbox = draw.textbbox((x, y), text, font=font)
            # Add padding
            draw.rectangle((bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2), fill=bg_rgb)

        draw.text((x, y), text, font=font, fill=text_color_rgb)
        res = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        np.copyto(frame, res)
        return frame
    except Exception as e:
        logger.debug("[draw_bilingual_text] Fallback to cv2.putText: %s", e)
        cv2.putText(
            frame,
            to_english_digits(text),
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            _as_bgr(color),
            2,
            cv2.LINE_AA,
        )
        return frame

