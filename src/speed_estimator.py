"""
Speed Estimation Module (Perspective & Homography-Calibrated)
============================================================
Accurately estimates real-world vehicle velocities (km/h) from CCTV video
by mapping perspective-distorted camera coordinates to top-down ground coordinates.
Includes robust horizon guards, median filtering, and sustained speeding validation.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrackSpeedHistory:
    """Historical positions and timestamps for a single vehicle track."""

    track_id: int
    timestamps: deque[float] = field(default_factory=lambda: deque(maxlen=80))
    positions: deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=80))
    ground_positions: deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=80))
    smoothed_speed_kmh: float = 0.0
    peak_speed_kmh: float = 0.0
    speeds_history: deque[float] = field(default_factory=lambda: deque(maxlen=40))
    
    # Measurements specifically collected inside the validated measurement zone
    zone_speeds: List[float] = field(default_factory=list)
    zone_ground_points: List[Tuple[float, float]] = field(default_factory=list)
    speeding_samples_count: int = 0
    total_valid_measurements: int = 0


class PerspectiveSpeedEstimator:
    """
    Estimates real-world vehicle velocity in km/h using Inverse Perspective Mapping (IPM).
    
    Transforms 2D camera coordinates (x, y) into a metric top-down plane (meters)
    using a 4-point homography transform.
    Enriches speeding validation using temporal consistency and horizon exclusion.
    """

    def __init__(
        self,
        source_polygon: Sequence[Sequence[float]],
        ground_width_meters: float = 8.0,
        ground_length_meters: float = 30.0,
        speed_limit_kmh: float = 60.0,
        speed_tolerance_kmh: float = 5.0,
        min_track_frames: int = 8,
        min_sustained_frames: int = 6,
        min_measurement_y: float = 260.0,
        window_duration_sec: float = 0.5,
    ) -> None:
        self.ground_width = float(ground_width_meters)
        self.ground_length = float(ground_length_meters)
        self.speed_limit_kmh = float(speed_limit_kmh)
        self.speed_tolerance_kmh = float(speed_tolerance_kmh)
        self.violation_threshold_kmh = self.speed_limit_kmh + self.speed_tolerance_kmh
        self.min_track_frames = int(min_track_frames)
        self.min_sustained_frames = int(min_sustained_frames)
        self.min_measurement_y = float(min_measurement_y)
        self.window_duration_sec = float(window_duration_sec)

        self._homography_matrix = None
        self.source_polygon = [list(p) for p in source_polygon]
        self._init_homography(source_polygon)
        self._tracks: Dict[int, TrackSpeedHistory] = {}

    @classmethod
    def from_config(
        cls,
        cfg: dict,
        frame_size: Optional[Tuple[int, int]] = None,
    ) -> "PerspectiveSpeedEstimator":
        """
        Creates and calibrates a PerspectiveSpeedEstimator from configuration.
        Auto-scales the source polygon and horizon threshold if reference_resolution is set
        and differs from the actual video frame resolution.
        """
        speed_cfg = cfg.get("speed", {})
        source_poly = list(
            speed_cfg.get(
                "source_polygon",
                [[750, 180], [1350, 180], [1850, 980], [450, 980]],
            )
        )
        # Deep copy points
        source_poly = [[float(p[0]), float(p[1])] for p in source_poly]

        ref_res = speed_cfg.get("reference_resolution")
        if ref_res is None:
            ref_res = cfg.get("lane_zone", {}).get("reference_resolution")

        min_meas_y = float(speed_cfg.get("min_measurement_y", 260.0))

        if frame_size is not None and ref_res is not None:
            frame_w, frame_h = frame_size
            ref_w, ref_h = int(ref_res[0]), int(ref_res[1])
            if ref_w > 0 and ref_h > 0 and (ref_w != frame_w or ref_h != frame_h):
                sx = frame_w / float(ref_w)
                sy = frame_h / float(ref_h)
                source_poly = [
                    [float(p[0] * sx), float(p[1] * sy)]
                    for p in source_poly
                ]
                min_meas_y = min_meas_y * sy
                logger.info(
                    "[SpeedEstimator] Auto-scaled speed polygon to frame resolution (%dx%d) using scale (%.3f, %.3f)",
                    frame_w,
                    frame_h,
                    sx,
                    sy,
                )

        ground_w = float(speed_cfg.get("ground_width_meters", 8.0))
        ground_l = float(speed_cfg.get("ground_length_meters", 30.0))
        speed_limit = float(speed_cfg.get("speed_limit_kmh", 60.0))
        speed_tol = float(speed_cfg.get("speed_tolerance_kmh", 5.0))
        min_track = int(speed_cfg.get("min_track_frames", 8))
        min_sustained = int(speed_cfg.get("min_sustained_frames", 6))
        window_sec = float(speed_cfg.get("window_duration_sec", 0.5))

        return cls(
            source_polygon=source_poly,
            ground_width_meters=ground_w,
            ground_length_meters=ground_l,
            speed_limit_kmh=speed_limit,
            speed_tolerance_kmh=speed_tol,
            min_track_frames=min_track,
            min_sustained_frames=min_sustained,
            min_measurement_y=min_meas_y,
            window_duration_sec=window_sec,
        )

    def _init_homography(self, source_polygon: Sequence[Sequence[float]]) -> None:
        if len(source_polygon) != 4:
            raise ValueError("source_polygon must contain exactly 4 points: [TL, TR, BR, BL]")

        src = np.array(source_polygon, dtype=np.float32)
        dst = np.array(
            [
                [0.0, 0.0],
                [self.ground_width, 0.0],
                [self.ground_width, self.ground_length],
                [0.0, self.ground_length],
            ],
            dtype=np.float32,
        )

        self._homography_matrix = cv2.getPerspectiveTransform(src, dst)
        logger.info(
            "[SpeedEstimator] Calibrated homography IPM: %.1fx%.1f m | Limit: %.1f km/h (+%.1f buffer = %.1f km/h)",
            self.ground_width,
            self.ground_length,
            self.speed_limit_kmh,
            self.speed_tolerance_kmh,
            self.violation_threshold_kmh,
        )

    def to_ground(self, px: float, py: float) -> Tuple[float, float]:
        """Maps frame pixel coordinates (px, py) to ground coordinates (gx, gy) in meters."""
        pt = np.array([[[px, py]]], dtype=np.float32)
        res = cv2.perspectiveTransform(pt, self._homography_matrix)[0][0]
        return float(res[0]), float(res[1])

    def update_track(
        self,
        track_id: int,
        centroid: Tuple[float, float],
        timestamp_sec: float,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> float:
        """
        Updates tracking history for a vehicle and calculates its current smoothed speed.
        
        Args:
            track_id: Vehicle track identifier.
            centroid: (cx, cy) pixel coordinates.
            timestamp_sec: Frame timestamp in seconds.
            bbox: Optional (x1, y1, x2, y2) bounding box; if provided, uses the bottom-center
                  of the bbox (contact point with road surface) for superior perspective accuracy.
                  
        Returns:
            Calculated current speed in km/h.
        """
        if track_id not in self._tracks:
            self._tracks[track_id] = TrackSpeedHistory(track_id=track_id)

        history = self._tracks[track_id]

        # Use bottom-center of vehicle bounding box (tire contact with the ground plane)
        if bbox is not None:
            contact_x = (bbox[0] + bbox[2]) / 2.0
            contact_y = float(bbox[3])
        else:
            contact_x, contact_y = centroid

        # Horizon Guard: If vehicle contact point is too far in perspective horizon,
        # perspective compression turns 1-2 px detector jitter into 80+ km/h fake spikes.
        in_valid_zone = contact_y >= self.min_measurement_y

        gx, gy = self.to_ground(contact_x, contact_y)

        history.timestamps.append(timestamp_sec)
        history.positions.append((contact_x, contact_y))
        history.ground_positions.append((gx, gy))

        if len(history.timestamps) < self.min_track_frames or not in_valid_zone:
            return history.smoothed_speed_kmh

        # Find position from a sliding time window (approx. window_duration_sec ago)
        curr_t = history.timestamps[-1]
        curr_g = history.ground_positions[-1]

        target_t = curr_t - self.window_duration_sec
        ref_idx = 0
        for i in range(len(history.timestamps) - 1, -1, -1):
            if history.timestamps[i] <= target_t:
                ref_idx = i
                break

        # Check if reference point was also within or near valid zone
        ref_y = history.positions[ref_idx][1]
        if ref_y < (self.min_measurement_y - 40):
            return history.smoothed_speed_kmh

        dt = curr_t - history.timestamps[ref_idx]
        if dt >= 0.15:  # Require at least 150ms elapsed
            ref_g = history.ground_positions[ref_idx]
            dist_meters = float(
                np.sqrt((curr_g[0] - ref_g[0]) ** 2 + (curr_g[1] - ref_g[1]) ** 2)
            )
            speed_ms = dist_meters / dt
            inst_speed_kmh = speed_ms * 3.6

            # Jitter & Physical Sanity Filter (between 5 km/h and 160 km/h)
            if 5.0 <= inst_speed_kmh <= 160.0:
                # Discard abrupt physical jumps (> 40 km/h jump in 0.2s is tracker wobble)
                if not history.speeds_history or abs(inst_speed_kmh - history.smoothed_speed_kmh) < 40.0:
                    history.speeds_history.append(inst_speed_kmh)
                    history.zone_speeds.append(inst_speed_kmh)
                    history.zone_ground_points.append(curr_g)
                    history.total_valid_measurements += 1
                    if inst_speed_kmh > self.violation_threshold_kmh:
                        history.speeding_samples_count += 1

        # Smooth speed using median of recent values (robust against detector jitter)
        if history.speeds_history:
            smoothed = float(np.median(list(history.speeds_history)))
            history.smoothed_speed_kmh = smoothed
            if smoothed > history.peak_speed_kmh:
                history.peak_speed_kmh = smoothed

        return history.smoothed_speed_kmh

    def get_speed(self, track_id: int) -> float:
        """Returns the current smoothed speed in km/h for a track ID."""
        history = self._tracks.get(track_id)
        return history.smoothed_speed_kmh if history else 0.0

    def get_peak_speed(self, track_id: int) -> float:
        """Returns the highest recorded speed in km/h for a track ID."""
        history = self._tracks.get(track_id)
        return history.peak_speed_kmh if history else 0.0

    def get_median_speed(self, track_id: int) -> float:
        """Returns the robust median speed calculated across the valid zone."""
        history = self._tracks.get(track_id)
        if not history or not history.zone_speeds:
            return history.smoothed_speed_kmh if history else 0.0
        return float(np.median(history.zone_speeds))

    def get_effective_speed(self, track_id: int) -> float:
        """
        Returns the most representative speed for violation assessment.
        Uses robust median across the measurement zone.
        """
        history = self._tracks.get(track_id)
        if not history:
            return 0.0
        if len(history.zone_speeds) >= 3:
            return float(np.median(history.zone_speeds))
        return history.smoothed_speed_kmh

    def get_distance_traveled(self, track_id: int) -> float:
        """Returns the total ground distance traveled in meters inside the zone."""
        history = self._tracks.get(track_id)
        if not history or len(history.zone_ground_points) < 2:
            return 0.0
        p_first = history.zone_ground_points[0]
        p_last = history.zone_ground_points[-1]
        return float(np.sqrt((p_last[0] - p_first[0]) ** 2 + (p_last[1] - p_first[1]) ** 2))

    def is_confirmed_speeding(self, track_id: int) -> bool:
        """
        Enriched Speeding Validation:
        Confirms violation ONLY when:
        1. Vehicle has been tracked for minimum valid zone frames.
        2. Traveled at least 5.0 meters in the calibrated zone.
        3. Robust median speed exceeds the speed limit + tolerance buffer (e.g. 60 + 5 = 65 km/h),
           OR at least 70% of zone measurements exceed the violation threshold.
        """
        history = self._tracks.get(track_id)
        if not history:
            return False

        if history.total_valid_measurements < self.min_sustained_frames:
            return False

        dist = self.get_distance_traveled(track_id)
        if dist < 4.0:
            return False

        median_s = self.get_median_speed(track_id)
        ratio_speeding = history.speeding_samples_count / max(1, history.total_valid_measurements)

        if median_s > self.violation_threshold_kmh:
            return True

        if ratio_speeding >= 0.70 and median_s > self.speed_limit_kmh:
            return True

        return False

    def is_speeding(self, track_id: int) -> bool:
        """Backward-compatible speeding check using enriched validation."""
        return self.is_confirmed_speeding(track_id)

    def remove_track(self, track_id: int) -> Optional[TrackSpeedHistory]:
        """Removes a track ID from memory once lost."""
        return self._tracks.pop(track_id, None)
