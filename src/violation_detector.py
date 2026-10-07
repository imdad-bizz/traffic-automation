from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ViolationEvent:
    track_id: int
    frame_idx: int
    reason: str
    direction_score: float
    progress_delta: float
    centroid: tuple[int, int]
    bbox: tuple[int, int, int, int]
    speed_kmh: float = 0.0


class ViolationDetector:
    """
    Detect lane violations based on motion against lane direction.

    Primary violation:
      - Opposite-direction travel inside lane polygon.

    Secondary violation:
      - Suspicious entry from the forbidden side (P1-P2 side).
      
    Speed violation:
      - Exceeding the speed limit in the lane.
    """

    def __init__(self, cfg: dict) -> None:
        vcfg = cfg["violation"]
        self._min_inside_points = int(vcfg["min_inside_points"])
        self._min_displacement_px = float(vcfg["min_displacement_px"])
        self._min_progress_px = float(vcfg["min_progress_px"])
        self._opposite_score_threshold = float(vcfg["opposite_score_threshold"])
        self._forbidden_entry_progress_ratio = float(vcfg["forbidden_entry_progress_ratio"])
        self._forbidden_entry_max_forward_px = float(vcfg["forbidden_entry_max_forward_px"])
        
        # Speed params (disabled by default since SpeedEstimator handles calibrated IPM speed)
        self._check_speed = bool(vcfg.get("check_speed", False))
        self._lane_length_meters = float(vcfg.get("lane_length_meters", 30.0))
        self._speed_limit_kmh = float(vcfg.get("speed_limit_kmh", 60.0))
        self._min_frames_for_speed = int(vcfg.get("min_frames_for_speed", 10))

        self._flagged: dict[int, ViolationEvent] = {}

    def is_flagged(self, track_id: int) -> bool:
        return track_id in self._flagged

    def get_event(self, track_id: int) -> ViolationEvent | None:
        return self._flagged.get(track_id)

    def total_violations(self) -> int:
        return len(self._flagged)

    def evaluate(
        self,
        track_id: int,
        bbox: tuple[int, int, int, int],
        centroid: tuple[int, int],
        frame_idx: int,
        lane_zone,
        motion_tracker,
        fps: float = 30.0,
    ) -> ViolationEvent | None:
        if track_id in self._flagged:
            return None

        inside_points = motion_tracker.get_inside_points(track_id)
        if len(inside_points) < self._min_inside_points:
            return None

        displacement = motion_tracker.get_displacement(track_id)
        if displacement is None:
            return None

        displacement_norm = float(np.linalg.norm(displacement))
        if displacement_norm < self._min_displacement_px:
            return None

        direction_vector = displacement / displacement_norm
        direction_score = float(np.dot(direction_vector, lane_zone.geometry.direction_unit))

        progress_delta = motion_tracker.get_progress_delta(track_id, lane_zone)
        if progress_delta is None:
            return None

        reason: str | None = None
        speed_kmh = 0.0

        # Primary violation: Opposite-direction travel inside lane polygon
        if direction_score <= self._opposite_score_threshold or progress_delta <= -self._min_progress_px:
            reason = "opposite_direction"
        else:
            # Secondary violation: Entered near exit side (P1-P2) and moving backwards or lingering
            start_ratio = motion_tracker.get_start_progress_ratio(track_id, lane_zone)
            if (
                start_ratio is not None
                and start_ratio >= self._forbidden_entry_progress_ratio
                and progress_delta <= self._forbidden_entry_max_forward_px
                and direction_score < 0.1  # Must NOT be moving in legal forward direction
            ):
                reason = "forbidden_entry_side"
            elif self._check_speed and len(inside_points) >= self._min_frames_for_speed and fps > 0:
                start_frame = motion_tracker.get_start_frame(track_id)
                if start_frame is not None and frame_idx > start_frame:
                    time_elapsed = (frame_idx - start_frame) / fps
                    lane_len_px = float(getattr(lane_zone.geometry, "direction_length", 1.0) or 1.0)
                    norm_progress = abs(progress_delta) / max(1.0, lane_len_px)
                    dist_meters = norm_progress * self._lane_length_meters
                    if time_elapsed > 0.2 and dist_meters > 2.0:
                        speed_ms = dist_meters / time_elapsed
                        calculated_speed = speed_ms * 3.6
                        if calculated_speed > self._speed_limit_kmh:
                            reason = "speeding"
                            speed_kmh = calculated_speed

        if reason is None:
            return None

        event = ViolationEvent(
            track_id=track_id,
            frame_idx=frame_idx,
            reason=reason,
            direction_score=direction_score,
            progress_delta=progress_delta,
            centroid=centroid,
            bbox=bbox,
            speed_kmh=speed_kmh,
        )
        self._flagged[track_id] = event
        return event
