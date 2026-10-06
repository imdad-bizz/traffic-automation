from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

Point = tuple[int, int]


@dataclass
class TrackMotionState:
    history: deque[tuple[int, Point, bool]] = field(default_factory=deque)
    last_seen_frame: int = 0


class MotionTracker:
    """Track centroid history per track ID for directional analysis."""

    def __init__(self, history_size: int = 20, stale_after_frames: int = 90) -> None:
        self._history_size = history_size
        self._stale_after_frames = stale_after_frames
        self._states: dict[int, TrackMotionState] = {}

    def update(
        self,
        track_id: int,
        centroid: Point,
        frame_idx: int,
        inside_lane: bool,
    ) -> None:
        state = self._states.get(track_id)
        if state is None:
            state = TrackMotionState(history=deque(maxlen=self._history_size))
            self._states[track_id] = state

        state.history.append((frame_idx, centroid, inside_lane))
        state.last_seen_frame = frame_idx

    def get_inside_points(self, track_id: int) -> list[Point]:
        state = self._states.get(track_id)
        if state is None:
            return []
        return [point for _, point, inside_lane in state.history if inside_lane]

    def get_start_frame(self, track_id: int) -> int | None:
        state = self._states.get(track_id)
        if state is None or not state.history:
            return None
        for frame_idx, _, inside_lane in state.history:
            if inside_lane:
                return frame_idx
        return None

    def get_displacement(self, track_id: int) -> np.ndarray | None:
        points = self.get_inside_points(track_id)
        if len(points) < 2:
            return None

        start = np.array(points[0], dtype=np.float32)
        end = np.array(points[-1], dtype=np.float32)
        return end - start

    def get_progress_delta(self, track_id: int, lane_zone) -> float | None:
        points = self.get_inside_points(track_id)
        if len(points) < 2:
            return None

        start_progress = lane_zone.progress_value(points[0])
        end_progress = lane_zone.progress_value(points[-1])
        return float(end_progress - start_progress)

    def get_start_progress_ratio(self, track_id: int, lane_zone) -> float | None:
        points = self.get_inside_points(track_id)
        if not points:
            return None
        return float(lane_zone.progress_ratio(points[0]))

    def remove(self, track_id: int) -> None:
        self._states.pop(track_id, None)

    def cleanup(self, active_ids: set[int], current_frame_idx: int) -> None:
        to_remove: list[int] = []
        for track_id, state in self._states.items():
            if track_id in active_ids:
                continue
            if current_frame_idx - state.last_seen_frame > self._stale_after_frames:
                to_remove.append(track_id)

        for track_id in to_remove:
            self.remove(track_id)
