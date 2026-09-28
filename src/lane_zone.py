from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

Point = tuple[int, int]


@dataclass(frozen=True)
class LaneGeometry:
    polygon: np.ndarray
    entry_midpoint: np.ndarray
    exit_midpoint: np.ndarray
    direction_unit: np.ndarray
    direction_length: float


class LaneZone:
    """
    Lane polygon and legal travel direction.

    Allowed direction is from edge P3-P4 (entry) toward P1-P2 (exit).
    """

    def __init__(self, points: Sequence[Sequence[int]]) -> None:
        if len(points) != 4:
            raise ValueError("lane_zone.points must contain exactly 4 points (P1..P4)")

        self.points: list[Point] = [(int(p[0]), int(p[1])) for p in points]
        polygon = np.array(self.points, dtype=np.int32)

        p1, p2, p3, p4 = [np.array(pt, dtype=np.float32) for pt in self.points]
        exit_midpoint = (p1 + p2) / 2.0
        entry_midpoint = (p3 + p4) / 2.0

        direction = exit_midpoint - entry_midpoint
        length = float(np.linalg.norm(direction))
        if length <= 1e-6:
            raise ValueError("Lane direction length is zero; check lane points")

        self.geometry = LaneGeometry(
            polygon=polygon,
            entry_midpoint=entry_midpoint,
            exit_midpoint=exit_midpoint,
            direction_unit=direction / length,
            direction_length=length,
        )

    @classmethod
    def from_config(
        cls,
        cfg: dict,
        frame_size: tuple[int, int] | None = None,
    ) -> "LaneZone":
        lane_cfg = cfg["lane_zone"]
        points = lane_cfg["points"]

        reference_resolution = lane_cfg.get("reference_resolution")
        if frame_size is not None and reference_resolution is not None:
            frame_w, frame_h = frame_size
            ref_w, ref_h = int(reference_resolution[0]), int(reference_resolution[1])

            if ref_w > 0 and ref_h > 0 and (ref_w != frame_w or ref_h != frame_h):
                sx = frame_w / ref_w
                sy = frame_h / ref_h
                points = [
                    [int(round(p[0] * sx)), int(round(p[1] * sy))]
                    for p in points
                ]

        return cls(points)

    def contains(self, point: Point) -> bool:
        return cv2.pointPolygonTest(
            self.geometry.polygon,
            (float(point[0]), float(point[1])),
            False,
        ) >= 0

    def progress_value(self, point: Point) -> float:
        vec = np.array(point, dtype=np.float32) - self.geometry.entry_midpoint
        return float(np.dot(vec, self.geometry.direction_unit))

    def progress_ratio(self, point: Point) -> float:
        return self.progress_value(point) / self.geometry.direction_length

    def distance_to_exit_side(self, point: Point) -> float:
        p1 = np.array(self.points[0], dtype=np.float32)
        p2 = np.array(self.points[1], dtype=np.float32)
        pt = np.array(point, dtype=np.float32)
        return _distance_point_to_segment(pt, p1, p2)

    def draw(
        self,
        frame: np.ndarray,
        polygon_color: tuple[int, int, int],
        thickness: int,
        draw_points: bool = True,
    ) -> None:
        cv2.polylines(
            frame,
            [self.geometry.polygon],
            isClosed=True,
            color=polygon_color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

        if draw_points:
            for idx, point in enumerate(self.points, start=1):
                cv2.circle(frame, point, 4, polygon_color, -1, lineType=cv2.LINE_AA)
                cv2.putText(
                    frame,
                    f"P{idx}",
                    (point[0] + 6, point[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    polygon_color,
                    2,
                    cv2.LINE_AA,
                )

        entry = tuple(np.round(self.geometry.entry_midpoint).astype(int))
        exit_pt = tuple(np.round(self.geometry.exit_midpoint).astype(int))
        cv2.arrowedLine(
            frame,
            entry,
            exit_pt,
            polygon_color,
            thickness,
            cv2.LINE_AA,
            tipLength=0.08,
        )


def _distance_point_to_segment(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    segment = b - a
    denom = float(np.dot(segment, segment))
    if denom <= 1e-6:
        return float(np.linalg.norm(point - a))

    t = float(np.dot(point - a, segment) / denom)
    t = max(0.0, min(1.0, t))
    projection = a + t * segment
    return float(np.linalg.norm(point - projection))
