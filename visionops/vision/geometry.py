from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def bottom_center(self) -> Point:
        return Point((self.x1 + self.x2) / 2.0, self.y2)


@dataclass(frozen=True)
class LineSegment:
    start: Point
    end: Point

    def signed_distance(self, point: Point) -> float:
        return ((self.end.x - self.start.x) * (point.y - self.start.y)) - (
            (self.end.y - self.start.y) * (point.x - self.start.x)
        )


def point_in_polygon(point: Point, polygon: list[Point]) -> bool:
    if len(polygon) < 3:
        return False
    inside = False
    j = len(polygon) - 1
    for i, pi in enumerate(polygon):
        pj = polygon[j]
        intersects = ((pi.y > point.y) != (pj.y > point.y)) and (
            point.x < (pj.x - pi.x) * (point.y - pi.y) / ((pj.y - pi.y) or 1e-9) + pi.x
        )
        if intersects:
            inside = not inside
        j = i
    return inside

