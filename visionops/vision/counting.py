from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import hypot

from visionops.vision.geometry import LineSegment, Point


class CrossingDirection(str, Enum):
    negative_to_positive = "negative_to_positive"
    positive_to_negative = "positive_to_negative"


@dataclass(frozen=True)
class CountingLineConfig:
    name: str
    line: LineSegment
    entry_direction: CrossingDirection = CrossingDirection.negative_to_positive
    min_displacement: float = 8.0


@dataclass
class TrackLineState:
    last_point: Point | None = None
    last_side: int | None = None
    counted_directions: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class CrossingEvent:
    track_id: int
    line_name: str
    event_type: str
    timestamp: float


class LineCrossingCounter:
    def __init__(self, config: CountingLineConfig) -> None:
        self.config = config
        self._tracks: dict[int, TrackLineState] = {}

    def update(self, track_id: int, point: Point, timestamp: float) -> CrossingEvent | None:
        side_value = self.config.line.signed_distance(point)
        line = self.config.line
        length = hypot(line.end.x - line.start.x, line.end.y - line.start.y)
        if length == 0:
            raise ValueError("Counting line must have distinct endpoints")
        side = 1 if side_value > 0 else -1 if side_value < 0 else 0
        state = self._tracks.setdefault(track_id, TrackLineState())

        # Keep the last stable point through the dead band, including slow crossings.
        if side == 0 or abs(side_value) / length < self.config.min_displacement / 2:
            return None

        if state.last_side is None or state.last_side == 0:
            state.last_side = side
            state.last_point = point
            return None

        if state.last_side == side:
            state.last_point = point
            return None

        previous = state.last_point
        if previous is None:
            return None
        old_distance = line.signed_distance(previous)
        fraction = old_distance / (old_distance - side_value)
        intersection = Point(previous.x + fraction * (point.x - previous.x),
                             previous.y + fraction * (point.y - previous.y))
        projection = ((intersection.x - line.start.x) * (line.end.x - line.start.x)
                      + (intersection.y - line.start.y) * (line.end.y - line.start.y)) / length ** 2

        direction = (
            CrossingDirection.negative_to_positive
            if state.last_side < side
            else CrossingDirection.positive_to_negative
        )
        event_type = "entry" if direction == self.config.entry_direction else "exit"
        dedupe_key = f"{event_type}:{direction.value}"
        state.last_side = side
        state.last_point = point

        if not 0 <= projection <= 1:
            return None
        if dedupe_key in state.counted_directions:
            return None
        state.counted_directions.add(dedupe_key)
        return CrossingEvent(track_id, self.config.name, event_type, timestamp)

    def forget_missing(self, active_track_ids: set[int], max_age_prune: bool = False) -> None:
        if max_age_prune:
            for track_id in list(self._tracks):
                if track_id not in active_track_ids:
                    self._tracks.pop(track_id, None)

