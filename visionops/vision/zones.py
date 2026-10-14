from __future__ import annotations

from dataclasses import dataclass

from visionops.vision.geometry import Point, point_in_polygon


@dataclass(frozen=True)
class ZoneConfig:
    name: str
    polygon: list[Point]


@dataclass
class ZoneVisitEvent:
    track_id: int
    zone_name: str
    event_type: str
    timestamp: float
    dwell_seconds: float | None = None
    reason: str = "observed"
    censored: bool = False
    left_censored: bool = False


@dataclass
class _VisitState:
    inside: bool = False
    entered_at: float | None = None
    last_seen: float = 0.0
    left_censored: bool = False
    recovered_gap: bool = False


class ZoneMonitor:
    def __init__(self, zones: list[ZoneConfig]) -> None:
        self.zones = zones
        self._state: dict[tuple[int, str], _VisitState] = {}
        self.unique_visitors: dict[str, set[int]] = {zone.name: set() for zone in zones}

    def update(self, track_id: int, point: Point, timestamp: float) -> list[ZoneVisitEvent]:
        events: list[ZoneVisitEvent] = []
        for zone in self.zones:
            key = (track_id, zone.name)
            first_observation = key not in self._state
            state = self._state.setdefault(key, _VisitState())
            state.last_seen = timestamp
            inside = point_in_polygon(point, zone.polygon)
            if inside and not state.inside:
                state.inside = True
                state.entered_at = timestamp
                state.left_censored = first_observation
                state.recovered_gap = False
                self.unique_visitors[zone.name].add(track_id)
                events.append(ZoneVisitEvent(track_id, zone.name, "zone_entry", timestamp,
                                             reason="initial_observation" if first_observation else "observed",
                                             censored=first_observation, left_censored=first_observation))
            elif not inside and state.inside:
                entered_at = state.entered_at if state.entered_at is not None else timestamp
                dwell = max(timestamp - entered_at, 0.0)
                state.inside = False
                state.entered_at = None
                events.append(ZoneVisitEvent(track_id, zone.name, "zone_exit", timestamp, dwell,
                                             "unobserved_entry" if state.left_censored else "recovered_gap" if state.recovered_gap else "observed",
                                             state.left_censored or state.recovered_gap, state.left_censored))
        return events

    def mark_recovered_gap(self, track_id: int) -> None:
        for (identity, _), state in self._state.items():
            if identity == track_id and state.inside:
                state.recovered_gap = True

    def expire_missing(self, active_track_ids: set[int], timestamp: float,
                       grace_seconds: float = 1.0) -> list[ZoneVisitEvent]:
        events = []
        for (track_id, zone_name), state in self._state.items():
            if state.inside and track_id not in active_track_ids and timestamp - state.last_seen > grace_seconds:
                entered = state.entered_at if state.entered_at is not None else state.last_seen
                events.append(ZoneVisitEvent(track_id, zone_name, "zone_exit", state.last_seen,
                                             max(state.last_seen - entered, 0.0), "track_lost", True, state.left_censored))
                state.inside, state.entered_at = False, None
        return events

    def occupancy(self, active_track_ids: set[int] | None = None) -> dict[str, int]:
        return {zone.name: sum(state.inside and (active_track_ids is None or track_id in active_track_ids)
                              for (track_id, name), state in self._state.items() if name == zone.name)
                for zone in self.zones}

    def close_open_visits(self, timestamp: float) -> list[ZoneVisitEvent]:
        events: list[ZoneVisitEvent] = []
        for (track_id, zone_name), state in self._state.items():
            if state.inside:
                entered_at = state.entered_at if state.entered_at is not None else timestamp
                end = min(timestamp, state.last_seen)
                events.append(ZoneVisitEvent(track_id, zone_name, "zone_exit", end,
                                             max(end - entered_at, 0.0), "video_end", True, state.left_censored))
                state.inside = False
                state.entered_at = None
        return events

