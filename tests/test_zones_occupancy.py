from visionops.vision.geometry import Point
from visionops.vision.occupancy import OccupancyState
from visionops.vision.zones import ZoneConfig, ZoneMonitor


def test_zone_entry_exit_and_dwell_uses_timestamps() -> None:
    monitor = ZoneMonitor(
        [
            ZoneConfig(
                "promo",
                [Point(0, 0), Point(100, 0), Point(100, 100), Point(0, 100)],
            )
        ]
    )

    assert monitor.update(9, Point(150, 50), 0.0) == []
    entered = monitor.update(9, Point(50, 50), 10.0)
    assert len(entered) == 1
    assert entered[0].event_type == "zone_entry"
    exited = monitor.update(9, Point(150, 50), 16.5)
    assert len(exited) == 1
    assert exited[0].event_type == "zone_exit"
    assert exited[0].dwell_seconds == 6.5
    assert monitor.unique_visitors["promo"] == {9}


def test_occupancy_records_negative_inconsistency() -> None:
    occupancy = OccupancyState(initial=0)
    occupancy.record_exit()
    assert occupancy.current == 0
    assert occupancy.inconsistencies
    occupancy.reconcile(5)
    assert occupancy.current == 5


def test_missing_track_is_censored_at_last_video_observation() -> None:
    monitor = ZoneMonitor([ZoneConfig("cashier", [Point(0, 0), Point(100, 0), Point(100, 100), Point(0, 100)])])
    monitor.update(8, Point(40, 40), 2)
    monitor.update(8, Point(40, 40), 4)
    assert monitor.occupancy({8}) == {"cashier": 1}
    assert monitor.occupancy(set()) == {"cashier": 0}
    assert monitor.expire_missing(set(), 4.5) == []
    event = monitor.expire_missing(set(), 6)[0]
    assert event.timestamp == 4
    assert event.dwell_seconds == 2
    assert event.censored is True
    assert event.reason == "track_lost"
    assert monitor.close_open_visits(10) == []


def test_end_of_video_does_not_invent_unseen_dwell() -> None:
    monitor = ZoneMonitor([ZoneConfig("cashier", [Point(0, 0), Point(100, 0), Point(100, 100), Point(0, 100)])])
    monitor.update(8, Point(40, 40), 10)
    monitor.update(8, Point(40, 40), 12)
    event = monitor.close_open_visits(15)[0]
    assert event.timestamp == 12 and event.dwell_seconds == 2 and event.censored


def test_first_seen_inside_zone_has_unobserved_entry() -> None:
    monitor = ZoneMonitor([ZoneConfig("cashier", [Point(0, 0), Point(100, 0), Point(100, 100), Point(0, 100)])])
    entry = monitor.update(8, Point(40, 40), 10)[0]
    assert entry.censored and entry.left_censored
    exit_event = monitor.update(8, Point(140, 40), 12)[0]
    assert exit_event.dwell_seconds == 2
    assert exit_event.censored and exit_event.reason == "unobserved_entry"

