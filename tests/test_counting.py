from visionops.vision.counting import CountingLineConfig, CrossingDirection, LineCrossingCounter
from visionops.vision.geometry import LineSegment, Point


def test_directional_line_crossing_counts_entry_once() -> None:
    counter = LineCrossingCounter(
        CountingLineConfig(
            name="door",
            line=LineSegment(Point(0, 10), Point(100, 10)),
            entry_direction=CrossingDirection.negative_to_positive,
            min_displacement=2,
        )
    )

    assert counter.update(7, Point(50, 5), 0.0) is None
    event = counter.update(7, Point(50, 20), 1.0)
    assert event is not None
    assert event.event_type == "entry"
    assert event.track_id == 7

    assert counter.update(7, Point(50, 30), 2.0) is None
    assert counter.update(7, Point(50, 40), 3.0) is None


def test_reverse_crossing_counts_exit() -> None:
    counter = LineCrossingCounter(
        CountingLineConfig(
            name="door",
            line=LineSegment(Point(0, 10), Point(100, 10)),
            entry_direction=CrossingDirection.negative_to_positive,
            min_displacement=2,
        )
    )

    counter.update(3, Point(10, 30), 0.0)
    event = counter.update(3, Point(10, 5), 1.0)
    assert event is not None
    assert event.event_type == "exit"


def test_hysteresis_blocks_tiny_jitter_crossing() -> None:
    counter = LineCrossingCounter(
        CountingLineConfig(
            name="door",
            line=LineSegment(Point(0, 10), Point(100, 10)),
            min_displacement=10,
        )
    )

    counter.update(2, Point(50, 9.8), 0.0)
    assert counter.update(2, Point(50, 10.2), 0.1) is None


def test_line_extension_is_not_a_crossing() -> None:
    counter = LineCrossingCounter(CountingLineConfig("door", LineSegment(Point(0, 10), Point(100, 10)), min_displacement=2))
    counter.update(4, Point(150, 5), 0)
    assert counter.update(4, Point(150, 20), 1) is None


def test_slow_crossing_and_repeated_direction_protection() -> None:
    counter = LineCrossingCounter(CountingLineConfig("door", LineSegment(Point(0, 10), Point(100, 10)), min_displacement=8))
    events = [counter.update(4, Point(50, y), t) for t, y in enumerate(range(0, 22))]
    assert [e.event_type for e in events if e] == ["entry"]
    assert counter.update(4, Point(50, 0), 30).event_type == "exit"
    assert counter.update(4, Point(50, 20), 31) is None
    assert counter.update(4, Point(50, 0), 32) is None

