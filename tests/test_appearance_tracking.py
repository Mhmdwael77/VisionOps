from collections import deque

import cv2
import numpy as np
import pytest
from ultralytics.engine.results import Boxes
from ultralytics.trackers.utils import matching

from visionops.vision.appearance import Observation, observe
from visionops.vision.appearance_tracker import GalleryTrack, lost_pair_cost, reject_ambiguous
from visionops.vision.tracking import TrackingSession
from visionops.vision.tracking_config import TrackingConfig
from visionops.vision.geometry import Point
from visionops.vision.zones import ZoneConfig, ZoneMonitor


def observation(frame=1, embedding=None, reliable=True):
    return Observation(frame, np.array([.5, .5], np.float32),
                       np.array([1, 0] if embedding is None else embedding, np.float32), reliable)


def test_recovery_requires_deep_appearance_motion_and_reliable_gallery():
    cfg = TrackingConfig(mode="botsort_reid", lost_buffer_frames=30)
    gallery = [observation(1), observation(2)]
    box = [20, 20, 60, 120]
    cost, reason = lost_pair_cost(box, [22, 20, 62, 120], 1, gallery, observation(5), 5, 3, cfg)
    assert cost < .1 and reason == "eligible"
    for current, predicted, maha, age in [(observation(5, [-1, 0]), box, 1, 3),
                                           (observation(5, None, False), box, 1, 3),
                                           (observation(5), [200, 20, 240, 120], 1, 3),
                                           (observation(5), box, 12, 3), (observation(5), box, 1, 31)]:
        assert lost_pair_cost(predicted, box, maha, gallery, current, 5, age, cfg)[0] == 1
    no_embedding = observation(5)
    no_embedding.embedding = None
    assert lost_pair_cost(box, box, 1, gallery, no_embedding, 5, 3, cfg)[0] == 1


def test_similar_clothing_and_ambiguous_candidates_do_not_reuse_ids():
    # Equal candidates in either direction must not be forced into an assignment.
    assert np.all(reject_ambiguous([[.1, .12]], [True], .08) == 1)
    assert np.all(reject_ambiguous([[.1], [.12]], [True, True], .08) == 1)
    assert reject_ambiguous([[.1], [.08]], [True, False], .08)[0, 0] == 1
    costs = reject_ambiguous([[.1, .8], [.6, .1]], [True, True], .08)
    pairs, _, _ = matching.linear_assignment(costs, thresh=.8)
    assert len(pairs) == 2 and len({pair[1] for pair in pairs}) == 2


def textured_frame():
    rng = np.random.default_rng(5)
    return rng.integers(10, 245, (180, 200, 3), dtype=np.uint8)


def test_partial_occluded_and_low_confidence_crops_do_not_pollute_gallery():
    cfg, image = TrackingConfig(), textured_frame()
    assert observe(image, [20, 20, 60, 120], .9, [1, 0], 1, cfg).reliable
    for bbox, confidence, overlapping in [([0, 0, 40, 100], .9, False), ([20, 20, 60, 120], .3, False),
                                         ([20, 20, 60, 120], .9, True), ([20, 20, 60, 40], .9, False)]:
        assert not observe(image, bbox, confidence, [1, 0], 1, cfg, overlapping).reliable
    first = GalleryTrack(np.array([40, 70, 40, 100, 0]), .9, 0, observation(1), cfg)
    assert len(first.features) == len(first.gallery) == 1  # 8.4.7 constructor regression.
    first._remember(GalleryTrack(np.array([40, 70, 40, 100, 0]), .3, 0, observation(2, reliable=False), cfg))
    assert len(first.gallery) == 1


@pytest.mark.parametrize("buffer", [10, 30, 60])
def test_actual_botsort_preserves_original_id_after_confident_recovery(buffer):
    cfg = TrackingConfig(mode="botsort_reid", lost_buffer_frames=buffer, gmc_method="none")
    tracker, image = TrackingSession(cfg, 13.093), textured_frame()
    boxes = Boxes(np.array([[20, 20, 60, 120, .9, 0]], np.float32), image.shape[:2])
    embedding = np.array([[1, 0]], np.float32)
    initial = int(tracker.update(boxes, image, embedding)[0, 4])
    tracker.update(boxes, image, embedding)
    for _ in range(buffer-1):
        tracker.update(Boxes(np.empty((0, 6), np.float32), image.shape[:2]), image, np.empty((0, 2)))
    recovered = tracker.update(boxes, image, embedding)
    assert int(recovered[0, 4]) == initial
    assert tracker.diagnostics.last["recovered"] == 1
    assert tracker.engine.max_time_lost == buffer


def test_expired_id_and_incompatible_embedding_are_not_recovered():
    cfg = TrackingConfig(mode="botsort_reid", lost_buffer_frames=2, gmc_method="none")
    image = textured_frame()
    tracker = TrackingSession(cfg, 13.093)
    boxes = Boxes(np.array([[20, 20, 60, 120, .9, 0]], np.float32), image.shape[:2])
    old = int(tracker.update(boxes, image, np.array([[1, 0]], np.float32))[0, 4])
    tracker.update(boxes, image, np.array([[1, 0]], np.float32))
    empty = Boxes(np.empty((0, 6), np.float32), image.shape[:2])
    for _ in range(4):
        tracker.update(empty, image, np.empty((0, 2)))
    tracker.update(boxes, image, np.array([[-1, 0]], np.float32))
    rows = tracker.update(boxes, image, np.array([[-1, 0]], np.float32))
    assert int(rows[0, 4]) != old
    assert tracker.diagnostics.last["recovered"] == 0


def test_actual_tracker_rejects_lookalike_color_with_incompatible_embedding():
    cfg = TrackingConfig(mode="botsort_reid", gmc_method="none")
    image, tracker = textured_frame(), TrackingSession(cfg, 13.093)
    boxes = Boxes(np.array([[20, 20, 60, 120, .9, 0]], np.float32), image.shape[:2])
    old = int(tracker.update(boxes, image, np.array([[1, 0]], np.float32))[0, 4])
    tracker.update(boxes, image, np.array([[1, 0]], np.float32))
    tracker.update(Boxes(np.empty((0, 6), np.float32), image.shape[:2]), image, np.empty((0, 2)))
    tracker.update(boxes, image, np.array([[-1, 0]], np.float32))
    rows = tracker.update(boxes, image, np.array([[-1, 0]], np.float32))
    assert int(rows[0, 4]) != old


def test_duplicate_assignment_causes_explicit_failure(monkeypatch):
    tracker = TrackingSession(TrackingConfig(), 13.093)
    duplicate = np.array([[20, 20, 60, 120, 1, .9, 0, 0], [22, 20, 62, 120, 2, .9, 0, 0]], np.float32)
    monkeypatch.setattr(tracker.engine, "update", lambda *args: duplicate)
    with pytest.raises(RuntimeError, match="duplicate"):
        tracker.update(Boxes(np.array([[20, 20, 60, 120, .9, 0]], np.float32), (180, 200)), textured_frame())


def test_camera_translation_uses_motion_compensation_without_id_aliases():
    cfg = TrackingConfig(mode="botsort_reid", gmc_method="sparseOptFlow")
    tracker = TrackingSession(cfg, 13.093)
    image = np.random.default_rng(15).integers(0, 255, (400, 400, 3), dtype=np.uint8)
    def boxes(offset):
        return Boxes(np.array([[80+offset, 80, 140+offset, 220, .9, 0]], np.float32), (400, 400))
    embedding = np.array([[1, 0]], np.float32)
    old = int(tracker.update(boxes(0), image, embedding)[0, 4])
    tracker.update(boxes(0), image, embedding)
    shifted = cv2.warpAffine(image, np.float32([[1, 0, 5], [0, 1, 0]]), (400, 400))
    tracker.update(Boxes(np.empty((0, 6), np.float32), (400, 400)), shifted, np.empty((0, 2)))
    shifted = cv2.warpAffine(image, np.float32([[1, 0, 10], [0, 1, 0]]), (400, 400))
    recovered = tracker.update(boxes(10), shifted, embedding)
    assert int(recovered[0, 4]) == old and tracker.engine.gmc_failures == 0


def test_recovered_zone_visit_is_continuous_but_gap_censored():
    zone = ZoneMonitor([ZoneConfig("aisle", [Point(0, 0), Point(10, 0), Point(10, 10), Point(0, 10)])])
    zone.update(1, Point(12, 5), 0)
    zone.update(1, Point(5, 5), 1)
    assert zone.expire_missing(set(), 2, grace_seconds=5) == []
    zone.mark_recovered_gap(1)
    zone.update(1, Point(6, 5), 3)
    event = zone.update(1, Point(12, 5), 4)[0]
    assert event.track_id == 1 and event.dwell_seconds == 3
    assert event.censored and event.reason == "recovered_gap"
