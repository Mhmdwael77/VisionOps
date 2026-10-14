"""Ultralytics BoT-SORT adapter with conservative, quality-gated lost recovery.

The Kalman filter, GMC, low-score rescue and one-to-one LAP assignment remain
Ultralytics implementations. No post-hoc ID aliases or detector-weight swaps.
"""
from collections import Counter, deque

import numpy as np
from ultralytics.trackers.basetrack import TrackState
from ultralytics.trackers.bot_sort import BOTSORT, BOTrack
from ultralytics.trackers.utils import matching

from visionops.vision.appearance import iou, observe, similarities


def lost_pair_cost(predicted, detected, mahalanobis, gallery, observation, frame, age, config):
    if age > config.lost_buffer_frames or not np.isfinite(mahalanobis) or mahalanobis > 9.4877:
        return 1.0, "motion"
    predicted, detected = np.asarray(predicted), np.asarray(detected)
    psize, dsize = predicted[2:]-predicted[:2], detected[2:]-detected[:2]
    if np.any(psize <= 0) or np.any(dsize <= 0):
        return 1.0, "size"
    center_distance = np.linalg.norm((predicted[:2]+predicted[2:]-detected[:2]-detected[2:])/2)/np.linalg.norm(psize)
    if iou(predicted, detected) < config.min_recovery_iou or center_distance > config.max_center_distance:
        return 1.0, "spatial"
    if np.any(dsize/psize < .6) or np.any(dsize/psize > 1.6):
        return 1.0, "partial_body"
    color, deep = similarities(gallery, observation, frame, config)
    if deep is None or color is None:
        return 1.0, "unreliable_or_missing_features"
    if deep < config.min_embedding_similarity or color < config.min_color_similarity:
        return 1.0, "appearance"
    return .45*(1-iou(predicted, detected)) + .45*(1-deep) + .10*(1-color), "eligible"


def reject_ambiguous(costs, lost_rows, margin):
    costs = np.asarray(costs, dtype=np.float64).copy()
    original = costs.copy()
    for row in np.flatnonzero(lost_rows):
        eligible = np.sort(original[row][original[row] < 1])
        if len(eligible) > 1 and eligible[1]-eligible[0] < margin:
            costs[row, :] = 1
    for column in range(costs.shape[1]):
        eligible = np.sort(original[:, column][original[:, column] < 1])
        if len(eligible) > 1 and eligible[1]-eligible[0] < margin:
            for row in np.flatnonzero(lost_rows):
                if original[row, column] <= eligible[0]+margin:
                    costs[row, column] = 1
        for row in np.flatnonzero(lost_rows):
            active = original[~np.asarray(lost_rows, dtype=bool), column]
            if len(active) and np.any(active <= original[row, column]+margin):
                costs[row, column] = 1
    return costs.astype(np.float32)


class GalleryTrack(BOTrack):
    def __init__(self, xywh, score, cls, observation, config):
        # 8.4.7 BOTrack(feat=...) accesses features/alpha before creating them.
        super().__init__(xywh, score, cls, feat=None)
        self.gallery = deque(maxlen=config.gallery_size)
        self.observation = observation
        if observation.reliable:
            self.gallery.append(observation)
            if observation.embedding is not None:
                self.update_features(observation.embedding.copy())

    def _remember(self, other):
        if other.observation.reliable:
            self.gallery.append(other.observation)

    def update(self, other, frame_id):
        super().update(other, frame_id)
        self._remember(other)

    def re_activate(self, other, frame_id, new_id=False):
        super().re_activate(other, frame_id, new_id=new_id)
        self._remember(other)


class AppearanceBOTSORT(BOTSORT):
    def remove_duplicate_stracks(self, tracked, lost):
        if self.args.with_reid:
            # Upstream IoU-only pruning can discard a new, contradictory identity
            # in favor of an older lost track. Keep lost hypotheses until expiry;
            # LAP and the session's assertions still enforce exclusive detections.
            return tracked, lost
        return super().remove_duplicate_stracks(tracked, lost)

    def __init__(self, args, config):
        super().__init__(args, frame_rate=30)
        self.config = config
        self.rejections = Counter()
        self.current_image = None
        self.embedding_values = None
        self.detection_indices = {}
        self.occluded = set()
        self.last_warp = np.eye(2, 3)
        self.gmc_failures = 0
        # Also expose GMC failures instead of silently swallowing them upstream.
        apply = self.gmc.apply
        def compensated(image, boxes=None):
            try:
                self.last_warp = apply(image, boxes)
                if not np.isfinite(self.last_warp).all():
                    raise ValueError("Nonfinite camera motion")
            except Exception:
                self.gmc_failures += 1
                self.last_warp = np.eye(2, 3)
            return self.last_warp
        self.gmc.apply = compensated

    def update(self, results, img=None, feats=None):
        self.current_image = img
        self.embedding_values = feats
        self.detection_indices = {tuple(row): index for index, row in enumerate(results.data)}
        boxes = results.xyxy
        self.occluded = {i for i, a in enumerate(boxes) if any(i != j and iou(a, b) > self.config.occlusion_iou for j, b in enumerate(boxes))}
        # init_track uses the full frame and aligned features stored above.
        return super().update(results, img, feats=None)

    def init_track(self, results, img=None):
        tracks = []
        for index, (xywh, xyxy, score, cls) in enumerate(zip(results.xywh, results.xyxy, results.conf, results.cls)):
            global_index = self.detection_indices[tuple(results.data[index])]
            embedding = self.embedding_values[global_index] if self.args.with_reid and self.embedding_values is not None else None
            observation = observe(self.current_image, xyxy, score, embedding, self.frame_id, self.config, global_index in self.occluded)
            tracks.append(GalleryTrack(np.append(xywh, global_index), score, cls, observation, self.config))
        return tracks

    def get_dists(self, tracks, detections):
        raw = matching.iou_distance(tracks, detections)
        costs = matching.fuse_score(raw, detections) if self.args.fuse_score else raw.copy()
        lost = np.array([track.state == TrackState.Lost for track in tracks], dtype=bool)
        for row, track in enumerate(tracks):
            for column, detection in enumerate(detections):
                color, deep = similarities(track.gallery, detection.observation, self.frame_id, self.config)
                if self.args.with_reid and deep is not None and raw[row, column] <= 1-self.args.proximity_thresh:
                    distance = (1-deep)/2
                    if distance <= 1-self.args.appearance_thresh:
                        costs[row, column] = min(costs[row, column], distance)
                    if deep < .5:
                        costs[row, column] = 1
                if lost[row] and self.args.with_reid and self.config.recovery_enabled:
                    measurement = np.array([detection.convert_coords(detection.tlwh)])
                    mahalanobis = track.kalman_filter.gating_distance(track.mean, track.covariance, measurement)[0]
                    costs[row, column], reason = lost_pair_cost(track.xyxy, detection.xyxy, mahalanobis, track.gallery,
                        detection.observation, self.frame_id, self.frame_id-track.end_frame, self.config)
                    if reason != "eligible":
                        self.rejections[reason] += 1
        if self.args.with_reid and self.config.recovery_enabled:
            filtered = reject_ambiguous(costs, lost, self.config.ambiguity_margin)
            self.rejections["ambiguous_pairs"] += int(np.sum((costs < 1) & (filtered >= 1)))
            return filtered
        return costs
