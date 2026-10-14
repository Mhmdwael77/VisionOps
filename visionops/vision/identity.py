"""Offline identity resolution for recorded video.

ByteTrack associates by motion only, so one person can be split across many IDs
(fragmentation) and one ID can hop between people during occlusion (switches).
After tracking, this module:

1. splits a raw track where its person-ReID appearance changes abruptly,
2. stitches non-overlapping segments that look like the same person,
3. drops identities observed for less than ``min_track_seconds``.

Appearance comes from a person-ReID model (OSNet trained on MSMT17), not from
detector features. Results are estimates; they are written to
``identity_resolution.json`` for audit.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np


def _unit(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


class OSNetEncoder:
    """torchreid OSNet person-ReID encoder; returns L2-normalized embeddings."""

    def __init__(self, weights: str, model_name: str = "osnet_ain_x1_0", device: str | None = None):
        import torch
        from torchreid.reid.utils import FeatureExtractor

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = FeatureExtractor(model_name=model_name, model_path=weights, device=device, verbose=False)

    def __call__(self, crops_rgb: list[np.ndarray]) -> np.ndarray:
        import torch

        with torch.no_grad():
            features = self.extractor(crops_rgb).cpu().numpy()
        return features / np.linalg.norm(features, axis=1, keepdims=True)


@dataclass
class Segment:
    raw_id: int
    start_frame: int
    end_frame: int
    first_seen: float
    last_seen: float
    embeddings: list = field(default_factory=list)  # (frame_index, vector)

    @property
    def mean(self):
        return _unit(np.mean([vector for _, vector in self.embeddings], axis=0)) if self.embeddings else None


class IdentityMap:
    def __init__(self, segments: list[Segment], identity_of: dict[int, int | None], report: dict):
        self._segments = defaultdict(list)
        for index, segment in enumerate(segments):
            self._segments[segment.raw_id].append((segment.start_frame, segment.end_frame, identity_of[index]))
        self.report = report

    def __call__(self, raw_id: int, frame_index: int) -> int | None:
        for start, end, identity in self._segments.get(raw_id, []):
            if start <= frame_index <= end:
                return identity
        return None


class IdentityResolver:
    def __init__(self, config, encoder=None, sample_stride: int = 2, batch_size: int = 64):
        self.config, self.encoder = config, encoder
        self.sample_stride, self.batch_size = sample_stride, batch_size
        self.observations = defaultdict(list)  # raw_id -> [(frame_index, timestamp)]
        self.embeddings = defaultdict(list)    # raw_id -> [(frame_index, vector)]
        self._batch, self._meta = [], []

    def observe(self, frame_index: int, timestamp: float, frame_bgr, detections: list[dict]) -> None:
        for detection in detections:
            if detection["track_id"] is not None:
                self.observations[detection["track_id"]].append((frame_index, timestamp))
        if self.encoder is None or frame_index % self.sample_stride:
            return
        for detection in detections:
            box = detection["bbox"]
            # Only clean, unoccluded, reasonably large crops describe a person reliably.
            if (detection["track_id"] is None or detection["confidence"] < 0.5 or box[3]-box[1] < 80
                    or any(_iou(box, other["bbox"]) > 0.2 for other in detections if other is not detection)):
                continue
            x1, y1, x2, y2 = (int(v) for v in box)
            crop = frame_bgr[max(0, y1):y2, max(0, x1):x2]
            if crop.size:
                self._batch.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                self._meta.append((detection["track_id"], frame_index))
        if len(self._batch) >= self.batch_size:
            self._flush()

    def _flush(self):
        if self._batch:
            for (raw_id, frame_index), vector in zip(self._meta, self.encoder(self._batch)):
                self.embeddings[raw_id].append((frame_index, vector))
        self._batch, self._meta = [], []

    def _split(self, raw_id: int) -> list[Segment]:
        """Cut a raw track wherever adjacent appearance windows disagree."""
        observed = self.observations[raw_id]
        vectors = self.embeddings.get(raw_id, [])
        window, threshold = 4, self.config.reid_split_similarity
        cuts = []

        def find(lo, hi):
            if hi - lo < 2*window:
                return
            scores = [(float(_unit(np.mean([v for _, v in vectors[i-window:i]], 0))
                             @ _unit(np.mean([v for _, v in vectors[i:i+window]], 0))), i)
                      for i in range(lo+window, hi-window+1)]
            score, index = min(scores)
            if score < threshold:
                cuts.append(index)
                find(lo, index)
                find(index, hi)

        if self.encoder is not None:
            find(0, len(vectors))
        boundaries = [(vectors[i-1][0] + vectors[i][0]) // 2 for i in sorted(cuts)]
        segments, start = [], observed[0][0]
        for boundary in boundaries + [observed[-1][0]]:
            span = [(f, t) for f, t in observed if start <= f <= boundary]
            if span:
                segments.append(Segment(raw_id, span[0][0], span[-1][0], span[0][1], span[-1][1],
                                        [(f, v) for f, v in vectors if span[0][0] <= f <= span[-1][0]]))
            start = boundary + 1
        return segments

    def resolve(self) -> IdentityMap:
        if self.encoder is not None:
            self._flush()
        segments = [segment for raw_id in sorted(self.observations) for segment in self._split(raw_id)]
        splits = len(segments) - len(self.observations)
        clusters = {index: [index] for index in range(len(segments))}
        merges = []
        if self.encoder is not None:
            clusters, merges = self._stitch(segments, clusters)
        # Order identities by first appearance and drop very short ones.
        ordered = sorted(clusters.values(), key=lambda members: min(segments[m].first_seen for m in members))
        identity_of, identities, dropped, next_id = {}, [], [], 1
        for members in ordered:
            first = min(segments[m].first_seen for m in members)
            last = max(segments[m].last_seen for m in members)
            raw = sorted({segments[m].raw_id for m in members})
            if last - first < self.config.min_track_seconds:
                dropped.append({"raw_track_ids": raw, "seconds": last-first})
                for m in members:
                    identity_of[m] = None
                continue
            for m in members:
                identity_of[m] = next_id
            identities.append({"id": next_id, "raw_track_ids": raw, "segments": len(members),
                               "first_seen": first, "last_seen": last})
            next_id += 1
        report = {"method": "OSNet-AIN person ReID split + stitch" if self.encoder is not None else "short-track filter only",
                  "raw_track_ids": len(self.observations), "segments_after_split": len(segments), "switch_splits": splits,
                  "stitch_merges": len(merges), "dropped_short_identities": len(dropped), "identities": len(identities),
                  "settings": {key: getattr(self.config, key) for key in
                               ("reid_split_similarity", "reid_merge_similarity", "reid_min_link_similarity",
                                "reid_max_gap_seconds", "reid_max_overlap_seconds", "min_track_seconds")},
                  "identity_table": identities, "dropped": dropped, "merges": merges,
                  "note": "Estimated person identities from appearance; not verified and not personal identification."}
        return IdentityMap(segments, identity_of, report)

    def _stitch(self, segments, clusters):
        """Agglomerative merge: never overlapping in time, within the gap limit, and every pair similar."""
        config, merges = self.config, []
        means = [segment.mean for segment in segments]

        def compatible(a, b):
            # A brief overlap is tolerated: around an ID switch the tracker often keeps
            # a box on the wrong person for a few frames. Real co-present people overlap longer.
            for x in a:
                for y in b:
                    sx, sy = segments[x], segments[y]
                    if min(sx.last_seen, sy.last_seen) - max(sx.first_seen, sy.first_seen) > config.reid_max_overlap_seconds:
                        return False
            end_a, start_b = max(segments[x].last_seen for x in a), min(segments[y].first_seen for y in b)
            end_b, start_a = max(segments[y].last_seen for y in b), min(segments[x].first_seen for x in a)
            return max(start_b-end_a, start_a-end_b) <= config.reid_max_gap_seconds

        while True:
            best = None
            keys = [k for k in clusters if all(means[m] is not None for m in clusters[k])]
            for i, a in enumerate(keys):
                for b in keys[i+1:]:
                    if not compatible(clusters[a], clusters[b]):
                        continue
                    similarity = [float(means[x] @ means[y]) for x in clusters[a] for y in clusters[b]]
                    if min(similarity) < config.reid_min_link_similarity:
                        continue
                    score = float(np.mean(similarity))
                    if score >= config.reid_merge_similarity and (best is None or score > best[0]):
                        best = (score, a, b)
            if best is None:
                return clusters, merges
            score, a, b = best
            merges.append({"similarity": score, "raw_track_ids": sorted({segments[m].raw_id for m in clusters[a] + clusters[b]})})
            clusters[a] += clusters.pop(b)
