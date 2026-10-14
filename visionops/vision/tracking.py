"""Shared tracking session using Ultralytics' maintained motion/assignment engines."""
import numpy as np

from visionops.vision.tracking_config import TrackingConfig


class TrackerDiagnostics:
    def __init__(self):
        self.last_seen = {}
        self.removed = set()
        self.recovered = 0
        self.gap_runs = 0
        self.last = {}

    def update(self, engine, ids, frame):
        gaps = {identity: frame-self.last_seen[identity]-1 for identity in ids
                if identity in self.last_seen and frame-self.last_seen[identity] > 1}
        self.recovered += len(gaps)
        self.gap_runs += len(gaps)
        for identity in ids:
            self.last_seen[identity] = frame
        self.removed.update(track.track_id for track in engine.removed_stracks)
        self.last = {"active": len(ids), "lost": len([track for track in engine.lost_stracks if track.track_id not in self.removed]),
                     "recovered": self.recovered, "recovered_this_frame": len(gaps), "removed": len(self.removed),
                     "generated_ids": len(self.last_seen), "within_id_gap_runs": self.gap_runs,
                     "gmc_failures": getattr(engine, "gmc_failures", 0)}
        return gaps


class TrackingSession:
    def __init__(self, config: TrackingConfig, processing_fps: float):
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import IterableSimpleNamespace, YAML
        from ultralytics.utils.checks import check_yaml

        self.config, self.processing_fps = config, processing_fps
        self.arguments = YAML.load(check_yaml("bytetrack.yaml" if config.mode == "bytetrack" else "botsort.yaml"))
        self.arguments.update(track_buffer=config.lost_buffer_frames, track_high_thresh=config.track_high_thresh,
                              track_low_thresh=config.track_low_thresh, new_track_thresh=config.new_track_thresh,
                              match_thresh=config.match_thresh)
        self.diagnostics = TrackerDiagnostics()
        if config.mode == "bytetrack":
            self.engine = BYTETracker(IterableSimpleNamespace(**self.arguments), frame_rate=30)
        else:
            from visionops.vision.appearance_tracker import AppearanceBOTSORT
            self.arguments.update(with_reid=config.mode == "botsort_reid", model="auto", gmc_method=config.gmc_method)
            self.engine = AppearanceBOTSORT(IterableSimpleNamespace(**self.arguments), config)
        # Ultralytics frame_rate scales buffers relative to 30. Here a buffer is
        # explicitly an update-frame count, as in its model.track callback.
        self.engine.max_time_lost = config.lost_buffer_frames
        self.identity_gaps = {}

    def update(self, boxes, image, embeddings=None):
        if self.config.mode == "botsort_reid" and len(boxes) and (embeddings is None or len(embeddings) != len(boxes)):
            raise RuntimeError("ReID embeddings are missing or not aligned with detections; refusing silent fallback")
        rows = self.engine.update(boxes, image, embeddings)
        if len(rows):
            ids, indices = rows[:, 4].astype(int), rows[:, -1].astype(int)
            if len(set(ids)) != len(ids) or len(set(indices)) != len(indices):
                raise RuntimeError("Tracker emitted a duplicate identity or detection assignment")
        else:
            ids = []
        self.identity_gaps = self.diagnostics.update(self.engine, ids, self.engine.frame_id)
        return rows


def result_embeddings(result, encoder=None):
    if encoder is not None:
        return encoder(result.orig_img, result.boxes.cpu().numpy().xyxy)
    features = getattr(result, "feats", None)
    if features is None:
        return None
    return features.cpu().numpy() if hasattr(features, "cpu") else np.asarray(features)


def apply_tracks(result, rows):
    import torch

    result.update(boxes=torch.as_tensor(rows[:, :7] if len(rows) else np.empty((0, 7)), dtype=torch.float32))
    return result
