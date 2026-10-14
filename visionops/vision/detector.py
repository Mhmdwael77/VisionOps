from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from visionops.vision.geometry import BBox
from visionops.vision.tracking_config import TrackingConfig


@dataclass(frozen=True)
class Detection:
    bbox: BBox
    confidence: float
    class_id: int
    track_id: int | None = None


class YOLOPersonDetector:
    def __init__(
        self,
        model_path: str | Path,
        confidence: float = 0.35,
        iou: float = 0.55,
        imgsz: int = 640,
        device: str | None = None,
        tracking: TrackingConfig | None = None,
    ) -> None:
        from ultralytics import YOLO

        self.model_path = str(model_path)
        self.model = YOLO(self.model_path)
        self.confidence = confidence
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.tracking = tracking or TrackingConfig()
        self.tracker_session = None

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = self.model.predict(
            frame,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
            classes=[0],
        )[0]
        detections: list[Detection] = []
        if result.boxes is None:
            return detections
        for box in result.boxes:
            cls = int(box.cls.item()) if box.cls is not None else -1
            if cls != 0:
                continue
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            conf = float(box.conf.item()) if box.conf is not None else 0.0
            detections.append(Detection(BBox(x1, y1, x2, y2), conf, cls))
        return detections

    def track_stream(self, source: str | int, stream: bool = True, vid_stride: int = 1, processing_fps: float = 30) -> Iterable:
        if not self.tracking.uses_upstream_baseline():
            return self._experimental_stream(source, vid_stride, processing_fps)
        return self.model.track(
            source=source,
            stream=stream,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            vid_stride=vid_stride,
            verbose=False,
            classes=[0],
        )

    def _experimental_stream(self, source, stride, fps):
        from visionops.vision.appearance import NativeFeatures, ONNXFeatures
        from visionops.vision.tracking import TrackingSession, apply_tracks, result_embeddings

        native = encoder = None
        if self.tracking.mode == "botsort_reid":
            if self.tracking.embedding_backend == "native":
                native = NativeFeatures(self.model)
            else:
                encoder = ONNXFeatures(self.tracking.embedding_model)
        self.tracker_session = TrackingSession(self.tracking, fps)
        try:
            for result in self.model.predict(source, stream=True, conf=self.confidence, iou=self.iou,
                imgsz=self.imgsz, device=self.device, vid_stride=stride, verbose=False, classes=[0]):
                embeddings = result_embeddings(result, encoder) if self.tracking.mode == "botsort_reid" else None
                rows = self.tracker_session.update(result.boxes.cpu().numpy(), result.orig_img, embeddings)
                result.identity_gaps = self.tracker_session.identity_gaps
                result.tracker_diagnostics = self.tracker_session.diagnostics.last
                yield apply_tracks(result, rows)
        finally:
            if native:
                native.close()

