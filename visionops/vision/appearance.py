"""Quality-gated clothing descriptors and explicit deep appearance encoders."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def unit(vector):
    if vector is None:
        return None
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    return vector / norm if np.isfinite(vector).all() and norm > 1e-8 else None


def iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    area = lambda box: max(0, box[2]-box[0]) * max(0, box[3]-box[1])
    union = area(a)+area(b)-intersection
    return intersection/union if union else 0.0


@dataclass
class Observation:
    frame: int
    histogram: np.ndarray | None
    embedding: np.ndarray | None
    reliable: bool


def observe(frame, bbox, confidence, embedding, frame_index, config, overlapping=False):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = np.asarray(bbox).astype(int)
    clipped = x1 <= 1 or y1 <= 1 or x2 >= width-1 or y2 >= height-1
    x1, x2 = np.clip([x1, x2], 0, width)
    y1, y2 = np.clip([y1, y2], 0, height)
    w, h = x2-x1, y2-y1
    histogram = None
    sharp = False
    if w >= 12 and h >= 32:
        # Central torso and lower clothing, omitting face and crop-edge background.
        crop = frame[y1+int(h*.2):y1+int(h*.85), x1+int(w*.15):x2-int(w*.15)]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).reshape(-1)
        histogram = histogram / max(float(histogram.sum()), 1)
        sharp = cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var() >= 8
    reliable = bool(confidence >= config.reliable_confidence and not clipped and not overlapping
                    and w >= 12 and h >= 32 and 1.25 <= h/max(w, 1) <= 4.5 and sharp)
    return Observation(frame_index, histogram, unit(embedding), reliable)


def similarities(gallery, observation, frame, config):
    recent = [item for item in gallery if frame-item.frame <= config.gallery_max_age_frames]
    if not observation.reliable or len(recent) < config.gallery_min_observations:
        return None, None
    histograms = [item.histogram for item in recent if item.histogram is not None]
    embeddings = [item.embedding for item in recent if item.embedding is not None]
    color = None
    if histograms and observation.histogram is not None:
        reference = np.mean(histograms, axis=0)
        color = float(np.sqrt(reference * observation.histogram).sum())
    deep = None
    if embeddings and observation.embedding is not None:
        reference = unit(np.mean(embeddings, axis=0))
        if reference is not None:
            deep = float(np.clip(reference @ observation.embedding, -1, 1))
    return color, deep


class NativeFeatures:
    """Ultralytics 8.4.7 Detect pre-hook + its NMS-aligned object feature extraction."""
    def __init__(self, model):
        from ultralytics.nn.modules.head import Detect

        head = model.model.model[-1]
        if not isinstance(head, Detect) or getattr(head, "end2end", False):
            raise RuntimeError("Detector has no supported native embeddings; no replacement weights were loaded")
        def capture(_module, inputs):
            if model.predictor is not None:
                model.predictor._feats = list(inputs[0])
        self.hook = head.register_forward_pre_hook(capture)

    def close(self):
        self.hook.remove()


class ONNXFeatures:
    """Opt-in local person-ReID ONNX: RGB, 256x128, ImageNet normalization, NxD output."""
    def __init__(self, path):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Local ReID model not found: {path}; no automatic download")
        self.net = cv2.dnn.readNetFromONNX(str(path))

    def __call__(self, image, boxes):
        output = []
        height, width = image.shape[:2]
        for box in boxes:
            x1, y1, x2, y2 = np.asarray(box).astype(int)
            crop = image[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
            if not crop.size:
                raise ValueError("Empty person crop for ReID")
            rgb = cv2.cvtColor(cv2.resize(crop, (128, 256)), cv2.COLOR_BGR2RGB).astype(np.float32)/255
            rgb = (rgb-np.array([.485, .456, .406], np.float32))/np.array([.229, .224, .225], np.float32)
            self.net.setInput(rgb.transpose(2, 0, 1)[None])
            embedding = unit(self.net.forward())
            if embedding is None:
                raise ValueError("ReID encoder returned an invalid embedding")
            output.append(embedding)
        return np.asarray(output, dtype=np.float32)
