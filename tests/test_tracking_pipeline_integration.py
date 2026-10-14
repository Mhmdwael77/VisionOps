import json
import shutil
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from ultralytics.engine.results import Results

from visionops.vision.retail_pipeline import run_retail_test
from visionops.vision.test_config import load_retail_config
from visionops.vision.tracking_config import TrackingConfig


def test_retail_loop_uses_recovered_engine_identity_for_visits(tmp_path, monkeypatch):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg needed for production-loop integration")
    source, checkpoint = tmp_path/"fixture.mp4", tmp_path/"fixture.pt"
    checkpoint.write_bytes(b"Isolated detector test double, not model weights")
    image = np.random.default_rng(5).integers(0, 255, (180, 200, 3), dtype=np.uint8)
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 12, (200, 180))
    for _ in range(6):
        writer.write(image)
    writer.release()
    class Detector:
        names = {0: "person"}
        ckpt_path = str(checkpoint)
        model = SimpleNamespace(yaml={}, parameters=lambda: [])
        def __init__(self, _path):
            self.frame = 0
        def predict(self, frame, **kwargs):
            boxes = np.array([[20, 20, 60, 120, .9, 0]], np.float32) if self.frame != 2 else np.empty((0, 6), np.float32)
            self.frame += 1
            return [Results(frame, path="fixture", names=self.names, boxes=boxes, speed={"inference": .1})]
    monkeypatch.setattr("ultralytics.YOLO", Detector)
    monkeypatch.setattr("visionops.vision.retail_pipeline.ONNXFeatures", lambda _path: lambda image, boxes: np.tile(np.array([[1, 0]], np.float32), (len(boxes), 1)))
    config = load_retail_config().model_copy(update={"source": str(source), "output_dir": str(tmp_path/"output"),
        "model_path": str(checkpoint), "device": "cpu", "tracking": TrackingConfig(mode="botsort_reid", embedding_backend="onnx", embedding_model="isolated-test-encoder", gmc_method="none", reid_enabled=False, min_track_seconds=0)})
    summary = run_retail_test(config)
    tracks = [json.loads(line) for line in (tmp_path/"output/tracks.jsonl").read_text().splitlines()]
    assert tracks[0]["detections"][0]["track_id"] == tracks[3]["detections"][0]["track_id"]
    assert summary["tracker"]["diagnostics"]["recovered"] == 1
    assert summary["tracker"]["label"] == "BoT-SORT + appearance"
    assert summary["analytics"]["unique_track_ids"] == 1
    visits = json.loads((tmp_path/"output/zone_visits.json").read_text())
    assert visits and all(visit["track_id"] == tracks[0]["detections"][0]["track_id"] for visit in visits)
    assert all(visit["censored"] for visit in visits)
