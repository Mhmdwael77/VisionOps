"""Feature-stability diagnostic on baseline IDs, explicitly not identity ground truth."""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visionops.vision.appearance import unit
from visionops.vision.retail_pipeline import distribution, write_json


root = Path("outputs/retail_store_test/tracking_experiment")
folder = root/json.loads((root/"latest.json").read_text())["relative_dir"]
with np.load(folder/"detections.npz", allow_pickle=False) as loaded:
    offsets, features = loaded["offsets"], loaded["features"]
    previous, similarities = {}, []
    for line in (folder/"bytetrack_30/tracks.jsonl").read_text().splitlines():
        frame = json.loads(line)
        index = frame["frame_index"]
        for detection in frame["detections"]:
            identity = detection["track_id"]
            vector = unit(features[offsets[index]+detection["detection_index"]])
            if vector is not None:
                if identity in previous and previous[identity][0] == index-1 and detection["confidence"] >= .5:
                    similarities.append(float(np.clip(previous[identity][1] @ vector, -1, 1)))
                previous[identity] = index, vector
result = {"consecutive_same_baseline_id_pairs": len(similarities), "cosine_similarity": distribution(similarities),
          "cosine_percentiles_5_25_50_75_95": np.percentile(similarities, [5, 25, 50, 75, 95]).tolist(),
          "below_recovery_threshold_0_85": sum(value < .85 for value in similarities),
          "interpretation": "Same emitted ByteTrack ID is a provisional alignment, not ground-truth person identity. Feature instability diagnostics are not identity accuracy or switch metrics."}
write_json(folder/"appearance_diagnostics.json", result)
print(json.dumps(result, indent=2))
