import json

import pytest

from visionops.vision.tracking_experiment import evaluate_identities


def test_annotated_identity_evaluator_counts_known_switch_and_fragment(tmp_path):
    pytest.importorskip("motmetrics")
    truth = tmp_path/"truth.csv"
    truth.write_text("frame_index,person_id,x1,y1,x2,y2\n0,9,10,10,40,80\n1,9,10,10,40,80\n2,9,10,10,40,80\n")
    predictions = [
        {"frame_index": 0, "detections": [{"track_id": 1, "bbox": [10, 10, 40, 80]}]},
        {"frame_index": 1, "detections": []},
        {"frame_index": 2, "detections": [{"track_id": 2, "bbox": [10, 10, 40, 80]}]},
    ]
    (tmp_path/"tracks.jsonl").write_text("\n".join(json.dumps(row) for row in predictions))
    metrics = evaluate_identities(truth, tmp_path, 3)
    assert metrics["num_switches"] == 1
    assert metrics["num_fragmentations"] == 1
    assert metrics["num_misses"] == 1
    assert metrics["num_false_positives"] == 0
