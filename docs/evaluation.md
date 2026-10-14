# Evaluation

The checkpoint metrics in the training notebook are validation results. They should not be presented as production retail performance.

Use `evaluation/evaluate_detector.py` to evaluate pretrained YOLO11s and the fine-tuned checkpoint on the same dataset split:

```bash
python evaluation/evaluate_detector.py --data path/to/data.yaml --split test --finetuned VisionOps_best.pt
```

The script writes measured precision, recall, mAP50, and mAP50-95 to `reports/detector_evaluation.json`.

Counting accuracy requires labeled video crossing events. Recommended metrics:

- Counting MAE: `mean(abs(predicted_count - ground_truth_count))`.
- Counting accuracy: `1 - abs(predicted_count - ground_truth_count) / max(ground_truth_count, 1)`.
- ID switches: report only when tracking ground truth identities are available.
- Processing FPS and latency: benchmark on the target machine and model settings.

If labeled test videos are unavailable, record that counting accuracy, ID switches, and real deployment performance cannot be verified yet.

