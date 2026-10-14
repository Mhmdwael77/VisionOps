from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate_model(model_path: str, data_yaml: str, split: str, imgsz: int, conf: float | None) -> dict:
    from ultralytics import YOLO

    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, split=split, imgsz=imgsz, conf=conf, verbose=False)
    return {
        "model": model_path,
        "split": split,
        "imgsz": imgsz,
        "confidence": conf,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLO detector checkpoints on one dataset split.")
    parser.add_argument("--data", required=True, help="Path to dataset data.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--pretrained", default="yolo11s.pt")
    parser.add_argument("--finetuned", default="VisionOps_best.pt")
    parser.add_argument("--output", default="reports/detector_evaluation.json")
    args = parser.parse_args()

    results = [
        evaluate_model(args.pretrained, args.data, args.split, args.imgsz, args.conf),
    ]
    if Path(args.finetuned).exists():
        results.append(evaluate_model(args.finetuned, args.data, args.split, args.imgsz, args.conf))
    else:
        results.append({"model": args.finetuned, "error": "checkpoint not found"})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "results": results}, indent=2))


if __name__ == "__main__":
    main()

