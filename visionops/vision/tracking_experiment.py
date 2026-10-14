"""Controlled full-video tracker replay: identical detections, weights and geometry."""
from collections import Counter, defaultdict, deque
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np

from visionops.vision.appearance import NativeFeatures, ONNXFeatures
from visionops.vision.counting import CountingLineConfig, CrossingDirection, LineCrossingCounter
from visionops.vision.geometry import BBox, LineSegment, Point
from visionops.vision.retail_pipeline import digest, distribution, write_json
from visionops.vision.tracking import TrackingSession, result_embeddings
from visionops.vision.zones import ZoneConfig, ZoneMonitor


ROOT = Path("outputs/retail_store_test/tracking_experiment")


def collect_detections(config, output, status):
    import torch
    import ultralytics
    from ultralytics import YOLO

    if not Path(config.model_path).is_file():
        raise FileNotFoundError("Comparison requires the configured detector checkpoint; no fallback or download is allowed")
    started = time.perf_counter()
    model = YOLO(config.model_path)
    person_ids = [key for key, value in model.names.items() if str(value).lower() in {"person", "people"}]
    if not person_ids:
        raise ValueError("Detector has no person class")
    device = config.device or ("0" if torch.cuda.is_available() else "cpu")
    native = NativeFeatures(model) if config.tracking.embedding_backend == "native" else None
    encoder = ONNXFeatures(config.tracking.embedding_model) if native is None else None
    cap = cv2.VideoCapture(config.source)
    if not cap.isOpened():
        raise RuntimeError("Cannot open comparison video")
    fps, total = cap.get(5), int(cap.get(7))
    if fps <= 0:
        raise ValueError("No valid source video FPS")
    boxes, features, timestamps, inference, predict_ms = [], [], [], [], []
    offsets = [0]
    index, previous = 0, -1
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
            if timestamp <= previous:
                timestamp = index/fps
            if timestamp <= previous:
                raise RuntimeError("Nonmonotonic video timestamps")
            previous = timestamp
            if device != "cpu" and torch.cuda.is_available():
                torch.cuda.synchronize()
            call = time.perf_counter()
            result = model.predict(image, conf=config.confidence, iou=config.iou, imgsz=config.imgsz,
                                   classes=person_ids, device=device, verbose=False)[0]
            embedding = result_embeddings(result, encoder)
            if device != "cpu" and torch.cuda.is_available():
                torch.cuda.synchronize()
            predict_ms.append((time.perf_counter()-call)*1000)
            detected = result.boxes.data.cpu().numpy()
            if len(detected) and (embedding is None or len(embedding) != len(detected)):
                raise RuntimeError("Native/local ReID feature extraction did not return aligned embeddings")
            boxes.append(detected)
            if len(detected):
                features.append(embedding)
            offsets.append(offsets[-1]+len(detected))
            timestamps.append(timestamp)
            inference.append(float(result.speed["inference"]))
            index += 1
            if index % 100 == 0:
                status(phase="detecting", processed_frames=index, total_frames=total)
                print(f"Shared detections {index}/{total}", flush=True)
    finally:
        cap.release()
        if native:
            native.close()
    if index != total or not features:
        raise RuntimeError("Incomplete detection cache or no usable person embeddings")
    cache = {"boxes": np.concatenate(boxes), "features": np.concatenate(features), "offsets": np.array(offsets),
             "timestamps": np.array(timestamps), "inference_ms": np.array(inference), "predict_ms": np.array(predict_ms)}
    np.savez_compressed(output/"detections.npz", **cache)
    metadata = {"source": config.source, "source_sha256": digest(Path(config.source)), "checkpoint": config.model_path,
                "checkpoint_sha256": digest(Path(config.model_path)), "fallback_used": False,
                "confidence": config.confidence, "iou": config.iou, "imgsz": config.imgsz,
                "frames": index, "fps": fps, "processing_frame_stride": 1, "duration_seconds": index/fps,
                "ultralytics": ultralytics.__version__, "device": device, "embedding_backend": config.tracking.embedding_backend,
                "embedding_model": config.tracking.embedding_model,
                "embedding_model_sha256": digest(Path(config.tracking.embedding_model)) if encoder else None,
                "embedding_dimensions": int(cache["features"].shape[1]),
                "embedding_note": "Native YOLO detection-head input features; not a person-ReID-trained checkpoint" if native else "User-supplied local ONNX; training provenance and preprocessing must be validated",
                "detection_seconds": time.perf_counter()-started, "prediction_ms": distribution(predict_ms),
                "inference_ms": distribution(inference), "cache_sha256": digest(output/"detections.npz")}
    write_json(output/"detection_metadata.json", metadata)
    return cache, metadata


def analytics_tools(config, width, height):
    def point(pair):
        return Point(pair[0]*width, pair[1]*height)
    line = config.counting_line
    counter = LineCrossingCounter(CountingLineConfig(line.name, LineSegment(point(line.start), point(line.end)),
        CrossingDirection(line.entry_direction), line.min_displacement))
    zones = ZoneMonitor([ZoneConfig(zone.name, [point(pair) for pair in zone.polygon]) for zone in config.zones])
    return counter, zones


def encode_and_check(working, target, fps, frames, shape):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for comparison video encoding")
    with (target.parent/"encoding.log").open("w", encoding="utf-8") as log:
        subprocess.run([ffmpeg, "-y", "-i", str(working), "-an", "-c:v", "libx264", "-preset", "fast",
                        "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)],
                       stdout=log, stderr=log, check=True)
    cap = cv2.VideoCapture(str(target))
    actual_fps = cap.get(5)
    size = (int(cap.get(3)), int(cap.get(4)))
    count = 0
    while cap.read()[0]:
        count += 1
    cap.release()
    if count != frames or size != shape or abs(actual_fps-fps) > .01:
        raise RuntimeError("Comparison video failed full decode/frame-count validation")
    working.unlink()


def replay(config, tracking, cache, metadata, output, status):
    from ultralytics.engine.results import Boxes

    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    cap, writer = cv2.VideoCapture(config.source), None
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = metadata["fps"]
    tracker = TrackingSession(tracking, fps)
    counter, zones = analytics_tools(config, width, height)
    grace = max(config.missing_grace_seconds, (tracking.lost_buffer_frames+1)/fps)
    trails = defaultdict(lambda: deque(maxlen=config.trajectory_points))
    sightings = Counter()
    events, visits, timeline, diagnostics, tracker_ms = [], [], [], [], []
    visit_starts = {}
    entries = exits = 0
    index = 0
    gaps_total = 0
    def zone_events(rows):
        for row in rows:
            data = asdict(row)
            events.append(data)
            key = row.track_id, row.zone_name
            if row.event_type == "zone_entry":
                visit_starts[key] = row.timestamp
            else:
                visits.append({**data, "entered_at": visit_starts.pop(key), "exited_at": row.timestamp})
    try:
        working = output/"annotated.working.mp4"
        writer = cv2.VideoWriter(str(working), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("Comparison video writer unavailable")
        with (output/"tracks.jsonl").open("w", encoding="utf-8") as log:
            for index, timestamp in enumerate(cache["timestamps"]):
                ok, image = cap.read()
                if not ok:
                    raise RuntimeError("Replay input ended before cached detections")
                begin, end = cache["offsets"][index:index+2]
                boxes = Boxes(cache["boxes"][begin:end].copy(), (height, width))
                call = time.perf_counter()
                rows = tracker.update(boxes, image, cache["features"][begin:end].copy() if tracking.mode == "botsort_reid" else None)
                tracker_ms.append((time.perf_counter()-call)*1000)
                diag = {"frame_index": index, "timestamp": float(timestamp), **tracker.diagnostics.last}
                diagnostics.append(diag)
                ids = {int(row[4]) for row in rows}
                gaps_total += sum(tracker.identity_gaps.values())
                detections = []
                for row in rows:
                    identity = int(row[4])
                    point = BBox(*row[:4]).bottom_center
                    gap = tracker.identity_gaps.get(identity, 0)
                    if gap:
                        zones.mark_recovered_gap(identity)
                    crossing = counter.update(identity, point, float(timestamp))
                    if crossing:
                        entries += crossing.event_type == "entry"
                        exits += crossing.event_type == "exit"
                        events.append({**asdict(crossing), "provisional": not config.counting_line.reviewed or bool(gap),
                                       "censored": bool(gap), "reason": "crossing_across_unobserved_gap" if gap else "observed_crossing"})
                    zone_events(zones.update(identity, point, float(timestamp)))
                    sightings[identity] += 1
                    trails[identity].append((int(point.x), int(point.y)))
                    detections.append({"track_id": identity, "bbox": row[:4].tolist(), "confidence": float(row[5]),
                                       "detection_index": int(row[-1]), "recovered_gap_frames": gap})
                    color = ((identity*43)%160+90, (identity*71)%160+90, (identity*97)%160+90)
                    x1, y1, x2, y2 = map(int, row[:4])
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(image, f"ID {identity}"+(" recovered" if gap else ""), (x1, max(15, y1-4)), 0, .5, color, 2)
                    if len(trails[identity]) > 1:
                        cv2.polylines(image, [np.array(trails[identity], np.int32)], False, color, 1)
                zone_events(zones.expire_missing(ids, float(timestamp), grace))
                zone_counts = zones.occupancy(ids)
                timeline.append({"frame_index": index, "timestamp": float(timestamp), "visible_people": len(ids), "zones": zone_counts})
                for zone in zones.zones:
                    cv2.polylines(image, [np.array([(p.x, p.y) for p in zone.polygon], np.int32)], True, (80, 190, 100), 2)
                line = counter.config.line
                cv2.line(image, (int(line.start.x), int(line.start.y)), (int(line.end.x), int(line.end.y)), (0, 210, 255), 2)
                cv2.rectangle(image, (int(width*.5), 0), (width, 65), (16, 20, 22), -1)
                cv2.putText(image, f"{output.name} | In {entries} Out {exits} PROVISIONAL", (int(width*.5)+8, 20), 0, .44, (255, 255, 255), 1)
                cv2.putText(image, f"Active {diag['active']} Lost {diag['lost']} Recovered {diag['recovered']} Removed {diag['removed']}",
                            (int(width*.5)+8, 43), 0, .43, (90, 230, 160), 1)
                writer.write(image)
                if index in {0, len(cache["timestamps"])//2, len(cache["timestamps"])-1}:
                    cv2.imwrite(str(output/f"frame_{index:06d}.jpg"), image)
                log.write(json.dumps({"frame_index": index, "timestamp": float(timestamp), "detections": detections})+"\n")
                if index % 100 == 0:
                    status(phase="tracking", variant=output.name, processed_frames=index+1, total_frames=metadata["frames"], diagnostics=diag)
    finally:
        cap.release()
        if writer:
            writer.release()
    zone_events(zones.close_open_visits(float(cache["timestamps"][-1])))
    replay_seconds = time.perf_counter()-started
    encode_and_check(working, output/"annotated.mp4", fps, metadata["frames"], (width, height))
    end_to_end = time.perf_counter()-started
    events.sort(key=lambda row: row["timestamp"])
    write_json(output/"events.json", {"events": events, "timestamp_unit": "video_seconds"})
    fields = ["timestamp", "track_id", "event_type", "zone_name", "line_name", "dwell_seconds", "censored", "left_censored", "reason", "provisional"]
    with (output/"events.csv").open("w", newline="", encoding="utf-8") as stream:
        csv_writer = csv.DictWriter(stream, fieldnames=fields)
        csv_writer.writeheader()
        csv_writer.writerows(events)
    with (output/"diagnostics.csv").open("w", newline="", encoding="utf-8") as stream:
        csv_writer = csv.DictWriter(stream, fieldnames=list(diagnostics[0]))
        csv_writer.writeheader()
        csv_writer.writerows(diagnostics)
    write_json(output/"zone_visits.json", visits)
    write_json(output/"occupancy.json", timeline)
    summary = {"variant": output.name, "tracking": tracking.model_dump(), "buffer_seconds": tracking.lost_buffer_frames/fps,
               "processing_fps": metadata["frames"]/replay_seconds, "replay_seconds": replay_seconds,
               "end_to_end_replay_fps": metadata["frames"]/end_to_end,
               "detector_charged_fps_estimate": metadata["frames"]/(replay_seconds+float(cache["predict_ms"].sum())/1000),
               "tracking_ms": distribution(tracker_ms), "detector_inference_ms": metadata["inference_ms"],
               "generated_ids": len(sightings), "one_or_two_observation_tracks": sum(count <= 2 for count in sightings.values()),
               "entries": entries, "exits": exits,
               "line_counts_provisional": not config.counting_line.reviewed or any(event.get("provisional") for event in events if event["event_type"] in {"entry", "exit"}),
               "zone_visits": len(visits), "censored_zone_visits": sum(visit["censored"] for visit in visits),
               "last_diagnostics": diagnostics[-1], "peak_active": max(row["active"] for row in diagnostics),
               "recovery_rejections": dict(getattr(tracker.engine, "rejections", {})), "unobserved_recovered_frames": gaps_total,
               "identity_metrics": None, "identity_metrics_unavailable_reason": "No annotated ground-truth identities supplied",
               "video": "annotated.mp4", "video_fully_decoded": True, "frames": len(diagnostics)}
    write_json(output/"summary.json", summary)
    return summary


def evaluate_identities(ground_truth, output, frames):
    """Optional explicit complete-annotation CSV, evaluated with motmetrics, IoU>=.5."""
    import motmetrics as mm

    truth = defaultdict(list)
    with Path(ground_truth).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            index, identity = int(row["frame_index"]), int(row["person_id"])
            if not 0 <= index < frames:
                raise ValueError("Ground-truth frame outside processed video")
            box = [float(row[key]) for key in ["x1", "y1", "x2", "y2"]]
            if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("Invalid ground-truth bounding box")
            truth[index].append((identity, box))
    if not truth:
        raise ValueError("No annotated ground-truth identities found")
    from visionops.vision.appearance import iou
    accumulator = mm.MOTAccumulator(auto_id=True)
    with (output/"tracks.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            targets, predicted = truth[row["frame_index"]], row["detections"]
            if len({item[0] for item in targets}) != len(targets):
                raise ValueError("Duplicate ground-truth identity in a frame")
            distances = np.array([[1-iou(target[1], detection["bbox"]) for detection in predicted] for target in targets], float).reshape(len(targets), len(predicted))
            distances[distances > .5] = np.nan
            accumulator.update([target[0] for target in targets], [detection["track_id"] for detection in predicted], distances)
    metrics = mm.metrics.create().compute(accumulator, metrics=["num_switches", "num_fragmentations", "idf1", "mota", "num_misses", "num_false_positives"], name="video")
    return {key: float(value) if np.isfinite(value) else None for key, value in metrics.loc["video"].items()}


def write_comparison_report(output, comparison):
    metadata = comparison["detector"]
    lines = ["# VisionOps Tracking Experiment", "", f"Run: {comparison['run_id']}", "",
             f"Same video: `{metadata['source']}` ({metadata['frames']} frames, {metadata['fps']:.6f} FPS, stride 1).",
             f"Detector: `{metadata['checkpoint']}` SHA256 `{metadata['checkpoint_sha256']}`; no weight fallback.",
             f"Thresholds: confidence {metadata['confidence']}, NMS IoU {metadata['iou']}, input {metadata['imgsz']}.",
             f"Ultralytics {metadata['ultralytics']}, device {metadata['device']}. Detection cache SHA256 `{metadata['cache_sha256']}`.",
             f"Embedding source: {metadata['embedding_note']}. Dimensions: {metadata['embedding_dimensions']}.", "",
             "## Controlled Results", "",
             "All variants replay the identical cached boxes and embeddings. Buffer counts are tracker updates, not 30-FPS-normalized seconds.",
             "Processing FPS measures actual decode/tracking/overlay/video-writer replay, excluding detector and final H.264 encoding.",
             "Charged FPS is an estimate adding the measured shared prediction time to replay, not a separately timed online run.",
             "Tracker latency includes feature galleries and GMC. Native feature extraction overhead is charged equally to every variant.", "",
             "| Variant | Buffer seconds | IDs (not people) | Recovered gaps | Tracker mean/p95 ms | Replay FPS | Charged FPS estimate | In / Out |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in comparison["results"]:
        lines.append(f"| {row['variant']} | {row['buffer_seconds']:.3f} | {row['generated_ids']} | {row['last_diagnostics']['recovered']} | {row['tracking_ms']['mean']:.2f} / {row['tracking_ms']['p95']:.2f} | {row['processing_fps']:.2f} | {row['detector_charged_fps_estimate']:.2f} | {row['entries']} / {row['exits']} |")
    lines += ["", "## Identity Evaluation", "",
              "Generated ID count, short tracks, and within-ID observation gaps are diagnostics, not measured identity fragmentation or ID switches.",
              "Fewer IDs are not proof of better tracking: false merges can also reduce IDs. No variant is promoted automatically.",
              "Ground-truth identity metrics: " + ("measured with motmetrics; see each summary.json" if comparison.get("ground_truth") else "UNAVAILABLE: no annotated identities supplied"), "",
              "## Recovery And Analytics", "",
              "Ultralytics supplies Kalman motion, camera-motion compensation, low-confidence rescue, and one-to-one LAP assignment.",
              "The opt-in assisted path requires reliable recent galleries, deep embedding and HSV agreement, predicted-box spatial/size gates, "
              "Mahalanobis distance <=9.4877, and a unique candidate with row/column ambiguity margins. Clothing color alone cannot recover an ID.",
              "Low-confidence, clipped, small/partial, blurred and strongly overlapping crops do not update galleries. Lost/removed IDs are never aliased after processing.",
              "The original engine ID feeds line counters, zone membership, dwell and trajectories. Zone grace covers the configured recovery buffer.",
              "Recovered gaps censor affected dwell spans. Crossings interpolated across an unobserved gap are provisional and censored; no hidden boxes or exact hidden paths are reconstructed.",
              "The existing line counter allows at most one event per ID per direction. It suppresses jitter but can suppress legitimate repeat visits; fragmented IDs can inflate counts.",
              "Counts reflect unreviewed candidate geometry, not confirmed entrance traffic. Initial occupancy is unknown.", "",
              "## Limitations And Compatibility", "",
              "The adapter fixes the installed 8.4.7 BOTrack feature-initialization order without modifying site-packages. Native features are detector features, not a person-ReID-trained model.",
              "No suitable dedicated person-ReID checkpoint is installed. Optional local ONNX models require RGB 256x128 ImageNet-normalized input and NxD embeddings; model provenance and domain fit remain operator responsibilities.",
              "Appearance may fail under illumination changes, severe occlusion, look-alike clothing and partial bodies; conservative rejection can increase fragmentation.",
              "GMC failure falls back to identity warp and is counted. Synthetic camera-motion tests do not establish performance on a moving-camera benchmark.",
              "Active/low-confidence association still uses upstream motion-based stages; ambiguous switches can remain. Buffer duration is not evidence of a person's identity.",
              "Reconnection cannot reconstruct exact occupancy, dwell, or line movement during occlusion. Ground-truth accuracy and counting accuracy are not claimed.", ""]
    (output/"evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_tracking_comparison(config, ground_truth=None, detection_cache=None):
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT.parent/"run.lock").exists():
        raise RuntimeError("Wait for the retail pipeline to finish before comparing trackers")
    lock = ROOT/"experiment.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(descriptor)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT/run_id
    output.mkdir()
    comparison = {"run_id": run_id, "relative_dir": run_id, "configuration": config.model_dump(), "results": [], "ground_truth": ground_truth,
                  "ground_truth_sha256": None}
    state = {"state": "running", "run_id": run_id, "relative_dir": run_id}
    def status(**values):
        state.update(values)
        write_json(ROOT/"status.json", state)
    try:
        if ground_truth:
            import motmetrics
            comparison["ground_truth_sha256"] = digest(Path(ground_truth))
        write_json(ROOT/"latest.json", {"relative_dir": run_id})
        status(phase="initializing")
        if detection_cache:
            folder = Path(detection_cache)
            metadata = json.loads((folder/"detection_metadata.json").read_text(encoding="utf-8"))
            expected = {"source_sha256": digest(Path(config.source)), "checkpoint_sha256": digest(Path(config.model_path)),
                        "confidence": config.confidence, "iou": config.iou, "imgsz": config.imgsz,
                        "embedding_backend": config.tracking.embedding_backend,
                        "embedding_model_sha256": digest(Path(config.tracking.embedding_model)) if config.tracking.embedding_backend == "onnx" else None}
            if any(metadata.get(key) != value for key, value in expected.items()) or digest(folder/"detections.npz") != metadata["cache_sha256"]:
                raise ValueError("Detection cache differs from the requested video, model, thresholds, or encoder")
            shutil.copy2(folder/"detections.npz", output/"detections.npz")
            shutil.copy2(folder/"detection_metadata.json", output/"detection_metadata.json")
            with np.load(output/"detections.npz", allow_pickle=False) as saved:
                cache = {key: saved[key] for key in saved.files}
        else:
            cache, metadata = collect_detections(config, output, status)
        comparison["detector"] = metadata
        variants = [(mode, buffer, True) for mode in ["bytetrack", "botsort", "botsort_reid"] for buffer in [10, 30, 60]]
        variants.append(("botsort_reid", 30, False))
        for number, (mode, buffer, assisted) in enumerate(variants):
            name = f"{mode}_{buffer}" + ("_native_association" if not assisted else "")
            tracking = config.tracking.model_copy(update={"mode": mode, "lost_buffer_frames": buffer, "recovery_enabled": assisted})
            status(phase="tracking", variant=name, variant_index=number+1, variant_count=len(variants), processed_frames=0)
            summary = replay(config, tracking, cache, metadata, output/name, status)
            if ground_truth:
                summary["identity_metrics"] = evaluate_identities(ground_truth, output/name, metadata["frames"])
                summary["identity_metrics_unavailable_reason"] = None
                write_json(output/name/"summary.json", summary)
            comparison["results"].append(summary)
            write_json(output/"comparison.json", comparison)
            print(f"Finished {name}: IDs={summary['generated_ids']} recovered={summary['last_diagnostics']['recovered']}", flush=True)
        write_comparison_report(output, comparison)
        status(state="completed", phase="completed", processed_frames=metadata["frames"],
               diagnostics=comparison["results"][-1]["last_diagnostics"])
        return comparison
    except Exception as exc:
        status(state="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        lock.unlink(missing_ok=True)
