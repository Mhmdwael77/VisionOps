"""Full-frame retail test runner with auditable artifacts and video-time analytics."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

import cv2
import numpy as np

from visionops.vision.counting import CountingLineConfig, CrossingDirection, LineCrossingCounter
from visionops.vision.geometry import BBox, LineSegment, Point
from visionops.vision.overlay import draw_track
from visionops.vision.test_config import RetailTestConfig
from visionops.vision.zones import ZoneConfig, ZoneMonitor
from visionops.vision.appearance import NativeFeatures, ONNXFeatures
from visionops.vision.tracking import TrackingSession, TrackerDiagnostics, apply_tracks, result_embeddings


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def overlap(a, b) -> float:
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def distribution(values: list[float]) -> dict:
    return {"mean": float(np.mean(values)), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "max": float(max(values))} if values else {}


def write_report(output: Path, summary: dict) -> None:
    manual_path = output / "manual_review.json"
    manual = json.loads(manual_path.read_text(encoding="utf-8")) if manual_path.exists() else {"observations": []}
    model, performance = summary["model"], summary["performance"]
    lines = ["# VisionOps Retail Store Test", "", f"Run ID: {summary['run_id']}", "",
             f"Source: `{summary['source']['path']}`. {summary['source']['decoded_frames']} frames; "
             f"{summary['source']['duration_seconds']:.3f} seconds; {summary['source']['width']} x {summary['source']['height']}.", "",
             f"Model: **{model['kind']}**, `{model['path']}`. Fallback used: **{model['fallback_used']}**.",
             f"Model selection note: {model['selection_note']}",
             f"Tracker: {summary.get('tracker', {}).get('label', 'ByteTrack')}. "
             "Tracker configuration and diagnostics are recorded in summary.json; fewer IDs do not prove better tracking.",
             f"Checkpoint SHA256: `{model['sha256']}`.", "",
             "## Measurements", "",
             f"- Processing loop: {performance['processing_fps']:.2f} FPS ({performance['processing_seconds']:.3f}s).",
             f"- End to end (model load, decoding, tracking, overlays, encoding, verification): {performance['end_to_end_fps']:.2f} FPS.",
             f"- YOLO inference latency: mean {performance['inference_ms']['mean']:.2f} ms; p95 {performance['inference_ms']['p95']:.2f} ms.",
             f"- Whole tracking call: mean {performance['tracking_call_ms']['mean']:.2f} ms; p95 {performance['tracking_call_ms']['p95']:.2f} ms.",
             f"- Unique tracker IDs: {summary['analytics']['unique_track_ids']}. These are not unique people.",
             f"- Provisional line events: {summary['analytics']['entries']} entries / {summary['analytics']['exits']} exits.", "",
             "## Geometry and Interpretation", "",
             f"Entrance line reviewed: **{summary['configuration']['counting_line']['reviewed']}**. "
             "The candidate line crosses the approach to the lower-right doorway; its physical location and direction need human review. "
             "Counts must not be presented as confirmed entrance traffic.", "",
             f"Zones reviewed: **{summary['configuration']['zones_reviewed']}**. Polygon coordinates are normalized to source dimensions. "
             "Membership uses bounding-box bottom centers. Occluded feet and perspective can put a person in the wrong zone.", "",
             "Zone occupancy counts currently observed tracked IDs inside each polygon. Dwell uses video presentation timestamps, "
             "never processing wall time. Lost tracks close at last observation after a configurable grace period. "
             "Track-loss and end-of-video visits are censored, not observed physical departures. "
             "A track first observed already inside a zone has an unobserved entry and is left-censored. "
             "Observed visible people and provisional line balance are different measures; initial store occupancy is unknown.", "",
             "Counting protection: finite line segment, a distance dead band, and at most one event per ID per direction. "
             "This protects against jitter but suppresses legitimate repeated same-direction visits by the same ID; "
             "an ID switch can still produce duplicate counts. Unobserved crossings cannot be recovered.", "",
             "## Verification", ""]
    for name, value in summary["verification"].items():
        lines.append(f"- {name}: {value}")
    lines += ["", "## Detected Issues and Review Candidates", "",
              "Automated diagnostics are candidates, not measured tracking errors:"]
    for name, count in summary["diagnostics"]["candidate_counts"].items():
        lines.append(f"- {name}: {count} candidate frame occurrences.")
    lines += ["", "Manual visual review (sampled frames only):"]
    if manual.get("observations") and manual.get("run_id") == summary["run_id"]:
        for observation in manual["observations"]:
            lines.append(f"- {observation['category']} at {observation['timestamp']:.3f}s: {observation['finding']} "
                         f"Evidence: `{observation['evidence']}`. Status: {observation['status']}.")
    else:
        lines.append("- No manual review recorded for this run yet. Missed people, occlusion, ID switches and counting errors remain unquantified.")
    lines += ["", "## Evaluation Limits", "",
              "No manually annotated ground truth was supplied. No precision, recall, mAP, IDF1, ID-switch rate, "
              "counting accuracy or dwell accuracy is asserted. Frame review can demonstrate specific failures but is not a full benchmark. "
              "`review_candidates.json` and evidence frames support further review. Audio is not included in the annotated output.", "",
              "## Reproduce", "", "```powershell", "python scripts/run_retail_test.py", "```", "",
              "Edit `configs/retail_store_test.json` or use the dashboard geometry editor before rerunning. "
              "All video, logs, timing data, evidence, effective configuration and reports are in this directory."]
    (output / "test_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_retail_test(config: RetailTestConfig, on_progress=None) -> dict:
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "tracking_experiment/experiment.lock").exists():
        raise RuntimeError("Wait for the controlled tracking experiment to finish")
    lock_path = output / "run.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("A retail test is already running (run.lock exists)") from exc
    os.close(lock_fd)
    started = time.perf_counter()
    run_id = datetime.now(timezone.utc).isoformat()
    cap, writer, native_features = None, None, None
    status = {"state": "running", "run_id": run_id, "processed_frames": 0, "progress": 0.0}
    write_json(output / "status.json", status)
    try:
        if on_progress:
            on_progress(status)
        import torch
        import ultralytics
        from ultralytics import YOLO
        from ultralytics.utils import YAML
        from ultralytics.utils.checks import check_yaml

        source = Path(config.source)
        if not source.is_file():
            raise FileNotFoundError(f"Video not found: {source}")
        model_path = Path(config.model_path)
        fallback_used = not model_path.is_file()
        selection_note = "Local fine-tuned checkpoint loaded successfully."
        if fallback_used:
            model_path = Path(config.fallback_model)
            if not model_path.exists():
                model_path = output / model_path.name
            selection_note = "Fine-tuned checkpoint missing; explicitly using pretrained fallback."
        try:
            model = YOLO(str(model_path))
        except Exception as exc:
            if fallback_used:
                raise
            selection_note = f"Fine-tuned checkpoint failed to load ({type(exc).__name__}: {exc}); using pretrained fallback."
            fallback_used = True
            model_path = Path(config.fallback_model)
            if not model_path.exists():
                model_path = output / model_path.name
            model = YOLO(str(model_path))
        names = model.names
        person_ids = [key for key, value in names.items() if str(value).lower() in {"person", "people"}]
        if not person_ids:
            raise ValueError(f"Checkpoint does not expose a person class: {names}")
        model_path = Path(model.ckpt_path)
        device = config.device or ("0" if torch.cuda.is_available() else "cpu")
        use_cuda = str(device) != "cpu" and torch.cuda.is_available()
        tracker_path = check_yaml("bytetrack.yaml")
        tracker_config = YAML.load(tracker_path)
        write_json(output / "tracker_config.json", tracker_config)
        write_json(output / "effective_config.json", config.model_dump())
        print(f"MODEL: {'pretrained fallback' if fallback_used else 'fine-tuned'} {model_path}; device={device}", flush=True)

        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise RuntimeError("OpenCV could not open the input video")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise ValueError("Source has no valid FPS; refusing to fabricate video timestamps")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(3)), int(cap.get(4))
        custom_tracking = not config.tracking.uses_upstream_baseline()
        session = TrackingSession(config.tracking, fps) if custom_tracking else None
        baseline_diagnostics = TrackerDiagnostics()
        encoder = None
        if config.tracking.mode == "botsort_reid":
            if config.tracking.embedding_backend == "native":
                native_features = NativeFeatures(model)
            else:
                encoder = ONNXFeatures(config.tracking.embedding_model)
        if session:
            tracker_config = session.arguments
            write_json(output / "tracker_config.json", tracker_config)
        tracker_label = {"bytetrack": "ByteTrack", "botsort": "BoT-SORT", "botsort_reid": "BoT-SORT + appearance"}[config.tracking.mode]
        zone_grace = max(config.missing_grace_seconds, (config.tracking.lost_buffer_frames+1)/fps) if custom_tracking else config.missing_grace_seconds
        working = output / "annotated.working.mp4"
        writer = cv2.VideoWriter(str(working), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("Could not initialize annotated MP4 writer")
        def point(pair):
            return Point(pair[0] * width, pair[1] * height)
        line = config.counting_line
        counter = LineCrossingCounter(CountingLineConfig(line.name, LineSegment(point(line.start), point(line.end)),
                                                       CrossingDirection(line.entry_direction), line.min_displacement))
        zones = ZoneMonitor([ZoneConfig(zone.name, [point(p) for p in zone.polygon]) for zone in config.zones])
        trails = defaultdict(lambda: deque(maxlen=config.trajectory_points))
        last_observed: dict[int, tuple[float, float, float]] = {}
        unique_ids: set[int] = set()
        track_first_seen, track_last_seen = {}, {}
        entries = exits = 0
        events, visits, candidates, evidence, occupancy_rows = [], [], [], [], []
        candidate_counts, saved_counts = Counter(), Counter()
        visit_starts = {}
        inference_ms, tracking_ms = [], []
        timestamp_fallbacks = 0
        frames = 0
        previous_timestamp = -1.0
        evidence_dir = output / "frames"
        evidence_dir.mkdir(exist_ok=True)
        sample_indices = set(np.linspace(0, max(total - 1, 0), 12).astype(int).tolist())
        loop_start = time.perf_counter()

        def add_zone_events(zone_events):
            for event in zone_events:
                data = asdict(event)
                data["frame_index"] = frames
                events.append(data)
                key = (event.track_id, event.zone_name)
                if event.event_type == "zone_entry":
                    visit_starts[key] = event.timestamp
                else:
                    entered = visit_starts.pop(key, event.timestamp)
                    visits.append({**data, "entered_at": entered, "exited_at": event.timestamp})

        resolver = None
        if config.tracking.reid_enabled or config.tracking.min_track_seconds > 0:
            from visionops.vision.identity import IdentityResolver, OSNetEncoder
            reid_encoder = None
            if config.tracking.reid_enabled:
                if not Path(config.tracking.reid_model).is_file():
                    raise FileNotFoundError(f"Person-ReID weights not found: {config.tracking.reid_model}")
                reid_encoder = OSNetEncoder(config.tracking.reid_model, device="cuda" if use_cuda else "cpu")
            resolver = IdentityResolver(config.tracking, reid_encoder)
        # Pass 1: detection and tracking. Identity resolution needs the whole video,
        # so analytics and rendering happen in pass 2 on the resolved IDs.
        tracked = []
        while True:
            ok, original = cap.read()
            if not ok:
                break
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if timestamp <= previous_timestamp:
                timestamp = frames / fps
                timestamp_fallbacks += 1
            if timestamp <= previous_timestamp:
                raise RuntimeError("Source presentation timestamps are not monotonic")
            previous_timestamp = timestamp
            if use_cuda:
                torch.cuda.synchronize()
            call_start = time.perf_counter()
            if session:
                result = model.predict(original, classes=person_ids, conf=config.confidence, iou=config.iou,
                                       imgsz=config.imgsz, device=device, verbose=False)[0]
                embeddings = result_embeddings(result, encoder) if config.tracking.mode == "botsort_reid" else None
                rows = session.update(result.boxes.cpu().numpy(), original, embeddings)
                result = apply_tracks(result, rows)
                identity_gaps = session.identity_gaps
                tracker_diagnostics = session.diagnostics.last
            else:
                result = model.track(original, persist=True, tracker="bytetrack.yaml", classes=person_ids,
                                     conf=config.confidence, iou=config.iou, imgsz=config.imgsz, device=device,
                                     verbose=False)[0]
                observed_ids = set(map(int, result.boxes.id.cpu().tolist())) if result.boxes.id is not None else set()
                identity_gaps = baseline_diagnostics.update(model.predictor.trackers[0], observed_ids, frames+1)
                tracker_diagnostics = baseline_diagnostics.last
            if use_cuda:
                torch.cuda.synchronize()
            tracking_ms.append((time.perf_counter() - call_start) * 1000)
            inference_ms.append(float(result.speed["inference"]))
            raw = [{"track_id": int(box.id.item()) if box.id is not None else None,
                    "confidence": float(box.conf.item()), "bbox": [float(v) for v in box.xyxy[0].cpu().tolist()]}
                   for box in (result.boxes if result.boxes is not None else [])]
            if resolver:
                resolver.observe(frames, timestamp, original, raw)
            tracked.append((timestamp, raw, set(identity_gaps), dict(tracker_diagnostics)))
            frames += 1
            if frames % 100 == 0:
                status.update(processed_frames=frames, total_frames=total, progress=min(0.6*frames/max(total, 1), 0.6),
                              phase="tracking", tracker_diagnostics=tracker_diagnostics)
                write_json(output / "status.json", status)
                if on_progress:
                    on_progress(status)
                print(f"Tracked {frames}/{total} | {frames/(time.perf_counter()-loop_start):.1f} FPS", flush=True)
        if frames == 0 or (total and frames != total):
            raise RuntimeError(f"Decode incomplete: {frames} frames read, {total} expected")
        identity = resolver.resolve() if resolver else None
        if identity:
            write_json(output / "identity_resolution.json", identity.report)
            print(f"Identities: {identity.report['raw_track_ids']} raw tracker IDs -> {identity.report['identities']} "
                  f"({identity.report['switch_splits']} switch splits, {identity.report['stitch_merges']} merges, "
                  f"{identity.report['dropped_short_identities']} short dropped)", flush=True)
        cap.release()
        cap = cv2.VideoCapture(str(source))
        frames = 0
        gap_aware = custom_tracking or identity is not None
        identity_last_frame = {}

        # Pass 2: analytics, overlays and exports on resolved identities.
        with (output / "tracks.jsonl").open("w", encoding="utf-8") as track_log:
            for timestamp, raw, identity_gaps, tracker_diagnostics in tracked:
                ok, original = cap.read()
                if not ok:
                    raise RuntimeError("Second decode pass ended before the tracked frames")
                if identity is not None:
                    resolved = []
                    for item in raw:
                        mapped = identity(item["track_id"], frames) if item["track_id"] is not None else None
                        if item["track_id"] is not None and mapped is None:
                            continue  # identity observed too briefly
                        resolved.append({**item, "track_id": mapped, "raw_track_id": item["track_id"]})
                    # Stitched identities may briefly overlap; keep the most confident box per identity.
                    best = {}
                    for item in resolved:
                        if item["track_id"] is not None and item["confidence"] > best.get(item["track_id"], {"confidence": -1})["confidence"]:
                            best[item["track_id"]] = item
                    raw = [item for item in resolved if item["track_id"] is None or best[item["track_id"]] is item]
                    seen = {item["track_id"] for item in raw if item["track_id"] is not None}
                    identity_gaps = {i for i in seen if i in identity_last_frame and frames-identity_last_frame[i] > 1}
                    identity_last_frame.update({i: frames for i in seen})
                detections, flags = [], []
                active_ids = set()
                for item in raw:
                    xyxy, confidence, track_id = item["bbox"], item["confidence"], item["track_id"]
                    detections.append(item)
                    if confidence < 0.4:
                        flags.append({"type": "low_confidence", "track_ids": [track_id]})
                    if track_id is None:
                        flags.append({"type": "unassigned_detection", "track_ids": []})
                        continue
                    active_ids.add(track_id)
                    if gap_aware and track_id in identity_gaps:
                        zones.mark_recovered_gap(track_id)
                    anchor = BBox(*xyxy).bottom_center
                    if track_id not in unique_ids:
                        track_first_seen[track_id] = timestamp
                        for old_id, (old_time, old_x, old_y) in last_observed.items():
                            if old_id not in active_ids and 0 < timestamp - old_time < 1.5 and np.hypot(anchor.x-old_x, anchor.y-old_y) < 70:
                                flags.append({"type": "possible_id_switch", "track_ids": [old_id, track_id]})
                                break
                    elif timestamp - last_observed[track_id][0] > 2.0 / fps:
                        flags.append({"type": "tracking_gap", "track_ids": [track_id],
                                      "gap_seconds": timestamp - last_observed[track_id][0]})
                    unique_ids.add(track_id)
                    track_last_seen[track_id] = timestamp
                    last_observed[track_id] = (timestamp, anchor.x, anchor.y)
                    trails[track_id].append((int(anchor.x), int(anchor.y)))
                    crossing = counter.update(track_id, anchor, timestamp)
                    if crossing:
                        entries += crossing.event_type == "entry"
                        exits += crossing.event_type == "exit"
                        across_gap = gap_aware and track_id in identity_gaps
                        events.append({**asdict(crossing), "frame_index": frames, "provisional": not line.reviewed or across_gap,
                                       "reason": "crossing_across_unobserved_gap" if across_gap else "observed_crossing", "censored": across_gap})
                    add_zone_events(zones.update(track_id, anchor, timestamp))
                add_zone_events(zones.expire_missing(active_ids, timestamp, zone_grace))
                for i, a in enumerate(detections):
                    for b in detections[i+1:]:
                        if overlap(a["bbox"], b["bbox"]) > 0.20:
                            flags.append({"type": "occlusion_overlap", "track_ids": [a["track_id"], b["track_id"]]})
                if not detections:
                    flags.append({"type": "no_detections", "track_ids": []})
                types = sorted({flag["type"] for flag in flags})
                candidate_counts.update(types)
                for flag in flags:
                    candidates.append({"frame_index": frames, "timestamp": timestamp, **flag, "confirmed": False})
                zone_counts = zones.occupancy(active_ids)
                occupancy_rows.append({"frame_index": frames, "timestamp": timestamp, "visible_people": len(active_ids),
                                       "entries": entries, "exits": exits, "provisional_line_balance": entries-exits,
                                       "zones": zone_counts})
                track_log.write(json.dumps({"frame_index": frames, "timestamp": timestamp, "detections": detections,
                                            "inference_ms": inference_ms[-1], "tracking_call_ms": tracking_ms[-1],
                                            "tracker_diagnostics": tracker_diagnostics}) + "\n")
                frame = original.copy()
                for index, zone in enumerate(zones.zones):
                    pts = np.array([(p.x, p.y) for p in zone.polygon], np.int32)
                    color = [(80, 210, 110), (220, 160, 70)][index % 2]
                    cv2.polylines(frame, [pts], True, color, 2)
                    cv2.putText(frame, f"{zone.name}: {zone_counts[zone.name]}", tuple(pts[0]), 0, 0.5, color, 2)
                start, end = counter.config.line.start, counter.config.line.end
                cv2.line(frame, (int(start.x), int(start.y)), (int(end.x), int(end.y)), (0, 210, 255), 3)
                cv2.putText(frame, "Entrance: " + ("reviewed" if line.reviewed else "UNREVIEWED"),
                            (int(start.x), int(start.y)-12), 0, 0.55, (0, 210, 255), 2)
                for detection in detections:
                    track_id = detection["track_id"]
                    color = draw_track(frame, detection["bbox"], track_id, detection["confidence"])
                    if track_id is not None and len(trails[track_id]) > 1:
                        cv2.polylines(frame, [np.array(trails[track_id], np.int32)], False, color, 2)
                # The upper-left cashier contains heads; keep telemetry over the right shelf signage.
                banner_x = int(width * 0.51)
                cv2.rectangle(frame, (banner_x, 0), (width, 55), (16, 19, 22), -1)
                cv2.putText(frame, f"t={timestamp:.2f}s  People={len(active_ids)}  In={entries} Out={exits} (provisional)",
                            (banner_x+8, 21), 0, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"{'PRETRAINED FALLBACK' if fallback_used else 'FINE-TUNED YOLO11s'} | {tracker_label} | zones {'reviewed' if config.zones_reviewed else 'UNREVIEWED'}",
                            (banner_x+8, 43), 0, 0.43, (100, 230, 200), 1)
                writer.write(frame)
                save_tags = [tag for tag in types if saved_counts[tag] < 3]
                if frames in sample_indices or save_tags:
                    stem = f"frame_{frames:06d}"
                    raw_file, annotated_file = f"frames/{stem}_source.jpg", f"frames/{stem}_annotated.jpg"
                    if not cv2.imwrite(str(output/raw_file), original) or not cv2.imwrite(str(output/annotated_file), frame):
                        raise RuntimeError("Failed to save evidence frames")
                    evidence.append({"frame_index": frames, "timestamp": timestamp, "source": raw_file,
                                     "annotated": annotated_file, "tags": types or ["routine_sample"],
                                     "tracks": sorted(active_ids)})
                    saved_counts.update(save_tags)
                frames += 1
                if frames % 100 == 0:
                    status.update(processed_frames=frames, total_frames=total, progress=min(0.6+0.39*frames/max(total,1), 0.99),
                                  phase="analytics", tracker_diagnostics=tracker_diagnostics)
                    write_json(output / "status.json", status)
                    if on_progress:
                        on_progress(status)
                    print(f"Processed {frames}/{total} | {frames/(time.perf_counter()-loop_start):.1f} FPS | IDs={len(unique_ids)}", flush=True)
        processing_seconds = time.perf_counter() - loop_start
        if frames != len(tracked):
            raise RuntimeError(f"Second pass decoded {frames} frames, {len(tracked)} tracked")
        add_zone_events(zones.close_open_visits(previous_timestamp))
        cap.release()
        cap = None
        writer.release()
        writer = None
        events.sort(key=lambda row: (row["timestamp"], row["frame_index"]))
        for index, event in enumerate(events):
            event["id"] = index + 1
        write_json(output / "events.json", {"run_id": run_id, "timestamp_unit": "video_seconds", "events": events})
        fields = ["id", "frame_index", "timestamp", "track_id", "event_type", "zone_name", "line_name",
                  "dwell_seconds", "reason", "censored", "left_censored", "provisional"]
        with (output / "events.csv").open("w", newline="", encoding="utf-8") as stream:
            csv_writer = csv.DictWriter(stream, fieldnames=fields)
            csv_writer.writeheader()
            csv_writer.writerows(events)
        write_json(output / "zone_visits.json", visits)
        write_json(output / "occupancy.json", occupancy_rows)
        write_json(output / "review_candidates.json", candidates)
        write_json(output / "evidence.json", evidence)
        with (output / "frame_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            fields = ["frame_index", "timestamp", "visible_people", "entries", "exits", "provisional_line_balance",
                      "inference_ms", "tracking_call_ms"] + [zone.name for zone in config.zones]
            csv_writer = csv.DictWriter(stream, fieldnames=fields)
            csv_writer.writeheader()
            for i, row in enumerate(occupancy_rows):
                csv_writer.writerow({**{key: value for key, value in row.items() if key != "zones"},
                                     **row["zones"], "inference_ms": inference_ms[i], "tracking_call_ms": tracking_ms[i]})
        write_json(output / "track_summary.json", [{"track_id": track_id, "first_seen": track_first_seen[track_id],
                                                    "last_seen": track_last_seen[track_id]} for track_id in sorted(unique_ids)])

        status.update(state="encoding", processed_frames=frames, progress=0.99)
        write_json(output / "status.json", status)
        if on_progress:
            on_progress(status)
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg is required for browser-compatible H.264 output; raw MP4 preserved")
        annotated = output / "annotated.mp4"
        with (output / "encoding.log").open("w", encoding="utf-8") as log:
            subprocess.run([ffmpeg, "-y", "-i", str(working), "-an", "-c:v", "libx264", "-preset", "fast",
                            "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(annotated)],
                           stdout=log, stderr=log, check=True)
        check_cap = cv2.VideoCapture(str(annotated))
        output_fps, output_width, output_height = check_cap.get(5), int(check_cap.get(3)), int(check_cap.get(4))
        output_frames = 0
        while check_cap.read()[0]:
            output_frames += 1
        check_cap.release()
        if output_frames != frames or (output_width, output_height) != (width, height) or abs(output_fps-fps) > 0.01:
            raise RuntimeError("Encoded output does not match source dimensions/frame count/FPS")
        working.unlink()

        zone_analytics = []
        for zone in config.zones:
            zone_visits = [v for v in visits if v["zone_name"] == zone.name]
            dwell = [v["dwell_seconds"] for v in zone_visits]
            observed_dwell = [v["dwell_seconds"] for v in zone_visits if not v["censored"]]
            zone_analytics.append({"name": zone.name, "unique_track_ids": len(zones.unique_visitors[zone.name]),
                                   "visits": len(zone_visits), "censored_visits": sum(v["censored"] for v in zone_visits),
                                   "average_dwell_seconds": float(np.mean(dwell)) if dwell else 0,
                                   "average_uncensored_dwell_seconds": float(np.mean(observed_dwell)) if observed_dwell else None,
                                   "peak_occupancy": max(row["zones"][zone.name] for row in occupancy_rows),
                                   "final_occupancy": occupancy_rows[-1]["zones"][zone.name]})
        end_to_end_seconds = time.perf_counter() - started
        summary = {"run_id": run_id, "state": "completed", "configuration": config.model_dump(),
                   "source": {"path": source.as_posix(), "sha256": digest(source), "fps": fps, "width": width, "height": height,
                              "decoded_frames": frames, "duration_seconds": frames/fps,
                              "timestamp_source": "OpenCV presentation timestamps; frame_index/FPS fallback if non-monotonic",
                              "timestamp_fallback_frames": timestamp_fallbacks},
                   "model": {"path": str(model_path), "sha256": digest(model_path), "fallback_used": fallback_used,
                             "kind": "pretrained YOLO11s" if fallback_used else "fine-tuned YOLO11s",
                             "selection_note": selection_note, "class_names": names,
                             "architecture": model.model.yaml.get("yaml_file", "unknown"),
                             "scale": model.model.yaml.get("scale"), "parameters": sum(p.numel() for p in model.model.parameters())},
                   "environment": {"python": platform.python_version(), "ultralytics": ultralytics.__version__,
                                   "opencv": cv2.__version__, "torch": torch.__version__, "device": str(device),
                                   "gpu": torch.cuda.get_device_name() if use_cuda else None},
                   "tracker": {"implementation": type(session.engine if session else model.predictor.trackers[0]).__name__,
                               "label": tracker_label, "config": tracker_config, "settings": config.tracking.model_dump(),
                               "diagnostics": tracker_diagnostics,
                               "buffer_seconds": config.tracking.lost_buffer_frames/fps,
                               "zone_loss_grace_seconds": zone_grace,
                               "embedding_note": "Native detector features, not person-ReID trained" if config.tracking.mode == "botsort_reid" and encoder is None else None},
                   "analytics": {"entries": entries, "exits": exits,
                                 "line_counts_provisional": not line.reviewed or any(event.get("provisional") for event in events if event["event_type"] in {"entry", "exit"}),
                                 "provisional_line_balance": entries-exits, "unique_track_ids": len(unique_ids),
                                 "peak_visible_people": max(r["visible_people"] for r in occupancy_rows),
                                 "final_visible_people": occupancy_rows[-1]["visible_people"], "event_count": len(events),
                                 "zones": zone_analytics},
                   "performance": {"processing_seconds": processing_seconds, "processing_fps": frames/processing_seconds,
                                   "end_to_end_seconds": end_to_end_seconds, "end_to_end_fps": frames/end_to_end_seconds,
                                   "inference_ms": distribution(inference_ms), "tracking_call_ms": distribution(tracking_ms),
                                   "steady_state_inference_ms": distribution(inference_ms[1:]),
                                   "note": "All-frame statistics include first-frame setup; steady-state excludes frame 0. End to end includes encoding and full output decode verification, excludes report/hash writing."},
                   "diagnostics": {"candidate_counts": dict(candidate_counts), "evidence_frames": len(evidence),
                                   "note": "Heuristics indicate review candidates, not confirmed misses, ID switches or accuracy."},
                   "verification": {"all_source_frames_processed": frames == total,
                                    "bytetrack_ids_observed": bool(unique_ids) if config.tracking.mode == "bytetrack" else None,
                                    "tracker_ids_observed": bool(unique_ids),
                                    "annotated_mp4_fully_decoded": output_frames == frames,
                                    "source_duration_preserved": abs(output_frames/output_fps - frames/fps) < 0.01,
                                    "csv_json_event_counts_match": True, "video_timestamp_dwell": True,
                                    "entrance_line_human_review": line.reviewed, "zones_human_review": config.zones_reviewed,
                                    "ground_truth_accuracy": "NOT MEASURED"},
                   "artifacts": {"video": "annotated.mp4", "events_csv": "events.csv", "events_json": "events.json",
                                 "report": "test_report.md", "metrics": "frame_metrics.csv", "evidence": "evidence.json"}}
        # Validate exports from disk, rather than assuming the write succeeded.
        with (output / "events.csv").open(newline="", encoding="utf-8") as stream:
            csv_rows = list(csv.DictReader(stream))
        summary["verification"]["csv_json_event_counts_match"] = len(csv_rows) == len(json.loads((output/"events.json").read_text())["events"])
        write_json(output / "summary.json", summary)
        write_report(output, summary)
        status.update(state="completed", processed_frames=frames, progress=1.0)
        write_json(output / "status.json", status)
        print(json.dumps({"state": "completed", "frames": frames, "events": len(events),
                          "fps": summary["performance"]["processing_fps"], "output": str(output)}), flush=True)
        return summary
    except Exception as exc:
        status.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(output / "status.json", status)
        raise
    finally:
        if native_features is not None:
            native_features.close()
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        lock_path.unlink(missing_ok=True)
