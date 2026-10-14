"""Render exact canonical boxes; smooth only observed, contiguous trajectories."""
from collections import defaultdict, deque
import json
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np

from backend.app.core.database import SessionLocal
from backend.app.models.tables import ReviewDataset, ReviewRender, ReviewRevision
from backend.app.services.review_store import OUTPUT, dump, sha256
from visionops.vision.review_analytics import derive, fingerprint, frame_annotations


def color(hex_value):
    return tuple(int(hex_value.lstrip("#")[i:i+2], 16) for i in (4, 2, 0))


def text(image, label, point, tone=(240, 244, 246), scale=.48):
    x, y = point
    (width, height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = max(0, min(x, image.shape[1]-width-10)), max(height+6, min(y, image.shape[0]-baseline-6))
    cv2.rectangle(image, (x, y-height-6), (x+width+9, y+baseline+3), (20, 23, 28), -1)
    cv2.putText(image, label, (x+4, y), cv2.FONT_HERSHEY_SIMPLEX, scale, tone, 1, cv2.LINE_AA)


def paint(frame, index, document, metadata, analytics, annotations, trails):
    height, width = frame.shape[:2]
    geometry = document["geometry"]
    overlay = frame.copy()
    for zone in geometry["zones"]:
        polygon = np.array([[round(x*width), round(y*height)] for x, y in zone["polygon"]], np.int32)
        tone = color(zone["color"])
        cv2.fillPoly(overlay, [polygon], tone)
        cv2.polylines(frame, [polygon], True, tone, 1, cv2.LINE_AA)
    frame[:] = cv2.addWeighted(overlay, .10, frame, .90, 0)
    line = geometry["counting_line"]
    p1, p2 = [tuple(round(v*size) for v, size in zip(pair, (width, height))) for pair in (line["start"], line["end"])]
    cv2.line(frame, p1, p2, (255, 210, 55), 2, cv2.LINE_AA)
    text(frame, "Entrance" if line["reviewed"] else "Entrance candidate / unreviewed", p1, (255, 220, 90), .4)
    visible = [a for a in annotations if a["visibility"] == "visible"]
    for a in visible:
        box = a["bbox"]
        identity = a["person_id"]
        tone = (192, 224, 58) if a["origin"] == "raw_model" else (139, 221, 255)
        x1, y1, x2, y2 = [round(v) for v in box]
        history = trails[identity]
        if history and history[-1][0] != index-1:
            history.clear()
        history.append((index, ((x1+x2)//2, y2)))
        if len(history) > 1:
            points = np.array([point for _, point in history], np.int32)
            cv2.polylines(frame, [points], False, tone, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), tone, 1, cv2.LINE_AA)
        label = document["identities"][identity]["label"]
        text(frame, f"{label} {'RAW' if a['origin'] == 'raw_model' else 'EDIT'}", (x1, y1-5), tone, .4)
    row = analytics["timeline"][index]
    state = document["frame_reviews"][index]["state"].upper()
    cv2.rectangle(frame, (0, 0), (width, 44), (19, 22, 27), -1)
    text(frame, f"VISIONOPS   {metadata['timestamps'][index]:06.2f}s   FRAME {index}   {state}", (12, 25), scale=.5)
    text(frame, f"Provisional IN {row['entries']}  OUT {row['exits']}   Visible {row['visible']}", (width-430, 25), (239, 221, 166), .47)
    offset = 20
    for zone in geometry["zones"]:
        text(frame, f"{zone['name']}: {row['zones'][zone['name']]} visible", (12, height-offset), color(zone["color"]), .45)
        offset += 29
    return {"frame_index": index, "timestamp": metadata["timestamps"][index], "review_state": state.lower(),
        "visible_annotations": [{"id": a["id"], "person_id": a["person_id"], "bbox": a["bbox"], "origin": a["origin"]} for a in visible],
        "metrics": row}


def render_revision(render_id):
    started = time.perf_counter()
    with SessionLocal() as db:
        job = db.get(ReviewRender, render_id)
        value = db.get(ReviewDataset, job.dataset_id)
        revision = db.get(ReviewRevision, (value.id, job.revision))
        source, source_hash = value.source_path, value.source_sha256
        document, metadata = json.loads(revision.document_json), json.loads(value.metadata_json)
        job.status = "running"
        db.commit()
        root = OUTPUT / value.id / f"revision_{job.revision}" / render_id
        root.mkdir(parents=True, exist_ok=True)
        capture = writer = None
        try:
            if sha256(source) != source_hash:
                raise ValueError("Original source video changed; refusing to render")
            analytics = derive(document, metadata)
            if fingerprint(analytics) != fingerprint(json.loads(revision.analytics_json)):
                raise ValueError("Persisted aggregates do not match this annotation revision")
            (root / "annotations.json").write_text(dump({"source_sha256": source_hash, "revision": job.revision,
                "document": document, "analytics": analytics}), encoding="utf-8")
            capture = cv2.VideoCapture(source)
            staging = root / "staging.mp4"
            writer = cv2.VideoWriter(str(staging), cv2.VideoWriter_fourcc(*"mp4v"), metadata["fps"], (metadata["width"], metadata["height"]))
            if not writer.isOpened():
                raise ValueError("Video writer unavailable")
            grouped = frame_annotations(document)
            trails = defaultdict(lambda: deque(maxlen=18))
            with (root / "overlay_ledger.jsonl").open("w", encoding="utf-8") as ledger:
                for index in range(len(metadata["timestamps"])):
                    ok, frame = capture.read()
                    if not ok:
                        raise ValueError(f"Source decode stopped at frame {index}")
                    row = paint(frame, index, document, metadata, analytics, grouped[index], trails)
                    ledger.write(dump(row)+"\n")
                    writer.write(frame)
                    if index in {0, len(metadata["timestamps"])//2, len(metadata["timestamps"])-1}:
                        cv2.imwrite(str(root / f"frame_{index:04d}.jpg"), frame)
                    if index % 100 == 0:
                        job.progress = (index+1)/len(metadata["timestamps"])*.8
                        db.commit()
            capture.release()
            writer.release()
            capture = writer = None
            encoder = shutil.which("ffmpeg")
            if not encoder:
                import imageio_ffmpeg
                encoder = imageio_ffmpeg.get_ffmpeg_exe()
            output = root / "presentation.mp4"
            subprocess.run([encoder, "-y", "-i", str(staging), "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "-an", str(output)], check=True, capture_output=True, timeout=600)
            decoded = cv2.VideoCapture(str(output))
            frames = 0
            while True:
                ok, frame = decoded.read()
                if not ok:
                    break
                if frame.shape[:2] != (metadata["height"], metadata["width"]):
                    raise ValueError("Output dimensions differ from source")
                frames += 1
            decoded.release()
            if frames != len(metadata["timestamps"]):
                raise ValueError("Presentation frame count does not match annotation ledger")
            with (root / "overlay_ledger.jsonl").open(encoding="utf-8") as stream:
                for index, line in enumerate(stream):
                    row = json.loads(line)
                    expected = [{key: a[key] for key in ("id", "person_id", "bbox", "origin")} for a in grouped[index] if a["visibility"] == "visible"]
                    if row["visible_annotations"] != expected or row["metrics"] != analytics["timeline"][index]:
                        raise ValueError("Overlay records disagree with canonical annotations or metrics")
            report = {"revision": job.revision, "annotation_fingerprint": revision.fingerprint,
                "source_sha256": source_hash, "output_sha256": sha256(output), "total_frames": frames,
                "reviewed_frames": analytics["reviewed_frames"], "fully_reviewed_frames": analytics["fully_reviewed_frames"],
                "corrected_detections": analytics["corrected_detections"], "corrected_identity_assignments": analytics["corrected_identity_assignments"],
                "reviewed_counting_events": analytics["reviewed_entries"]+analytics["reviewed_exits"],
                "explicit_ambiguities": analytics["explicit_ambiguities"], "unconfirmed_identities": analytics["unconfirmed_identities"],
                "remaining_ambiguities_assessed": analytics["full_review"], "accuracy": "NOT MEASURED; partial human review only",
                "all_output_frames_decoded": True, "overlay_ledger_matches_annotations": True,
                "aggregates_recomputed_from_revision": True, "render_seconds": time.perf_counter()-started,
                "note": "Exact boxes, no interpolated detections. Trajectories join consecutive observations only. Unknown initial/live occupancy and unique visitors unavailable."}
            (root / "validation_report.json").write_text(dump(report), encoding="utf-8")
            staging.unlink()
            job.status, job.progress, job.output_path, job.report_json = "completed", 1, str(output.resolve()), dump(report)
            db.commit()
        except Exception as exc:
            if capture:
                capture.release()
            if writer:
                writer.release()
            db.rollback()
            job = db.get(ReviewRender, render_id)
            job.status, job.error = "failed", str(exc)
            db.commit()
            raise
