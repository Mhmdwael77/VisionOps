"""Versioned human edits. Raw inference is immutable and is never overwritten."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from uuid import uuid4

import cv2
from sqlalchemy import select
from sqlalchemy.orm.exc import StaleDataError

from backend.app.models.tables import AnalysisRecord, ProcessingJob, ReviewDataset, ReviewRevision
from visionops.vision.review_analytics import derive, fingerprint

OUTPUT = Path("outputs/retail_store_test/reviewed")


def dump(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"), default=lambda v: v.isoformat())


def sha256(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256")
    return digest.hexdigest()


def initialize(db):
    record = db.scalar(select(AnalysisRecord).join(ProcessingJob).where(
        AnalysisRecord.kind == "retail", ProcessingJob.status == "completed").order_by(ProcessingJob.started_at.desc()))
    if not record:
        raise ValueError("Complete a real retail analysis before creating a review workspace")
    existing = db.scalar(select(ReviewDataset).where(ReviewDataset.source_job_id == record.job_id))
    if existing:
        return existing
    root = Path("outputs/retail_store_test")
    summary = json.loads(record.summary_json)
    snapshot = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if snapshot["run_id"] != record.run_id:
        raise ValueError("Raw track file is not from this analysis; rerun/export matching tracks first")
    source = Path(summary["source"]["path"]).resolve()
    source_hash = sha256(source)
    if source_hash != summary["source"]["sha256"]:
        raise ValueError("Source video hash differs from the original analysis")
    rows = [json.loads(line) for line in (root / "tracks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != summary["source"]["decoded_frames"] or any(row["frame_index"] != index for index, row in enumerate(rows)):
        raise ValueError("Raw tracks do not cover the original frames in order")
    metadata = {key: summary["source"][key] for key in ("width", "height", "fps", "duration_seconds")}
    metadata.update(timestamps=[row["timestamp"] for row in rows], source_run_id=record.run_id,
                    source_model=summary["model"], missing_grace_seconds=summary["configuration"]["missing_grace_seconds"])
    configuration = summary["configuration"]
    annotations, identities = [], {}
    for row in rows:
        for detection in row["detections"]:
            identity = f"P{detection['track_id']:03d}"
            identities.setdefault(identity, {"label": identity, "confirmed": False, "note": "Raw tracker ID, not a verified person identity"})
            annotations.append({"id": f"raw-{row['frame_index']}-{detection['track_id']}", "frame_index": row["frame_index"],
                "person_id": identity, "bbox": detection["bbox"], "visibility": "visible", "confidence": detection["confidence"],
                "origin": "raw_model", "raw_track_id": detection["track_id"], "uncertainty": ""})
    document = {"schema_version": 1, "annotations": annotations, "identities": identities,
        "frame_reviews": [{"state": "unreviewed", "note": ""} for _ in rows],
        "geometry": {"counting_line": {**configuration["counting_line"], "reviewed": False},
                     "zones": [{**zone, "reviewed": False, "color": ["#36dfb5", "#ffbe69"][index % 2]} for index, zone in enumerate(configuration["zones"])]},
        "event_corrections": {}, "manual_events": [], "deleted_detections": 0}
    validate(document, metadata)
    dataset = ReviewDataset(id=str(uuid4()), source_job_id=record.job_id, source_path=str(source), source_sha256=source_hash,
        metadata_json=dump(metadata), raw_json=dump(document), document_json=dump(document), analytics_json=dump(derive(document, metadata)), revision=0)
    db.add(dataset)
    db.flush()
    db.add(ReviewRevision(dataset_id=dataset.id, revision=0, action="import_raw", note="Unreviewed copy of immutable real-video inference",
        document_json=dataset.document_json, analytics_json=dataset.analytics_json, fingerprint=fingerprint(document)))
    db.commit()
    return dataset


def validate(doc, metadata):
    count = len(metadata["timestamps"])
    if not count or any(not math.isfinite(t) or (i and t <= metadata["timestamps"][i-1]) for i, t in enumerate(metadata["timestamps"])):
        raise ValueError("Video timestamps must be finite and strictly increasing")
    if len(doc["frame_reviews"]) != count:
        raise ValueError("Every video frame needs a review state")
    for review in doc["frame_reviews"]:
        if review["state"] not in {"unreviewed", "reviewed", "ambiguous"}:
            raise ValueError("Invalid review state")
    if not doc["identities"] or any(not isinstance(v.get("confirmed"), bool) or not v.get("label") for v in doc["identities"].values()):
        raise ValueError("Identities need a label and explicit confirmation state")
    unique, per_frame = set(), set()
    for a in doc["annotations"]:
        frame = a["frame_index"]
        if type(frame) is not int or not 0 <= frame < count or a["person_id"] not in doc["identities"]:
            raise ValueError("Annotation frame or identity is invalid")
        if a["id"] in unique or (frame, a["person_id"]) in per_frame:
            raise ValueError("An identity may have only one observation per frame; resolve overlapping tracks first")
        unique.add(a["id"])
        per_frame.add((frame, a["person_id"]))
        if a["origin"] not in {"raw_model", "manual_correction", "manual_added"}:
            raise ValueError("Invalid annotation provenance")
        if a["visibility"] not in {"visible", "occluded"}:
            raise ValueError("Visibility must be visible or occluded")
        if a["visibility"] == "occluded":
            if a.get("bbox") is not None:
                raise ValueError("Occlusion records must not invent a visible bounding box")
        else:
            box = a["bbox"]
            if not isinstance(box, list) or len(box) != 4 or not all(isinstance(n, (int, float)) and math.isfinite(n) for n in box):
                raise ValueError("Expected four finite pixel coordinates")
            if not (0 <= box[0] < box[2] <= metadata["width"] and 0 <= box[1] < box[3] <= metadata["height"]):
                raise ValueError("Bounding box must be inside the source frame and have positive area")
    geometry = doc["geometry"]
    line = geometry["counting_line"]
    if line["entry_direction"] not in {"negative_to_positive", "positive_to_negative"} or line["start"] == line["end"]:
        raise ValueError("Invalid line endpoints or direction")
    if not isinstance(line["min_displacement"], (int, float)) or not math.isfinite(line["min_displacement"]) or line["min_displacement"] < 0:
        raise ValueError("Invalid line dead band")
    names = [z["name"] for z in geometry["zones"]]
    if len(names) != len(set(names)) or any(not name for name in names):
        raise ValueError("Zones need unique nonempty names")
    for z in geometry["zones"]:
        if not isinstance(z["color"], str) or len(z["color"]) != 7 or not z["color"].startswith("#"):
            raise ValueError("Zone color must use #RRGGBB")
        try:
            int(z["color"][1:], 16)
        except ValueError as exc:
            raise ValueError("Invalid zone color") from exc
        if len(z["polygon"]) < 3:
            raise ValueError("Zones need at least three vertices")
        points = z["polygon"]
        if abs(sum(points[i][0]*points[(i+1)%len(points)][1]-points[(i+1)%len(points)][0]*points[i][1] for i in range(len(points)))) < 1e-6:
            raise ValueError("Zone polygon has zero area")
    for pair in [line["start"], line["end"]] + [p for z in geometry["zones"] for p in z["polygon"]]:
        if len(pair) != 2 or not all(isinstance(n, (int, float)) and math.isfinite(n) and 0 <= n <= 1 for n in pair):
            raise ValueError("Geometry must use finite normalized coordinates")
    for value in [line, *geometry["zones"]]:
        if type(value["reviewed"]) is not bool:
            raise ValueError("Geometry review state must be explicit")
    event_ids = set()
    for e in doc["manual_events"]:
        if e["id"] in event_ids or e["person_id"] not in doc["identities"] or type(e["frame_index"]) is not int or not 0 <= e["frame_index"] < count:
            raise ValueError("Invalid manual event identity, ID or frame")
        event_ids.add(e["id"])
        allowed = {"line": {"entry", "exit"}, "zone": {"zone_entry", "zone_exit"}}
        if e["kind"] not in allowed or e["event_type"] not in allowed[e["kind"]] or e["review_status"] not in {"accepted", "rejected", "unreviewed"}:
            raise ValueError("Invalid event type or review status")
        if e["region"] not in ([line["name"]] if e["kind"] == "line" else names):
            raise ValueError("Event region must exist in the geometry")
    for e in doc["event_corrections"].values():
        if e["status"] not in {"accepted", "rejected", "unreviewed"}:
            raise ValueError("Invalid event decision")


def edit(db, dataset, revision, action, payload, note):
    if revision != dataset.revision:
        raise RuntimeError("This workspace changed. Reload before saving your edit")
    if not note.strip():
        raise ValueError("A review note is required for every change")
    doc, metadata = json.loads(dataset.document_json), json.loads(dataset.metadata_json)
    touched = set()
    if action == "annotation":
        old = next((a for a in doc["annotations"] if a["id"] == payload.get("id")), None)
        if payload.get("id") and old is None:
            raise ValueError("Annotation does not exist")
        a = deepcopy(old) if old else {"id": "manual-"+str(uuid4()), "origin": "manual_added", "confidence": None}
        for key in ("frame_index", "person_id", "bbox", "visibility", "uncertainty"):
            if key in payload:
                a[key] = payload[key]
        if a["person_id"] not in doc["identities"]:
            doc["identities"][a["person_id"]] = {"label": a["person_id"], "confirmed": False, "note": note}
        if old:
            touched.add(old["frame_index"])
            a["origin"] = "manual_correction"
            a["identity_corrected"] = old.get("identity_corrected", False) or a["person_id"] != old["person_id"]
            doc["annotations"].remove(old)
        touched.add(a["frame_index"])
        doc["annotations"].append(a)
    elif action == "delete_annotation":
        old = next(a for a in doc["annotations"] if a["id"] == payload["id"])
        touched.add(old["frame_index"])
        doc["annotations"].remove(old)
        doc["deleted_detections"] = doc.get("deleted_detections", 0)+1
    elif action in {"merge", "split"}:
        source, target = payload["source"], payload["target"]
        if source == target or source not in doc["identities"]:
            raise ValueError("Choose distinct source and target identities")
        if action == "merge" and target not in doc["identities"]:
            raise ValueError("Merge target must exist")
        if action == "split" and target in doc["identities"]:
            raise ValueError("Split target must be a new identity")
        doc["identities"].setdefault(target, {"label": target, "confirmed": False, "note": note})
        start, end = (payload.get("start_frame", 0), payload.get("end_frame", len(metadata["timestamps"])-1)) if action == "split" else (0, len(metadata["timestamps"])-1)
        changed = [a for a in doc["annotations"] if a["person_id"] == source and start <= a["frame_index"] <= end]
        if not changed:
            raise ValueError("No source observations in the selected range")
        for a in changed:
            touched.add(a["frame_index"])
            a.update(person_id=target, identity_corrected=True, origin="manual_correction")
        if action == "merge":
            for a in doc["annotations"]:
                if a["person_id"] == target:
                    touched.add(a["frame_index"])
        for e in doc["manual_events"]:
            if e["person_id"] == source and start <= e["frame_index"] <= end:
                e.update(person_id=target, review_status="unreviewed")
        doc["identities"][target].update(confirmed=False, note=note)
    elif action == "identity":
        identity = doc["identities"][payload["person_id"]]
        identity.update(label=payload["label"], confirmed=payload["confirmed"], note=note)
    elif action == "review_frames":
        start, end = payload["start_frame"], payload["end_frame"]
        if type(start) is not int or type(end) is not int or not 0 <= start <= end < len(metadata["timestamps"]):
            raise ValueError("Invalid review range")
        for i in range(start, end+1):
            doc["frame_reviews"][i] = {"state": payload["state"], "note": note}
    elif action == "geometry":
        doc["geometry"] = payload["geometry"]
    elif action == "event":
        event = next(e for e in json.loads(dataset.analytics_json)["events"] if e["id"] == payload["id"])
        if event["origin"] == "manual_event":
            next(e for e in doc["manual_events"] if e["id"] == event["id"]).update(review_status=payload["status"], note=note)
        else:
            doc["event_corrections"][event["id"]] = {"evidence_hash": event["evidence_hash"], "status": payload["status"], "note": note}
    elif action == "manual_event":
        doc["manual_events"].append({key: payload[key] for key in ("person_id", "frame_index", "kind", "region", "event_type")} |
            {"id": "event-"+str(uuid4()), "review_status": "accepted", "note": note, "censored": False})
    elif action == "import":
        if payload["source_sha256"] != dataset.source_sha256:
            raise ValueError("Imported annotations belong to a different video")
        doc = payload["document"]
        # Imports are draft edits, never an automatic human-review certificate.
        for a in doc["annotations"]:
            a["origin"] = "manual_correction"
        doc["frame_reviews"] = [{"state": "unreviewed", "note": "Imported draft requires review"} for _ in doc["frame_reviews"]]
        doc["event_corrections"] = {}
        for e in doc["manual_events"]:
            e["review_status"] = "unreviewed"
        for identity in doc["identities"].values():
            identity["confirmed"] = False
        doc["geometry"]["counting_line"]["reviewed"] = False
        for zone in doc["geometry"]["zones"]:
            zone["reviewed"] = False
    else:
        raise ValueError("Unsupported edit action")
    for frame in touched:
        doc["frame_reviews"][frame] = {"state": "unreviewed", "note": "Annotation changed; review again"}
    validate(doc, metadata)
    result = derive(doc, metadata)
    dataset.revision += 1
    dataset.document_json, dataset.analytics_json = dump(doc), dump(result)
    db.add(ReviewRevision(dataset_id=dataset.id, revision=dataset.revision, action=action, note=note,
        document_json=dataset.document_json, analytics_json=dataset.analytics_json, fingerprint=fingerprint(doc)))
    try:
        db.commit()
    except StaleDataError as exc:
        db.rollback()
        raise RuntimeError("Concurrent edit detected; reload the workspace") from exc
    return dataset


def public(dataset):
    doc = json.loads(dataset.document_json)
    return {"id": dataset.id, "revision": dataset.revision, "source_job_id": dataset.source_job_id,
        "source_sha256": dataset.source_sha256, "metadata": json.loads(dataset.metadata_json),
        "geometry": doc["geometry"], "identities": doc["identities"], "analytics": json.loads(dataset.analytics_json),
        "fingerprint": fingerprint(doc), "updated_at": dataset.updated_at}


def read_frame(dataset, index):
    metadata = json.loads(dataset.metadata_json)
    if not 0 <= index < len(metadata["timestamps"]):
        raise ValueError("Frame is outside the source video")
    capture = cv2.VideoCapture(dataset.source_path)
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            raise ValueError("Source frame cannot be decoded")
        return frame
    finally:
        capture.release()
