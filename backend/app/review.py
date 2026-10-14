"""Real-video review API, backed by SQLite snapshots and immutable revisions."""
import csv
import io
import json
from uuid import uuid4

import cv2
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

from backend.app.core.database import get_db
from backend.app.models.tables import ReviewDataset, ReviewRender, ReviewRevision
from backend.app.services import review_store as store

router = APIRouter(prefix="/api/review", tags=["Human video review"])


def dataset(db, dataset_id):
    value = db.get(ReviewDataset, dataset_id)
    if not value:
        raise HTTPException(404, "Review dataset not found")
    return value


@router.get("")
def latest(db=Depends(get_db)):
    value = db.scalar(select(ReviewDataset).order_by(ReviewDataset.updated_at.desc()))
    return store.public(value) if value else None


@router.post("/initialize")
def initialize(db=Depends(get_db)):
    try:
        return store.public(store.initialize(db))
    except (ValueError, OSError) as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/{dataset_id}")
def read(dataset_id: str, db=Depends(get_db)):
    return store.public(dataset(db, dataset_id))


class Edit(BaseModel):
    revision: int = Field(ge=0)
    action: str
    payload: dict
    note: str = Field(min_length=1, max_length=4000)


@router.post("/{dataset_id}/edit")
def edit(dataset_id: str, body: Edit, db=Depends(get_db)):
    try:
        return store.public(store.edit(db, dataset(db, dataset_id), body.revision, body.action, body.payload, body.note))
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    except (ValueError, KeyError, StopIteration, TypeError, IndexError) as exc:
        db.rollback()
        raise HTTPException(422, str(exc) or "Invalid review edit") from exc


@router.get("/{dataset_id}/frames/{index}")
def frame(dataset_id: str, index: int, raw: bool = False, db=Depends(get_db)):
    value = dataset(db, dataset_id)
    doc = json.loads(value.raw_json if raw else value.document_json)
    if not 0 <= index < len(doc["frame_reviews"]):
        raise HTTPException(404, "Frame not found")
    return {"revision": value.revision, "frame_index": index, "review": doc["frame_reviews"][index],
        "annotations": [a for a in doc["annotations"] if a["frame_index"] == index]}


@router.get("/{dataset_id}/images/{index}.jpg")
def image(dataset_id: str, index: int, db=Depends(get_db)):
    try:
        frame = store.read_frame(dataset(db, dataset_id), index)
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise ValueError("Frame encoding failed")
        return Response(encoded.tobytes(), media_type="image/jpeg")
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/{dataset_id}/source.mp4")
def source(dataset_id: str, db=Depends(get_db, scope="function")):
    value = dataset(db, dataset_id)
    return FileResponse(value.source_path, media_type="video/mp4")


@router.get("/{dataset_id}/history")
def history(dataset_id: str, db=Depends(get_db)):
    dataset(db, dataset_id)
    return [{"revision": row.revision, "action": row.action, "note": row.note, "fingerprint": row.fingerprint,
             "created_at": row.created_at} for row in db.scalars(select(ReviewRevision).where(
                 ReviewRevision.dataset_id == dataset_id).order_by(ReviewRevision.revision.desc()))]


@router.get("/{dataset_id}/export/{format}")
def export(dataset_id: str, format: str, raw: bool = False, db=Depends(get_db)):
    value = dataset(db, dataset_id)
    doc = json.loads(value.raw_json if raw else value.document_json)
    if format == "json":
        body = {**store.public(value), "document": doc, "provenance": "immutable_raw" if raw else "review_workspace"}
        if raw:
            body["analytics"] = store.derive(doc, json.loads(value.metadata_json))
            body["revision"], body["fingerprint"] = 0, store.fingerprint(doc)
        return Response(store.dump(body), media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="annotations-r{value.revision}.json"'})
    if format not in {"events.csv", "visits.csv", "annotations.csv"}:
        raise HTTPException(404, "Export format not available")
    active = store.derive(doc, json.loads(value.metadata_json)) if raw else json.loads(value.analytics_json)
    rows = doc["annotations"] if format == "annotations.csv" else active["events" if format == "events.csv" else "visits"]
    keys = sorted({key for row in rows for key in row})
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=["revision", "dataset_id", *keys])
    writer.writeheader()
    for row in rows:
        safe = {key: (store.dump(v) if isinstance(v, (dict, list)) else v) for key, v in row.items()}
        for key, v in safe.items():
            if isinstance(v, str) and v.startswith(("=", "+", "-", "@")):
                safe[key] = "'"+v
        writer.writerow({"revision": 0 if raw else value.revision, "dataset_id": value.id, **safe})
    return Response(stream.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{format}"'})


@router.post("/{dataset_id}/render", status_code=202)
def render(dataset_id: str, tasks: BackgroundTasks, db=Depends(get_db, scope="function")):
    from backend.app.services.review_renderer import render_revision
    value = dataset(db, dataset_id)
    running = db.scalar(select(ReviewRender).where(ReviewRender.dataset_id == dataset_id, ReviewRender.status.in_(["queued", "running"])))
    if running:
        raise HTTPException(409, "A presentation render is already running")
    row = ReviewRender(id=str(uuid4()), dataset_id=dataset_id, revision=value.revision, status="queued")
    db.add(row)
    db.commit()
    tasks.add_task(render_revision, row.id)
    return {"id": row.id, "revision": row.revision, "status": row.status}


@router.get("/{dataset_id}/renders")
def renders(dataset_id: str, db=Depends(get_db)):
    dataset(db, dataset_id)
    return [{"id": row.id, "revision": row.revision, "status": row.status, "progress": row.progress,
             "error": row.error, "report": json.loads(row.report_json) if row.report_json else None}
            for row in db.scalars(select(ReviewRender).where(ReviewRender.dataset_id == dataset_id).order_by(ReviewRender.created_at.desc()))]


@router.get("/{dataset_id}/renders/{render_id}.mp4")
def rendered(dataset_id: str, render_id: str, db=Depends(get_db, scope="function")):
    row = db.get(ReviewRender, render_id)
    if not row or row.dataset_id != dataset_id or row.status != "completed":
        raise HTTPException(404, "Presentation render is not complete")
    return FileResponse(row.output_path, media_type="video/mp4", filename=f"visionops-reviewed-r{row.revision}.mp4")
