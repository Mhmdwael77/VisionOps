from __future__ import annotations

import asyncio
import csv
import io
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.config import get_settings
from backend.app.core.database import SessionLocal, get_db, init_db
from backend.app.models.tables import Camera, Event, ProcessingJob
from backend.app.schemas.api import (
    AppConfigRead,
    AppConfigUpdate,
    CameraCreate,
    CameraRead,
    EventRead,
    JobCreate,
    JobRead,
    OverviewAnalytics,
    TrafficPoint,
    ZoneAnalytics,
)
from backend.app.services import analytics
from backend.app.services.analysis_store import recover_retail_analysis
from backend.app.services.configuration import read_config, update_config
from backend.app.services.processor import ALLOWED_VIDEO_SUFFIXES, create_job, process_job
from backend.app.retail_test import router as retail_test_router
from backend.app.tracking_experiments import router as tracking_experiments_router
from backend.app.review import router as review_router


settings = get_settings()
EXPERIMENT_LOCK = Path("outputs/retail_store_test/tracking_experiment/experiment.lock")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    with SessionLocal() as db:
        recover_retail_analysis(db)
    yield


app = FastAPI(title="VisionOps API", version="0.1.0", lifespan=lifespan)
app.include_router(retail_test_router)
app.include_router(tracking_experiments_router)
app.include_router(review_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _redact_source(source: str) -> str:
    parsed = urlsplit(source)
    if parsed.scheme.lower() != "rtsp" or "@" not in parsed.netloc:
        return source
    host = parsed.netloc.split("@", 1)[1]
    return urlunsplit((parsed.scheme, f"***:***@{host}", parsed.path, parsed.query, parsed.fragment))


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_path": settings.effective_model,
        "model_available": settings.model_exists,
    }


@app.get("/api/cameras", response_model=list[CameraRead])
def list_cameras(db: Session = Depends(get_db)) -> list[Camera]:
    cameras = db.scalars(select(Camera).order_by(Camera.id.desc())).all()
    for camera in cameras:
        camera.source = _redact_source(camera.source)
    return cameras


@app.post("/api/cameras", response_model=CameraRead)
def add_camera(payload: CameraCreate, db: Session = Depends(get_db)) -> Camera:
    camera = Camera(name=payload.name, source=payload.source, source_type=payload.source_type, status="configured")
    db.add(camera)
    db.commit()
    db.refresh(camera)
    camera.source = _redact_source(camera.source)
    return camera


@app.get("/api/videos", response_model=list[JobRead])
def list_videos(db: Session = Depends(get_db)) -> list[ProcessingJob]:
    return db.scalars(select(ProcessingJob).order_by(ProcessingJob.created_at.desc())).all()


@app.post("/api/videos/upload", response_model=JobRead)
async def upload_video(request: Request, db: Session = Depends(get_db)) -> ProcessingJob:
    filename = request.headers.get("x-filename", "upload.mp4")
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(status_code=400, detail="Unsupported video type.")
    body = await request.body()
    if len(body) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Uploaded file is too large.")
    safe_name = Path(filename).name
    target = settings.upload_dir / safe_name
    counter = 1
    while target.exists():
        target = settings.upload_dir / f"{Path(safe_name).stem}_{counter}{suffix}"
        counter += 1
    target.write_bytes(body)
    return create_job(db, str(target))


@app.post("/api/jobs", response_model=JobRead)
def start_job(payload: JobCreate, background_tasks: BackgroundTasks, db: Session = Depends(get_db)) -> ProcessingJob:
    if EXPERIMENT_LOCK.exists():
        raise HTTPException(409, "Wait for the controlled tracker comparison to finish")
    try:
        job = create_job(db, payload.source_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    background_tasks.add_task(process_job, job.id)
    return job


@app.post("/api/jobs/{job_id}/start", response_model=JobRead)
def start_existing_job(job_id: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)) -> ProcessingJob:
    if EXPERIMENT_LOCK.exists():
        raise HTTPException(409, "Wait for the controlled tracker comparison to finish")
    job = db.get(ProcessingJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    background_tasks.add_task(process_job, job.id)
    return job


@app.get("/api/jobs/{job_id}", response_model=JobRead)
def get_job(job_id: str, db: Session = Depends(get_db)) -> ProcessingJob:
    job = db.get(ProcessingJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@app.get("/api/jobs/{job_id}/download")
def download_output(job_id: str, db: Session = Depends(get_db, scope="function")) -> FileResponse:
    job = db.get(ProcessingJob, job_id)
    if job is None or not job.output_path:
        raise HTTPException(status_code=404, detail="Output not available.")
    path = Path(job.output_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Output file missing.")
    return FileResponse(path, media_type="video/mp4", filename=path.name)


@app.get("/api/analytics/overview", response_model=OverviewAnalytics)
def analytics_overview(analysis_id: str = "latest", db: Session = Depends(get_db)) -> OverviewAnalytics:
    return analytics.overview(db, analysis_id)


@app.get("/api/analytics/traffic", response_model=list[TrafficPoint])
def analytics_traffic(analysis_id: str = "latest", db: Session = Depends(get_db)) -> list[TrafficPoint]:
    return analytics.traffic(db, analysis_id)


@app.get("/api/analytics/zones", response_model=list[ZoneAnalytics])
def analytics_zones(analysis_id: str = "latest", db: Session = Depends(get_db)) -> list[ZoneAnalytics]:
    return analytics.zones(db, analysis_id)


@app.get("/api/events", response_model=list[EventRead])
def list_events(analysis_id: str = "latest", limit: int = Query(default=100, ge=1, le=500), db: Session = Depends(get_db)) -> list[EventRead]:
    return analytics.event_log(db, analysis_id, limit)


@app.get("/api/config", response_model=AppConfigRead)
def get_config(db: Session = Depends(get_db)) -> AppConfigRead:
    return read_config(db)


@app.put("/api/config", response_model=AppConfigRead)
def put_config(payload: AppConfigUpdate, db: Session = Depends(get_db)) -> AppConfigRead:
    return update_config(db, payload)


@app.get("/api/exports/events.csv")
def export_events(analysis_id: str = "latest", db: Session = Depends(get_db)) -> Response:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["job_id", "track_id", "event_type", "timestamp", "label", "payload_json", "provisional", "censored"])
    for row in reversed(analytics.event_log(db, analysis_id)):
        writer.writerow([row.job_id, row.track_id, row.event_type, row.timestamp, row.label,
                         row.payload_json, row.provisional, row.censored])
    return Response(output.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=events.csv"})


@app.websocket("/ws/analytics")
async def analytics_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            with SessionLocal() as db:
                await websocket.send_json(analytics.overview(db).model_dump(mode="json"))
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        return

