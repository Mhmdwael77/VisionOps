"""Serve only the scoped retail test artifacts and its editable geometry."""
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from visionops.vision.retail_pipeline import write_json
from visionops.vision.test_config import RetailTestConfig, load_retail_config
from backend.app.core.database import get_db
from backend.app.models.tables import AnalysisRecord, ProcessingJob
from backend.app.services.analysis_store import execute_retail_analysis
from backend.app.services.analytics import retail_results
from backend.app.services.configuration import read_config


router = APIRouter(prefix="/api/retail-test", tags=["Retail video test"])
OUTPUT = Path("outputs/retail_store_test")
CONFIG = Path("configs/retail_store_test.json")


@router.get("")
def results(db: Session = Depends(get_db)) -> dict:
    return retail_results(db)


@router.get("/config")
def get_config(db: Session = Depends(get_db)) -> RetailTestConfig:
    return load_retail_config(CONFIG).model_copy(update={"tracking": read_config(db).tracking})


@router.put("/config")
def save_config(payload: RetailTestConfig) -> RetailTestConfig:
    if (OUTPUT / "run.lock").exists():
        raise HTTPException(409, "Wait for the current run to finish before changing geometry.")
    write_json(CONFIG, payload.model_dump())
    return payload


@router.post("/run", status_code=202)
def start_run(background_tasks: BackgroundTasks, db: Session = Depends(get_db)) -> dict:
    running = db.scalar(select(ProcessingJob.id).join(AnalysisRecord).where(
        AnalysisRecord.kind == "retail", ProcessingJob.status.in_(["queued", "running"])))
    if (OUTPUT / "run.lock").exists() or Path("outputs/retail_store_test/tracking_experiment/experiment.lock").exists() or running:
        raise HTTPException(409, "The retail video is already processing.")
    config = get_config(db)
    if not Path(config.source).is_file():
        raise HTTPException(404, "Retail video missing. Run python scripts/inspect_retail_video.py first.")
    background_tasks.add_task(execute_retail_analysis, config)
    return {"state": "queued"}


@router.get("/artifacts/{filename:path}")
def artifact(filename: str, db: Session = Depends(get_db, scope="function")) -> FileResponse:
    root = OUTPUT.resolve()
    if filename != "source_frame_0000.jpg":
        record = db.scalar(select(AnalysisRecord).join(ProcessingJob).where(
            AnalysisRecord.kind == "retail", ProcessingJob.status == "completed").order_by(ProcessingJob.started_at.desc()))
        if record and record.artifact_dir:
            root = Path(record.artifact_dir).resolve()
    path = (root / filename).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.suffix not in {".mp4", ".json", ".csv", ".jpg", ".md"}:
        raise HTTPException(404, "Artifact not available")
    media_type = {".mp4": "video/mp4", ".json": "application/json", ".csv": "text/csv", ".jpg": "image/jpeg", ".md": "text/plain"}[path.suffix]
    return FileResponse(path, media_type=media_type)
