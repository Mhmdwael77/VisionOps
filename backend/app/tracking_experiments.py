"""Explicit experiment outputs are isolated from operational Retail/Overview analyses."""
import json
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import select

from backend.app.core.database import get_db
from backend.app.services.configuration import read_config
from backend.app.models.tables import ProcessingJob
from visionops.vision.test_config import load_retail_config
from visionops.vision.tracking_experiment import ROOT, run_tracking_comparison


router = APIRouter(prefix="/api/tracking-experiments", tags=["Tracking experiments"])


def read(path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


@router.get("")
def latest():
    manifest = read(ROOT/"latest.json", {})
    relative = manifest.get("relative_dir")
    comparison = None
    if relative:
        folder = (ROOT/relative).resolve()
        if not folder.is_relative_to(ROOT.resolve()):
            raise HTTPException(500, "Invalid experiment manifest")
        comparison = read(folder/"comparison.json", None)
    return {"status": read(ROOT/"status.json", {"state": "not_started"}), "comparison": comparison}


@router.post("/run", status_code=202)
def run(background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    if (ROOT/"experiment.lock").exists() or Path("outputs/retail_store_test/run.lock").exists() or db.scalar(select(ProcessingJob.id).where(ProcessingJob.status == "running")):
        raise HTTPException(409, "Wait for current video processing to finish")
    config = load_retail_config().model_copy(update={"tracking": read_config(db).tracking})
    background_tasks.add_task(run_tracking_comparison, config)
    return {"state": "queued"}


@router.get("/artifacts/{filename:path}")
def artifact(filename: str):
    path = (ROOT/filename).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file() or path.suffix not in {".mp4", ".json", ".csv", ".md", ".jpg"}:
        raise HTTPException(404, "Experiment artifact not available")
    media = {".mp4": "video/mp4", ".json": "application/json", ".csv": "text/csv", ".md": "text/plain", ".jpg": "image/jpeg"}
    return FileResponse(path, media_type=media[path.suffix])
