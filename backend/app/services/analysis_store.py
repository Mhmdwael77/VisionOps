"""The database is the read source for completed analyses; files are import/export artifacts."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from uuid import NAMESPACE_URL, uuid4, uuid5

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.database import SessionLocal
from backend.app.models.tables import AnalysisRecord, Event, ProcessingJob, ZoneVisit


def read_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def video_key(source: str, sha256: str | None = None) -> str:
    if sha256:
        return "sha256:" + sha256
    return "path:" + str(uuid5(NAMESPACE_URL, str(Path(source).resolve()).casefold()))


def import_retail_analysis(db: Session, output: Path, job_id: str | None = None) -> ProcessingJob | None:
    completed_now = job_id is not None
    status = read_json(output / "status.json", {})
    summary = read_json(output / "summary.json", {})
    if status.get("state") != "completed" or summary.get("state") != "completed":
        return None
    run_id = summary["run_id"]
    if run_id != status.get("run_id"):
        raise ValueError("Retail status and summary refer to different runs")
    existing = db.scalar(select(AnalysisRecord).where(AnalysisRecord.run_id == run_id))
    if existing:
        return db.get(ProcessingJob, existing.job_id)
    event_file = read_json(output / "events.json")
    if not event_file or event_file.get("run_id") != run_id:
        raise ValueError("Retail event log does not match the completed run")
    events = event_file["events"]
    visits = read_json(output / "zone_visits.json")
    timeline = read_json(output / "occupancy.json")
    if visits is None or timeline is None:
        raise ValueError("Retail visits or occupancy are missing")
    entries = sum(event["event_type"] == "entry" for event in events)
    exits = sum(event["event_type"] == "exit" for event in events)
    declared = summary["analytics"]
    if (entries, exits, len(events)) != (declared["entries"], declared["exits"], declared["event_count"]):
        raise ValueError("Retail event log and summary counts disagree")
    for visit in visits:
        if visit["dwell_seconds"] < 0 or abs(visit["dwell_seconds"] - (visit["exited_at"]-visit["entered_at"])) > 1e-6:
            raise ValueError("Zone dwell must match its video timestamps")
    job_id = job_id or str(uuid5(NAMESPACE_URL, "visionops:retail:" + run_id))
    job = db.get(ProcessingJob, job_id)
    if job is None:
        job = ProcessingJob(id=job_id, source_path=summary["source"]["path"])
        db.add(job)
        db.flush()
    # Freeze media per run so later processing cannot change historical playback.
    archive = output / "runs" / job_id
    archive.mkdir(parents=True, exist_ok=True)
    for name in ["annotated.mp4", "summary.json", "events.json", "events.csv", "zone_visits.json", "occupancy.json",
                 "evidence.json", "effective_config.json", "manual_review.json", "test_report.md", "frame_metrics.csv",
                 "source_frame_0000.jpg"]:
        if (output / name).is_file():
            shutil.copy2(output / name, archive / name)
    if (output / "frames").is_dir():
        shutil.copytree(output / "frames", archive / "frames", dirs_exist_ok=True)
    record = db.get(AnalysisRecord, job_id)
    if record is None:
        record = AnalysisRecord(job_id=job_id)
        db.add(record)
    manual = read_json(output / "manual_review.json")
    if manual and manual.get("run_id") != run_id:
        manual = None
    record.run_id, record.kind = run_id, "retail"
    record.video_key = video_key(job.source_path, summary["source"].get("sha256"))
    record.artifact_dir = str(archive.resolve())
    record.summary_json = json.dumps(summary)
    record.timeline_json = json.dumps(timeline)
    record.evidence_json = json.dumps(read_json(output / "evidence.json", []))
    record.manual_review_json = json.dumps(manual) if manual else None
    for event in events:
        db.add(Event(job_id=job_id, track_id=event["track_id"], event_type=event["event_type"], timestamp=event["timestamp"],
                     label=event.get("zone_name") or event.get("line_name"), payload_json=json.dumps(event)))
    for visit in visits:
        db.add(ZoneVisit(job_id=job_id, track_id=visit["track_id"], zone_name=visit["zone_name"],
                         entered_at=visit["entered_at"], exited_at=visit["exited_at"], dwell_seconds=visit["dwell_seconds"],
                         censored=visit.get("censored"), left_censored=visit.get("left_censored"), close_reason=visit.get("reason")))
    job.entries, job.exits = entries, exits
    job.current_occupancy = 0  # Historical analytics never populate a live-occupancy value.
    job.started_at = datetime.fromisoformat(run_id)
    job.completed_at = datetime.now(timezone.utc) if completed_now else job.completed_at
    job.created_at = job.created_at or job.started_at
    job.output_path = str((archive / "annotated.mp4").resolve()) if (archive / "annotated.mp4").is_file() else None
    job.status, job.progress, job.error = "completed", 1.0, None
    db.commit()
    db.refresh(job)
    return job


def recover_retail_analysis(db: Session, output: Path = Path("outputs/retail_store_test")) -> None:
    if not (output / "run.lock").exists():
        import_retail_analysis(db, output)


def execute_retail_analysis(config, job_id: str | None = None) -> dict:
    from visionops.vision.retail_pipeline import run_retail_test

    with SessionLocal() as db:
        from backend.app.services.configuration import read_config
        config = config.model_copy(update={"tracking": read_config(db).tracking})
        job_id = job_id or str(uuid4())
        job = ProcessingJob(id=job_id, source_path=config.source, status="running", started_at=datetime.now(timezone.utc))
        db.add(job)
        db.flush()
        record = AnalysisRecord(job_id=job_id, run_id="pending:" + job_id,
                                video_key=video_key(config.source), kind="retail", summary_json="{}")
        db.add(record)
        db.commit()
        def progress(status):
            job.progress = min(status.get("progress", 0), 0.99)
            record.summary_json = json.dumps({"processing": status})
            db.commit()
        try:
            summary = run_retail_test(config, on_progress=progress)
            imported = import_retail_analysis(db, Path(config.output_dir), job_id)
            if imported is None:
                raise RuntimeError("Processing finished without persistable completed artifacts")
            return summary
        except Exception as exc:
            db.rollback()
            job = db.get(ProcessingJob, job_id)
            job.status, job.error, job.completed_at = "failed", str(exc), datetime.now(timezone.utc)
            db.commit()
            raise
