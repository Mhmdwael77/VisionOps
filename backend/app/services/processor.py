from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from backend.app.core.config import get_settings
from backend.app.core.database import SessionLocal
from backend.app.models.tables import AnalysisRecord, CountingLine, Event, JobStatus, ProcessingJob, Zone, ZoneVisit
from backend.app.services.analysis_store import video_key
from visionops.vision.retail_pipeline import digest
from backend.app.services.configuration import read_config
from visionops.vision.counting import CountingLineConfig, CrossingDirection, LineCrossingCounter
from visionops.vision.detector import YOLOPersonDetector
from visionops.vision.geometry import BBox, LineSegment, Point
from visionops.vision.occupancy import OccupancyState
from visionops.vision.zones import ZoneConfig, ZoneMonitor
from visionops.vision.overlay import draw_track


ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}


def create_job(db: Session, source_path: str) -> ProcessingJob:
    path = Path(source_path)
    if not str(source_path).isdigit() and path.suffix.lower() not in ALLOWED_VIDEO_SUFFIXES:
        raise ValueError("Unsupported video source. Use MP4, MOV, AVI, MKV, RTSP, webcam index, or local path.")
    job = ProcessingJob(id=str(uuid.uuid4()), source_path=source_path, status=JobStatus.queued.value)
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def _load_line_configs(db: Session, width: int, height: int) -> list[CountingLineConfig]:
    rows = db.scalars(select(CountingLine).order_by(CountingLine.id)).all()
    if not rows:
        rows = [
            CountingLine(
                name="Default entrance",
                x1=0,
                y1=height * 0.5,
                x2=width,
                y2=height * 0.5,
                entry_direction=CrossingDirection.negative_to_positive.value,
                min_displacement=8.0,
            )
        ]
    return [
        CountingLineConfig(
            name=row.name,
            line=LineSegment(Point(row.x1, row.y1), Point(row.x2, row.y2)),
            entry_direction=CrossingDirection(row.entry_direction),
            min_displacement=row.min_displacement,
        )
        for row in rows
    ]


def _load_zone_configs(db: Session) -> list[ZoneConfig]:
    zones = db.scalars(select(Zone).order_by(Zone.id)).all()
    configs: list[ZoneConfig] = []
    for zone in zones:
        points = [Point(float(point["x"]), float(point["y"])) for point in json.loads(zone.polygon_json)]
        configs.append(ZoneConfig(zone.name, points))
    return configs


def _draw_overlays(frame, counters: list[LineCrossingCounter], zone_monitor: ZoneMonitor | None) -> None:
    for counter in counters:
        line = counter.config.line
        cv2.line(
            frame,
            (int(line.start.x), int(line.start.y)),
            (int(line.end.x), int(line.end.y)),
            (0, 220, 255),
            2,
        )
        cv2.putText(frame, counter.config.name, (int(line.start.x) + 8, int(line.start.y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
    if zone_monitor:
        for zone in zone_monitor.zones:
            pts = [(int(point.x), int(point.y)) for point in zone.polygon]
            for idx, start in enumerate(pts):
                end = pts[(idx + 1) % len(pts)]
                cv2.line(frame, start, end, (90, 170, 255), 2)
            if pts:
                cv2.putText(frame, zone.name, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 170, 255), 2)


def _add_event(db: Session, job_id: str, track_id: int, event_type: str, timestamp: float, label: str | None, payload: dict | None = None) -> None:
    db.add(
        Event(
            job_id=job_id,
            track_id=track_id,
            event_type=event_type,
            timestamp=round(timestamp, 3),
            label=label,
            payload_json=json.dumps(payload or {}),
        )
    )


def process_job(job_id: str) -> None:
    settings = get_settings()
    db = SessionLocal()
    writer = None
    cap = None
    try:
        job = db.get(ProcessingJob, job_id)
        if job is None:
            return
        db.execute(delete(Event).where(Event.job_id == job_id))
        db.execute(delete(ZoneVisit).where(ZoneVisit.job_id == job_id))
        job.status = JobStatus.running.value
        job.started_at = datetime.now(timezone.utc)
        job.progress = 0.0
        job.error = None
        db.commit()

        source = int(job.source_path) if str(job.source_path).isdigit() else job.source_path
        runtime_config = read_config(db)
        frame_interval = max(runtime_config.frame_interval, 1)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {job.source_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        cap.release()
        cap = None

        output_path = settings.output_dir / f"{job_id}_annotated.mp4"
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), max(fps / frame_interval, 1.0), (width, height))
        detector = YOLOPersonDetector(
            settings.effective_model,
            confidence=runtime_config.detector_confidence,
            iou=runtime_config.detector_iou,
            imgsz=runtime_config.detector_imgsz,
            device=settings.detector_device,
            tracking=runtime_config.tracking,
        )

        counters = [LineCrossingCounter(config) for config in _load_line_configs(db, width, height)]
        zone_monitor = ZoneMonitor(_load_zone_configs(db))
        occupancy = OccupancyState()
        all_track_ids: set[int] = set()
        timeline = []
        peak_visible = 0
        frame_index = 0
        last_commit = time.monotonic()

        for result in detector.track_stream(source, vid_stride=frame_interval, processing_fps=fps/frame_interval):
            frame_index += frame_interval
            timestamp = frame_index / fps
            frame = result.orig_img.copy()
            active_track_ids: set[int] = set()
            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                for xyxy, track_id_tensor, conf_tensor in zip(boxes.xyxy, boxes.id, boxes.conf):
                    track_id = int(track_id_tensor.item())
                    if track_id in getattr(result, "identity_gaps", {}):
                        zone_monitor.mark_recovered_gap(track_id)
                    active_track_ids.add(track_id)
                    all_track_ids.add(track_id)
                    bbox = BBox(*[float(value) for value in xyxy.tolist()])
                    point = bbox.bottom_center
                    draw_track(frame, (bbox.x1, bbox.y1, bbox.x2, bbox.y2), track_id, float(conf_tensor.item()))

                    for counter in counters:
                        crossing = counter.update(track_id, point, timestamp)
                        if crossing:
                            if crossing.event_type == "entry":
                                occupancy.record_entry()
                            else:
                                occupancy.record_exit()
                            gap = track_id in getattr(result, "identity_gaps", {})
                            _add_event(db, job_id, track_id, crossing.event_type, timestamp, crossing.line_name,
                                       {"provisional": True, "censored": gap, "reason": "crossing_across_unobserved_gap" if gap else "observed_crossing"})

                    for zone_event in zone_monitor.update(track_id, point, timestamp):
                        _add_event(db, job_id, track_id, zone_event.event_type, timestamp, zone_event.zone_name,
                                   {"dwell_seconds": zone_event.dwell_seconds, "censored": zone_event.censored,
                                    "reason": zone_event.reason, "provisional": True})
                        if zone_event.event_type == "zone_entry":
                            db.add(ZoneVisit(job_id=job_id, track_id=track_id, zone_name=zone_event.zone_name, entered_at=timestamp,
                                             censored=zone_event.censored, left_censored=zone_event.left_censored))
                        else:
                            db.flush()
                            visit = db.scalars(
                                select(ZoneVisit)
                                .where(ZoneVisit.job_id == job_id)
                                .where(ZoneVisit.track_id == track_id)
                                .where(ZoneVisit.zone_name == zone_event.zone_name)
                                .where(ZoneVisit.exited_at.is_(None))
                                .order_by(ZoneVisit.id.desc())
                            ).first()
                            if visit:
                                visit.exited_at = timestamp
                                visit.dwell_seconds = zone_event.dwell_seconds
                                visit.censored, visit.close_reason = zone_event.censored, zone_event.reason

            for counter in counters:
                counter.forget_missing(active_track_ids)
            peak_visible = max(peak_visible, len(active_track_ids))
            timeline.append({"timestamp": timestamp, "visible_people": len(active_track_ids), "zones": zone_monitor.occupancy(active_track_ids)})
            _draw_overlays(frame, counters, zone_monitor)
            cv2.putText(frame, f"Entries {occupancy.entries} Exits {occupancy.exits} Occ {occupancy.current}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            writer.write(frame)

            job.entries = occupancy.entries
            job.exits = occupancy.exits
            job.current_occupancy = occupancy.current
            if total_frames > 0:
                job.progress = min(frame_index / total_frames, 0.99)
            if time.monotonic() - last_commit > 1.5:
                db.commit()
                last_commit = time.monotonic()

        for zone_event in zone_monitor.close_open_visits(frame_index / fps):
            _add_event(db, job_id, zone_event.track_id, zone_event.event_type, zone_event.timestamp, zone_event.zone_name,
                       {"dwell_seconds": zone_event.dwell_seconds, "censored": zone_event.censored, "reason": zone_event.reason, "provisional": True})
            db.flush()
            visit = db.scalars(select(ZoneVisit).where(ZoneVisit.job_id == job_id, ZoneVisit.track_id == zone_event.track_id,
                ZoneVisit.zone_name == zone_event.zone_name, ZoneVisit.exited_at.is_(None)).order_by(ZoneVisit.id.desc())).first()
            if visit:
                visit.exited_at, visit.dwell_seconds = zone_event.timestamp, zone_event.dwell_seconds
                visit.censored, visit.close_reason = zone_event.censored, zone_event.reason

        source_hash = digest(Path(job.source_path)) if Path(job.source_path).is_file() else None
        record = db.get(AnalysisRecord, job_id)
        if record is None:
            record = AnalysisRecord(job_id=job_id)
            db.add(record)
        record.run_id, record.kind = job_id, "video"
        record.video_key = video_key(job.source_path, source_hash)
        record.summary_json = json.dumps({"source": {"path": job.source_path, "sha256": source_hash, "duration_seconds": frame_index/fps},
            "configuration": {"tracking": runtime_config.tracking.model_dump(), "counting_line": {"reviewed": False}, "zones_reviewed": False,
                              "zones": [zone.model_dump() for zone in runtime_config.zones]},
            "analytics": {"unique_track_ids": len(all_track_ids), "final_visible_people": timeline[-1]["visible_people"] if timeline else None,
                          "peak_visible_people": peak_visible}})
        record.timeline_json = json.dumps(timeline)

        job.status = JobStatus.completed.value
        job.output_path = str(output_path)
        job.progress = 1.0
        job.completed_at = datetime.now(timezone.utc)
        db.commit()
    except Exception as exc:  # noqa: BLE001
        job = db.get(ProcessingJob, job_id)
        if job:
            job.status = JobStatus.failed.value
            job.error = str(exc)
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
    finally:
        if writer is not None:
            writer.release()
        if cap is not None:
            cap.release()
        db.close()

