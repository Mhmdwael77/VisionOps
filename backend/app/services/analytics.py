"""Query persisted video analyses with explicit historical and censoring semantics."""
from collections import defaultdict
import json
from pathlib import Path

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models.tables import AnalysisRecord, Event, ProcessingJob, ZoneVisit
from backend.app.schemas.api import AnalysisRead, EventRead, OverviewAnalytics, TrafficPoint, ZoneAnalytics
from backend.app.services.analysis_store import video_key


DWELL_DEFINITION = (
    "Mean tracked span averages measured closed visits, including censored spans. "
    "Complete-only mean excludes unknown entry times, lost tracks, video-end closures and unknown legacy censoring. "
    "Open or inconsistent visits are incomplete and excluded from both means. "
    "These are video-time track spans, not verified customer dwell; ID fragmentation and provisional zones affect estimates."
)


def records(db):
    jobs = db.scalars(select(ProcessingJob).order_by(ProcessingJob.started_at.desc(), ProcessingJob.created_at.desc(), ProcessingJob.id)).all()
    metadata = {row.job_id: row for row in db.scalars(select(AnalysisRecord)).all()}
    return jobs, metadata


def selected(db, analysis_id="latest"):
    jobs, metadata = records(db)
    completed = [job for job in jobs if job.status == "completed"]
    if analysis_id == "latest":
        chosen = completed[:1]
    elif analysis_id == "all":
        videos = set()
        chosen = []
        for job in completed:
            key = metadata[job.id].video_key if job.id in metadata else video_key(job.source_path)
            if key not in videos:
                chosen.append(job)
                videos.add(key)
    else:
        chosen = [job for job in completed if job.id == analysis_id]
        if not chosen:
            raise HTTPException(404, "Completed analysis not found")
    return jobs, metadata, chosen


def job_summary(job, metadata):
    record = metadata.get(job.id)
    return json.loads(record.summary_json) if record else {}


def analysis_read(job, metadata) -> AnalysisRead:
    summary = job_summary(job, metadata)
    config = summary.get("configuration", {})
    reviewed = config.get("counting_line", {}).get("reviewed", False)
    return AnalysisRead(id=job.id, source_name=Path(job.source_path).name, source_path=job.source_path, status=job.status,
                        started_at=job.started_at, completed_at=job.completed_at,
                        entries=job.entries if job.status == "completed" else None,
                        exits=job.exits if job.status == "completed" else None,
                        provisional=not reviewed or summary.get("analytics", {}).get("line_counts_provisional", False),
                        entrance_reviewed=reviewed, zones_reviewed=config.get("zones_reviewed", False),
                        output_available=bool(job.output_path and Path(job.output_path).is_file()), error=job.error)


def dwell_stats(visits):
    def measurable(row):
        return (row.exited_at is not None and row.dwell_seconds is not None and row.dwell_seconds >= 0
                and abs(row.dwell_seconds-(row.exited_at-row.entered_at)) < 0.002)
    measured = [row for row in visits if measurable(row)]
    complete = [row for row in measured if row.censored is False and row.left_censored is not True
                and row.close_reason not in {"track_lost", "video_end", "unobserved_entry"}]
    mean = lambda rows: sum(row.dwell_seconds for row in rows)/len(rows) if rows else None
    return {"visits": len(visits), "censored_visits": sum(row.censored is True for row in visits),
            "incomplete_visits": len(visits)-len(measured), "unknown_censoring_visits": sum(row.censored is None for row in visits),
            "complete_visits": len(complete), "measured_visits": len(measured),
            "average_dwell_seconds": mean(measured), "average_uncensored_dwell_seconds": mean(complete)}


def event_read(event, jobs, metadata):
    data = EventRead.model_validate(event).model_dump()
    payload = json.loads(event.payload_json or "{}")
    job = jobs[event.job_id]
    review = analysis_read(job, metadata)
    provisional = review.provisional if event.event_type in {"entry", "exit"} else not review.zones_reviewed
    data.update(provisional=payload.get("provisional", provisional),
                censored=payload.get("censored"), source_name=Path(job.source_path).name)
    return EventRead(**data)


def traffic_for(events, chosen, metadata):
    crossings = [event for event in events if event.event_type in {"entry", "exit"}]
    durations = [job_summary(job, metadata).get("source", {}).get("duration_seconds", 0) for job in chosen]
    horizon = max([event.timestamp for event in crossings] + durations + [0])
    if not chosen or not horizon:
        return []
    buckets = {start: {"entry": 0, "exit": 0} for start in range(0, int(horizon)+1, 15)}
    for event in crossings:
        buckets[int(event.timestamp//15)*15][event.event_type] += 1
    return [TrafficPoint(bucket=f"{start//60:02d}:{start%60:02d}", start_seconds=start,
                         entries=counts["entry"], exits=counts["exit"]) for start, counts in sorted(buckets.items())]


def zones_for(visits, chosen, metadata):
    grouped = defaultdict(list)
    for job in chosen:
        for zone in job_summary(job, metadata).get("configuration", {}).get("zones", []):
            grouped[zone["name"]] = []
    for visit in visits:
        grouped[visit.zone_name].append(visit)
    timeline = json.loads(metadata[chosen[0].id].timeline_json) if len(chosen) == 1 and chosen[0].id in metadata else []
    result = []
    for name, rows in sorted(grouped.items()):
        observed = [point.get("zones", {}).get(name) for point in timeline]
        known = [value for value in observed if value is not None]
        result.append(ZoneAnalytics(zone_name=name, tracker_id_count=len({(row.job_id, row.track_id) for row in rows}),
                                   historical_final_occupancy=observed[-1] if observed else None,
                                   historical_peak_occupancy=max(known) if known else None, **dwell_stats(rows)))
    return result


def overview(db: Session, analysis_id: str = "latest") -> OverviewAnalytics:
    jobs, metadata, chosen = selected(db, analysis_id)
    ids = [job.id for job in chosen]
    events = db.scalars(select(Event).where(Event.job_id.in_(ids))).all() if ids else []
    visits = db.scalars(select(ZoneVisit).where(ZoneVisit.job_id.in_(ids))).all() if ids else []
    job_map = {job.id: job for job in jobs}
    job_order = {job.id: i for i, job in enumerate(jobs)}
    recent = sorted(events, key=lambda event: (job_order[event.job_id], -event.timestamp, -event.id))[:20]
    summaries = [job_summary(job, metadata) for job in chosen]
    tracker_counts = [s.get("analytics", {}).get("unique_track_ids") for s in summaries]
    reviews = [analysis_read(job, metadata).entrance_reviewed for job in chosen]
    historical = summaries[0].get("analytics", {}) if len(summaries) == 1 else {}
    provisional_events = any(json.loads(event.payload_json or "{}").get("provisional", False)
                             for event in events if event.event_type in {"entry", "exit"})
    if analysis_id not in {"latest", "all"}:
        key = metadata[chosen[0].id].video_key if chosen and chosen[0].id in metadata else (video_key(chosen[0].source_path) if chosen else None)
        history = [job for job in jobs if (metadata[job.id].video_key if job.id in metadata else video_key(job.source_path)) == key]
    else:
        history = jobs
    return OverviewAnalytics(
        analysis_id=analysis_id, selected_analysis_ids=ids,
        analyses=[analysis_read(job, metadata) for job in jobs if job.status == "completed"],
        history=[analysis_read(job, metadata) for job in history[:50]],
        visitors_unavailable_reason="Unique customers cannot be determined from tracker IDs or provisional line crossings.",
        occupancy_unavailable_reason="Live occupancy is unavailable: these are historical videos, and initial store occupancy is unknown.",
        historical_visible_people=historical.get("final_visible_people"), historical_peak_visible_people=historical.get("peak_visible_people"),
        tracker_id_count=sum(tracker_counts) if tracker_counts and all(count is not None for count in tracker_counts) else None,
        entries=sum(event.event_type == "entry" for event in events) if chosen else None,
        exits=sum(event.event_type == "exit" for event in events) if chosen else None,
        event_count=len(events), line_counts_provisional=not reviews or not all(reviews) or provisional_events,
        entrance_geometry_status=("unavailable" if not reviews else "reviewed" if all(reviews) else "unreviewed" if not any(reviews) else "mixed"),
        **dwell_stats(visits), dwell_definition=DWELL_DEFINITION,
        completed_jobs=len(chosen), running_jobs=sum(job.status in {"queued", "running"} for job in history),
        recent_events=[event_read(event, job_map, metadata) for event in recent],
        traffic=traffic_for(events, chosen, metadata), zones=zones_for(visits, chosen, metadata))


def traffic(db: Session, analysis_id: str = "latest") -> list[TrafficPoint]:
    return overview(db, analysis_id).traffic


def zones(db: Session, analysis_id: str = "latest") -> list[ZoneAnalytics]:
    return overview(db, analysis_id).zones


def event_log(db: Session, analysis_id: str = "latest", limit: int | None = None) -> list[EventRead]:
    jobs, metadata, chosen = selected(db, analysis_id)
    order = {job.id: i for i, job in enumerate(jobs)}
    rows = db.scalars(select(Event).where(Event.job_id.in_([job.id for job in chosen]))).all()
    rows.sort(key=lambda row: (order[row.job_id], -row.timestamp, -row.id))
    job_map = {job.id: job for job in jobs}
    return [event_read(row, job_map, metadata) for row in rows[:limit]]


def retail_results(db: Session) -> dict:
    jobs, metadata = records(db)
    retail_jobs = [job for job in jobs if job.id in metadata and metadata[job.id].kind == "retail"]
    if not retail_jobs:
        return {"status": {"state": "not_started", "progress": 0}, "summary": None,
                "timeline": [], "events": [], "evidence": [], "manual_review": None}
    job = retail_jobs[0]
    record = metadata[job.id]
    summary = job_summary(job, metadata)
    status = {**summary.get("processing", {}), "state": job.status, "progress": job.progress,
              "run_id": record.run_id, "error": job.error}
    if job.status != "completed":
        return {"status": status, "summary": None, "timeline": [], "events": [], "evidence": [], "manual_review": None}
    scoped = overview(db, job.id)
    summary["analytics"].update(entries=scoped.entries, exits=scoped.exits, event_count=scoped.event_count,
                                 line_counts_provisional=scoped.line_counts_provisional)
    zone_map = {zone.zone_name: zone for zone in scoped.zones}
    for zone in summary["analytics"].get("zones", []):
        stored = zone_map.get(zone["name"])
        if stored:
            zone.update(average_dwell_seconds=stored.average_dwell_seconds,
                        average_uncensored_dwell_seconds=stored.average_uncensored_dwell_seconds,
                        visits=stored.visits, censored_visits=stored.censored_visits, incomplete_visits=stored.incomplete_visits)
    event_rows = db.scalars(select(Event).where(Event.job_id == job.id).order_by(Event.timestamp, Event.id)).all()
    timeline = json.loads(record.timeline_json)
    sampled = timeline[::10]
    if timeline and (not sampled or sampled[-1] != timeline[-1]):
        sampled.append(timeline[-1])
    return {"status": status, "analysis_id": job.id, "summary": summary, "timeline": sampled,
            "events": [json.loads(event.payload_json) for event in event_rows],
            "evidence": json.loads(record.evidence_json),
            "manual_review": json.loads(record.manual_review_json) if record.manual_review_json else None}
