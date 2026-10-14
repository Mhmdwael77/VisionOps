from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.core.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[str] = mapped_column(String(40), default="file")
    status: Mapped[str] = mapped_column(String(40), default="offline")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    source_path: Mapped[str] = mapped_column(Text, nullable=False)
    output_path: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(40), default=JobStatus.queued.value, index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str | None] = mapped_column(Text)
    entries: Mapped[int] = mapped_column(Integer, default=0)
    exits: Mapped[int] = mapped_column(Integer, default=0)
    current_occupancy: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    events: Mapped[list["Event"]] = relationship(back_populates="job", cascade="all, delete-orphan")
    zone_visits: Mapped[list["ZoneVisit"]] = relationship(back_populates="job", cascade="all, delete-orphan")


class Event(Base):
    __tablename__ = "events"
    __table_args__ = (UniqueConstraint("job_id", "track_id", "event_type", "timestamp", "label", name="uq_job_track_event_time_label"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("processing_jobs.id"), index=True)
    track_id: Mapped[int] = mapped_column(Integer, index=True)
    event_type: Mapped[str] = mapped_column(String(60), index=True)
    timestamp: Mapped[float] = mapped_column(Float, index=True)
    label: Mapped[str | None] = mapped_column(String(120))
    payload_json: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    job: Mapped[ProcessingJob] = relationship(back_populates="events")


class CountingLine(Base):
    __tablename__ = "counting_lines"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120), default="Main entrance")
    x1: Mapped[float] = mapped_column(Float, default=0.0)
    y1: Mapped[float] = mapped_column(Float, default=0.0)
    x2: Mapped[float] = mapped_column(Float, default=100.0)
    y2: Mapped[float] = mapped_column(Float, default=0.0)
    entry_direction: Mapped[str] = mapped_column(String(40), default="negative_to_positive")
    min_displacement: Mapped[float] = mapped_column(Float, default=8.0)


class Zone(Base):
    __tablename__ = "zones"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    polygon_json: Mapped[str] = mapped_column(Text, nullable=False)


class ZoneVisit(Base):
    __tablename__ = "zone_visits"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("processing_jobs.id"), index=True)
    track_id: Mapped[int] = mapped_column(Integer, index=True)
    zone_name: Mapped[str] = mapped_column(String(120), index=True)
    entered_at: Mapped[float] = mapped_column(Float)
    exited_at: Mapped[float | None] = mapped_column(Float)
    dwell_seconds: Mapped[float | None] = mapped_column(Float)
    censored: Mapped[bool | None] = mapped_column(Boolean)
    left_censored: Mapped[bool | None] = mapped_column(Boolean)
    close_reason: Mapped[str | None] = mapped_column(String(60))

    job: Mapped[ProcessingJob] = relationship(back_populates="zone_visits")


class AnalysisRecord(Base):
    __tablename__ = "analysis_records"

    job_id: Mapped[str] = mapped_column(ForeignKey("processing_jobs.id"), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    video_key: Mapped[str] = mapped_column(String(160), index=True)
    kind: Mapped[str] = mapped_column(String(30), default="video")
    artifact_dir: Mapped[str | None] = mapped_column(Text)
    summary_json: Mapped[str] = mapped_column(Text, default="{}")
    timeline_json: Mapped[str] = mapped_column(Text, default="[]")
    evidence_json: Mapped[str] = mapped_column(Text, default="[]")
    manual_review_json: Mapped[str | None] = mapped_column(Text)


class SchemaVersion(Base):
    __tablename__ = "schema_versions"
    version: Mapped[int] = mapped_column(primary_key=True)


class AppConfig(Base):
    __tablename__ = "app_config"

    key: Mapped[str] = mapped_column(String(120), primary_key=True)
    value_json: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class ReviewDataset(Base):
    __tablename__ = "review_datasets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    source_job_id: Mapped[str] = mapped_column(ForeignKey("processing_jobs.id"), unique=True)
    source_path: Mapped[str] = mapped_column(Text)
    source_sha256: Mapped[str] = mapped_column(String(64))
    metadata_json: Mapped[str] = mapped_column(Text)
    raw_json: Mapped[str] = mapped_column(Text)
    document_json: Mapped[str] = mapped_column(Text)
    analytics_json: Mapped[str] = mapped_column(Text, default="{}")
    revision: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    __mapper_args__ = {"version_id_col": revision, "version_id_generator": False}


class ReviewRevision(Base):
    __tablename__ = "review_revisions"

    dataset_id: Mapped[str] = mapped_column(ForeignKey("review_datasets.id"), primary_key=True)
    revision: Mapped[int] = mapped_column(Integer, primary_key=True)
    action: Mapped[str] = mapped_column(String(100))
    note: Mapped[str] = mapped_column(Text, default="")
    document_json: Mapped[str] = mapped_column(Text)
    analytics_json: Mapped[str] = mapped_column(Text)
    fingerprint: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ReviewRender(Base):
    __tablename__ = "review_renders"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("review_datasets.id"), index=True)
    revision: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(30), default="queued")
    progress: Mapped[float] = mapped_column(Float, default=0)
    output_path: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    report_json: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

