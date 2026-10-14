from datetime import datetime
from pydantic import BaseModel, Field
from visionops.vision.tracking_config import TrackingConfig


class CameraCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    source: str = Field(min_length=1)
    source_type: str = "file"


class CameraRead(CameraCreate):
    id: int
    status: str

    model_config = {"from_attributes": True}


class JobCreate(BaseModel):
    source_path: str = Field(min_length=1)


class JobRead(BaseModel):
    id: str
    source_path: str
    output_path: str | None
    status: str
    progress: float
    error: str | None
    entries: int
    exits: int
    current_occupancy: int
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    model_config = {"from_attributes": True}


class EventRead(BaseModel):
    id: int
    job_id: str
    track_id: int
    event_type: str
    timestamp: float
    label: str | None
    payload_json: str | None
    provisional: bool | None = None
    censored: bool | None = None
    source_name: str | None = None

    model_config = {"from_attributes": True}


class PointSchema(BaseModel):
    x: float
    y: float


class CountingLineSchema(BaseModel):
    name: str = "Main entrance"
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 100.0
    y2: float = 0.0
    entry_direction: str = "negative_to_positive"
    min_displacement: float = 8.0


class ZoneSchema(BaseModel):
    name: str
    polygon: list[PointSchema]


class AppConfigRead(BaseModel):
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    model_path: str
    model_available: bool
    fallback_model: str
    detector_confidence: float
    detector_iou: float
    detector_imgsz: int
    frame_interval: int
    counting_lines: list[CountingLineSchema]
    zones: list[ZoneSchema]


class AppConfigUpdate(BaseModel):
    tracking: TrackingConfig | None = None
    detector_confidence: float | None = Field(default=None, ge=0.01, le=1.0)
    detector_iou: float | None = Field(default=None, ge=0.01, le=1.0)
    detector_imgsz: int | None = Field(default=None, ge=160, le=1920)
    frame_interval: int | None = Field(default=None, ge=1, le=120)
    counting_lines: list[CountingLineSchema] | None = None
    zones: list[ZoneSchema] | None = None


class TrafficPoint(BaseModel):
    bucket: str
    entries: int
    exits: int
    start_seconds: int = 0


class ZoneAnalytics(BaseModel):
    zone_name: str
    tracker_id_count: int
    historical_final_occupancy: int | None = None
    historical_peak_occupancy: int | None = None
    visits: int
    censored_visits: int
    incomplete_visits: int
    unknown_censoring_visits: int
    complete_visits: int
    measured_visits: int
    average_dwell_seconds: float | None
    average_uncensored_dwell_seconds: float | None


class AnalysisRead(BaseModel):
    id: str
    source_name: str
    source_path: str
    status: str
    started_at: datetime | None
    completed_at: datetime | None
    entries: int | None
    exits: int | None
    provisional: bool
    entrance_reviewed: bool
    zones_reviewed: bool
    output_available: bool
    error: str | None


class OverviewAnalytics(BaseModel):
    analysis_id: str
    selected_analysis_ids: list[str]
    analyses: list[AnalysisRead]
    history: list[AnalysisRead]
    total_visitors: int | None = None
    visitors_unavailable_reason: str
    current_occupancy: int | None = None
    occupancy_unavailable_reason: str
    historical_visible_people: int | None
    historical_peak_visible_people: int | None
    tracker_id_count: int | None
    entries: int | None
    exits: int | None
    event_count: int
    line_counts_provisional: bool
    entrance_geometry_status: str
    average_dwell_seconds: float | None
    average_uncensored_dwell_seconds: float | None
    visits: int
    censored_visits: int
    incomplete_visits: int
    unknown_censoring_visits: int
    complete_visits: int
    measured_visits: int
    dwell_definition: str
    completed_jobs: int
    running_jobs: int
    recent_events: list[EventRead]
    traffic: list[TrafficPoint]
    zones: list[ZoneAnalytics]

