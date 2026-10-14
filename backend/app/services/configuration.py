from __future__ import annotations

import json

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from backend.app.core.config import get_settings
from backend.app.models.tables import AppConfig, CountingLine, Zone
from backend.app.schemas.api import AppConfigRead, AppConfigUpdate, CountingLineSchema, PointSchema, ZoneSchema
from visionops.vision.tracking_config import TrackingConfig


def _read_override(db: Session, key: str, default):
    row = db.get(AppConfig, key)
    if row is None:
        return default
    return json.loads(row.value_json)


def _write_override(db: Session, key: str, value) -> None:
    row = db.get(AppConfig, key)
    encoded = json.dumps(value)
    if row is None:
        db.add(AppConfig(key=key, value_json=encoded))
    else:
        row.value_json = encoded


def read_config(db: Session) -> AppConfigRead:
    settings = get_settings()
    lines = db.scalars(select(CountingLine).order_by(CountingLine.id)).all()
    zones = db.scalars(select(Zone).order_by(Zone.id)).all()
    return AppConfigRead(
        tracking=TrackingConfig.model_validate(_read_override(db, "tracking", {})),
        model_path=settings.effective_model,
        model_available=settings.model_exists,
        fallback_model=settings.fallback_model,
        detector_confidence=float(_read_override(db, "detector_confidence", settings.detector_confidence)),
        detector_iou=float(_read_override(db, "detector_iou", settings.detector_iou)),
        detector_imgsz=int(_read_override(db, "detector_imgsz", settings.detector_imgsz)),
        frame_interval=int(_read_override(db, "frame_interval", settings.frame_interval)),
        counting_lines=[
            CountingLineSchema(
                name=line.name,
                x1=line.x1,
                y1=line.y1,
                x2=line.x2,
                y2=line.y2,
                entry_direction=line.entry_direction,
                min_displacement=line.min_displacement,
            )
            for line in lines
        ],
        zones=[
            ZoneSchema(
                name=zone.name,
                polygon=[PointSchema(**point) for point in json.loads(zone.polygon_json)],
            )
            for zone in zones
        ],
    )


def update_config(db: Session, payload: AppConfigUpdate) -> AppConfigRead:
    if payload.tracking is not None:
        _write_override(db, "tracking", payload.tracking.model_dump())
    for key in ("detector_confidence", "detector_iou", "detector_imgsz", "frame_interval"):
        value = getattr(payload, key)
        if value is not None:
            _write_override(db, key, value)
    if payload.counting_lines is not None:
        db.execute(delete(CountingLine))
        for line in payload.counting_lines:
            db.add(
                CountingLine(
                    name=line.name,
                    x1=line.x1,
                    y1=line.y1,
                    x2=line.x2,
                    y2=line.y2,
                    entry_direction=line.entry_direction,
                    min_displacement=line.min_displacement,
                )
            )
    if payload.zones is not None:
        db.execute(delete(Zone))
        for zone in payload.zones:
            db.add(
                Zone(
                    name=zone.name,
                    polygon_json=json.dumps([point.model_dump() for point in zone.polygon]),
                )
            )
    db.commit()
    return read_config(db)

