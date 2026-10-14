from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.core.database import SessionLocal, init_db
from backend.app.schemas.api import AppConfigUpdate, CountingLineSchema, PointSchema, ZoneSchema
from backend.app.services.configuration import update_config


if __name__ == "__main__":
    init_db()
    data = json.loads(Path("configs/default_config.json").read_text(encoding="utf-8"))
    payload = AppConfigUpdate(
        counting_lines=[CountingLineSchema(**line) for line in data.get("counting_lines", [])],
        zones=[
            ZoneSchema(
                name=zone["name"],
                polygon=[PointSchema(**point) for point in zone.get("polygon", [])],
            )
            for zone in data.get("zones", [])
        ],
    )
    with SessionLocal() as db:
        update_config(db, payload)
    print("Default counting lines and zones loaded.")

