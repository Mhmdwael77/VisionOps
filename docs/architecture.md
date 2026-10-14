# VisionOps Architecture

VisionOps follows this MVP flow:

Video source -> YOLO11 person detection -> Ultralytics ByteTrack -> line/zone event processors -> SQLite persistence -> FastAPI REST/WebSocket -> React dashboard.

## Backend

The FastAPI application lives in `backend/app`. Long-running processing jobs are launched through FastAPI background tasks so ordinary API reads remain responsive. SQLite is used locally through SQLAlchemy models that are compatible with PostgreSQL-style typed columns and explicit relationships.

Key modules:

- `backend/app/main.py`: REST, WebSocket, upload, export, and download endpoints.
- `backend/app/services/processor.py`: video processing, annotation, event persistence, and job state.
- `backend/app/services/analytics.py`: persisted analytics aggregation.
- `backend/app/services/configuration.py`: line and zone configuration.

## Vision

Reusable computer vision logic lives in `visionops/vision`:

- `detector.py`: lazy Ultralytics YOLO loading and ByteTrack stream integration.
- `counting.py`: direction-aware line crossing with duplicate prevention and displacement hysteresis.
- `zones.py`: polygonal zone entry, exit, unique visitor, and dwell-time state.
- `occupancy.py`: entry/exit occupancy state with inconsistency recording.

Tracking IDs can switch during occlusion. VisionOps treats IDs as operational tracking IDs, not personal identity.

## Data

Tables include cameras, processing jobs, events, counting lines, zones, zone visits, and application config. Re-running a job clears prior events for that job before processing so one job cannot silently double-count itself.

## Privacy

The platform avoids face recognition, demographic inference, and identity inference. RTSP credentials are redacted in camera list responses. Uploaded files should be retained only as long as needed for analytics and demonstration.

