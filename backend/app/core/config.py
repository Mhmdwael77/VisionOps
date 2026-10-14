import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class Settings:
    app_name: str = "VisionOps"
    api_prefix: str = "/api"
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"])
    database_url: str = "sqlite:///./visionops.db"
    upload_dir: Path = Path("uploads")
    output_dir: Path = Path("outputs")
    model_path: Path = Path("VisionOps_best.pt")
    fallback_model: str = "yolo11s.pt"
    detector_confidence: float = 0.35
    detector_iou: float = 0.55
    detector_imgsz: int = 640
    detector_device: str | None = None
    frame_interval: int = 1
    max_upload_mb: int = 500
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @property
    def model_exists(self) -> bool:
        return self.model_path.exists()

    @property
    def effective_model(self) -> str:
        return str(self.model_path if self.model_exists else self.fallback_model)


@lru_cache
def get_settings() -> Settings:
    settings = Settings(
        database_url=os.getenv("VISIONOPS_DATABASE_URL", Settings.database_url),
        upload_dir=Path(os.getenv("VISIONOPS_UPLOAD_DIR", str(Settings.upload_dir))),
        output_dir=Path(os.getenv("VISIONOPS_OUTPUT_DIR", str(Settings.output_dir))),
        model_path=Path(os.getenv("VISIONOPS_MODEL_PATH", str(Settings.model_path))),
        fallback_model=os.getenv("VISIONOPS_FALLBACK_MODEL", Settings.fallback_model),
        detector_confidence=float(os.getenv("VISIONOPS_DETECTOR_CONFIDENCE", Settings.detector_confidence)),
        detector_iou=float(os.getenv("VISIONOPS_DETECTOR_IOU", Settings.detector_iou)),
        detector_imgsz=int(os.getenv("VISIONOPS_DETECTOR_IMGSZ", Settings.detector_imgsz)),
        detector_device=os.getenv("VISIONOPS_DETECTOR_DEVICE") or None,
        frame_interval=max(1, int(os.getenv("VISIONOPS_FRAME_INTERVAL", Settings.frame_interval))),
        max_upload_mb=int(os.getenv("VISIONOPS_MAX_UPLOAD_MB", Settings.max_upload_mb)),
        log_level=os.getenv("VISIONOPS_LOG_LEVEL", Settings.log_level),  # type: ignore[arg-type]
    )
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    return settings

