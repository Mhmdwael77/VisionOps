"""Validated, resolution-independent configuration for the retail video run."""
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator
from visionops.vision.tracking_config import TrackingConfig


class RegionConfig(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    polygon: list[tuple[float, float]] = Field(min_length=3, max_length=32)


class EntranceConfig(BaseModel):
    name: str = "Entrance candidate"
    start: tuple[float, float]
    end: tuple[float, float]
    entry_direction: Literal["positive_to_negative", "negative_to_positive"]
    min_displacement: float = Field(default=12, ge=0, le=200)
    reviewed: bool = False


class RetailTestConfig(BaseModel):
    source: str = "data/videos/retail_store.mp4"
    output_dir: str = "outputs/retail_store_test"
    model_path: str = "VisionOps_best.pt"
    fallback_model: str = "yolo11s.pt"
    confidence: float = Field(default=0.25, gt=0, le=1)
    iou: float = Field(default=0.55, gt=0, le=1)
    imgsz: int = Field(default=960, ge=160, le=1920, multiple_of=32)
    device: str | None = None
    trajectory_points: int = Field(default=45, ge=2, le=500)
    missing_grace_seconds: float = Field(default=1, ge=0, le=30)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    zones_reviewed: bool = False
    zones: list[RegionConfig] = Field(min_length=2, max_length=10)
    counting_line: EntranceConfig

    @model_validator(mode="after")
    def validate_geometry(self):
        points = [self.counting_line.start, self.counting_line.end]
        for zone in self.zones:
            points.extend(zone.polygon)
            area = sum(a[0] * b[1] - b[0] * a[1]
                       for a, b in zip(zone.polygon, zone.polygon[1:] + zone.polygon[:1]))
            if abs(area) < 1e-6:
                raise ValueError("Zone polygon must have nonzero area")
        if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in points):
            raise ValueError("Geometry coordinates must be normalized to [0, 1]")
        if self.counting_line.start == self.counting_line.end:
            raise ValueError("Counting line endpoints must differ")
        if len({zone.name for zone in self.zones}) != len(self.zones):
            raise ValueError("Zone names must be unique")
        if Path(self.output_dir).as_posix() != "outputs/retail_store_test":
            raise ValueError("Retail test outputs must stay under outputs/retail_store_test")
        if Path(self.source).as_posix() != "data/videos/retail_store.mp4":
            raise ValueError("Retail test source must be data/videos/retail_store.mp4")
        return self


def load_retail_config(path: Path | str = "configs/retail_store_test.json") -> RetailTestConfig:
    return RetailTestConfig.model_validate_json(Path(path).read_text(encoding="utf-8"))
