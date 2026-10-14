"""Explicit opt-in tracking settings shared by jobs and controlled experiments."""
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class TrackingConfig(BaseModel):
    mode: Literal["bytetrack", "botsort", "botsort_reid"] = "bytetrack"
    lost_buffer_frames: int = Field(default=90, ge=1, le=300)
    # Association thresholds. Upstream ByteTrack defaults (0.25/0.1/0.25/0.8) assume
    # ~30 FPS footage; at low CCTV frame rates people move further between updates,
    # IoU matching fails and every miss spawns a new ID. These defaults were chosen
    # from a sweep on the 13 FPS retail clip (138 -> ~63 emitted IDs).
    track_high_thresh: float = Field(default=0.25, ge=0.05, le=0.95)
    track_low_thresh: float = Field(default=0.10, ge=0.01, le=0.9)
    new_track_thresh: float = Field(default=0.50, ge=0.05, le=0.99)
    match_thresh: float = Field(default=0.90, ge=0.3, le=0.99)
    # Offline identity resolution for recorded video (see visionops/vision/identity.py).
    reid_enabled: bool = True
    reid_model: str = "models/osnet_ain_x1_0_msmt17.pt"
    reid_split_similarity: float = Field(default=0.66, ge=0.0, le=1.0)
    reid_merge_similarity: float = Field(default=0.80, ge=0.5, le=0.999)
    reid_min_link_similarity: float = Field(default=0.72, ge=0.0, le=0.999)
    reid_max_gap_seconds: float = Field(default=30.0, ge=0.0, le=600.0)
    reid_max_overlap_seconds: float = Field(default=1.0, ge=0.0, le=10.0)
    min_track_seconds: float = Field(default=1.0, ge=0.0, le=30.0)
    gmc_method: Literal["none", "sparseOptFlow", "orb", "ecc"] = "sparseOptFlow"
    recovery_enabled: bool = True
    embedding_backend: Literal["native", "onnx"] = "native"
    embedding_model: str | None = None
    gallery_size: int = Field(default=20, ge=2, le=100)
    gallery_min_observations: int = Field(default=2, ge=2, le=20)
    gallery_max_age_frames: int = Field(default=120, ge=10, le=600)
    reliable_confidence: float = Field(default=0.50, ge=0.25, le=0.99)
    occlusion_iou: float = Field(default=0.35, gt=0, lt=1)
    min_embedding_similarity: float = Field(default=0.85, ge=0.5, le=0.999)
    min_color_similarity: float = Field(default=0.65, ge=0, le=1)
    min_recovery_iou: float = Field(default=0.20, ge=0.05, le=0.9)
    max_center_distance: float = Field(default=0.75, gt=0, le=2)
    ambiguity_margin: float = Field(default=0.08, ge=0.01, le=0.5)

    @model_validator(mode="after")
    def validate_gallery(self):
        if self.gallery_min_observations > self.gallery_size:
            raise ValueError("Minimum reliable observations exceeds gallery size")
        if self.mode == "botsort_reid" and self.embedding_backend == "onnx" and not self.embedding_model:
            raise ValueError("A local ONNX embedding model path is required")
        if self.track_low_thresh > self.track_high_thresh:
            raise ValueError("Low-score threshold exceeds the high-score threshold")
        return self

    def uses_upstream_baseline(self) -> bool:
        """True when the stock `model.track(tracker="bytetrack.yaml")` path is equivalent."""
        return (self.mode == "bytetrack" and self.lost_buffer_frames == 30 and self.track_high_thresh == 0.25
                and self.track_low_thresh == 0.10 and self.new_track_thresh == 0.25 and self.match_thresh == 0.80)
