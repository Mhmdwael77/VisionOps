"""Run full-frame, identical-detection tracker comparisons without changing baseline outputs."""
import argparse
from pathlib import Path
import sys
from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.core.database import SessionLocal, init_db
from backend.app.services.configuration import read_config
from backend.app.models.tables import ProcessingJob
from visionops.vision.test_config import load_retail_config
from visionops.vision.tracking_experiment import run_tracking_comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", help="Fully annotated CSV: frame_index,person_id,x1,y1,x2,y2; install motmetrics")
    parser.add_argument("--detection-cache", help="Reuse a validated earlier run directory; hashes and detector settings must match")
    args = parser.parse_args()
    init_db()
    with SessionLocal() as db:
        if db.scalar(select(ProcessingJob.id).where(ProcessingJob.status == "running")):
            raise RuntimeError("Wait for active video processing before running a controlled comparison")
        config = load_retail_config().model_copy(update={"tracking": read_config(db).tracking})
    run_tracking_comparison(config, args.ground_truth, args.detection_cache)
