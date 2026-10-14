"""Run the complete retail-store test and retain every artifact in its output directory."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visionops.vision.retail_pipeline import write_report
from backend.app.core.database import init_db
from backend.app.services.analysis_store import execute_retail_analysis
from visionops.vision.test_config import load_retail_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/retail_store_test.json")
    parser.add_argument("--report-only", action="store_true", help="Refresh the report using existing results and manual review")
    arguments = parser.parse_args()
    config = load_retail_config(arguments.config)
    if arguments.report_only:
        output = Path(config.output_dir)
        if (output / "run.lock").exists():
            raise RuntimeError("Wait for processing to complete before regenerating the report")
        write_report(output, json.loads((output / "summary.json").read_text(encoding="utf-8")))
        print(f"Report refreshed: {output / 'test_report.md'}")
    else:
        init_db()
        execute_retail_analysis(config)
