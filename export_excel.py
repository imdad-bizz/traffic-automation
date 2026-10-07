"""
Standalone Excel Exporter & Plate Post-Processor
================================================
Generates high-precision executive Excel report (.xlsx) with embedded
image thumbnails from detections and plate crops.

Usage:
    python export_excel.py [--config config.yaml] [--output outputs/traffic_violations_alpr.xlsx]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from process_crops import process_all_crops
from src.utils import load_config, setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone Traffic Surveillance Excel Exporter")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--crops-dir", type=str, default="outputs/plate_crops", help="Crops directory")
    parser.add_argument("--db", type=str, default="outputs/detections.db", help="Detections SQLite DB path")
    parser.add_argument("--output", type=str, default="outputs/traffic_violations_alpr.xlsx", help="Excel output path")
    parser.add_argument("--no-gpt", action="store_true", help="Disable GPT AI vision post-processing")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)

    speed_cfg = cfg.get("speed", {})
    speed_limit = float(speed_cfg.get("speed_limit_kmh", 60.0))
    speed_tol = float(speed_cfg.get("speed_tolerance_kmh", 5.0))
    gpt_key = cfg.get("ai", {}).get("openai_api_key") or None

    out = process_all_crops(
        crops_dir=args.crops_dir,
        db_path=args.db,
        excel_path=args.output,
        speed_limit_kmh=speed_limit,
        speed_tolerance_kmh=speed_tol,
        use_gpt=not args.no_gpt,
        gpt_api_key=gpt_key,
    )
    if out:
        logger.info("Successfully generated Excel report at: %s", out)
    else:
        logger.warning("No records were exported.")


if __name__ == "__main__":
    main()
