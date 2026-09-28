
from __future__ import annotations

import argparse
import logging

from pipeline import LaneViolationPipeline
from src.utils import load_config, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lane violation detection pipeline (wrong-way detection)",
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=None,
        help="Video source path, webcam index (0/1/2), or RTSP URL",
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    parser.add_argument("--no-display", action="store_true", help="Disable preview window")
    parser.add_argument("--save-video", action="store_true", help="Force save annotated video")
    parser.add_argument("--no-save-video", action="store_true", help="Disable annotated video saving")
    parser.add_argument("--decimation", type=int, default=None, help="Override frame decimation (1=every frame)")
    parser.add_argument("--fps", type=float, default=None, help="Override source FPS")
    return parser.parse_args()


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> None:
    if args.source is not None:
        cfg["input"]["source"] = args.source

    if args.no_display:
        cfg["display"]["show_window"] = False

    if args.save_video:
        cfg["output"]["save_annotated_video"] = True

    if args.no_save_video:
        cfg["output"]["save_annotated_video"] = False

    if args.decimation is not None:
        cfg["input"]["frame_decimation"] = max(1, int(args.decimation))

    if args.fps is not None and args.fps > 0:
        cfg["input"]["force_fps"] = float(args.fps)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    apply_cli_overrides(cfg, args)

    setup_logging(cfg)
    logger = logging.getLogger(__name__)

    logger.info("Starting lane violation pipeline")
    logger.info("Input source: %s", cfg["input"]["source"])

    pipeline = LaneViolationPipeline(cfg)
    pipeline.run()

    logger.info("Pipeline finished")


if __name__ == "__main__":
    main()
