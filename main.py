from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.utils import load_config, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bangladeshi Number Plate Detection, Speed Estimation & Traffic Violation Monitoring Pipeline",
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=None,
        help="Video source path, webcam index (0/1/2), or RTSP URL (defaults to config.yaml input.source)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "lane", "speed", "plate", "traffic_light", "excel"],
        help="Execution mode: 'all' (Master unified: Wrong-lane + Speed + ALPR + Excel), 'lane' (Wrong-lane only), 'speed' (Speed estimation only), 'plate' (ALPR & plate capture only), 'traffic_light' (Red light violation), or 'excel' (Generate Excel audit report)",
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    parser.add_argument("--no-display", action="store_true", help="Disable preview window")
    parser.add_argument("--save-video", action="store_true", help="Force save annotated video")
    parser.add_argument("--no-save-video", action="store_true", help="Disable annotated video saving")
    parser.add_argument("--decimation", type=int, default=None, help="Override frame decimation (1=every frame)")
    parser.add_argument("--fps", type=float, default=None, help="Override source FPS")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after processing N frames")

    # Options for traffic light mode
    parser.add_argument("--strip-half-width", type=int, default=35, help="Half-width in pixels for the stop-strip violation zone")
    parser.add_argument("--csv", type=str, default=None, help="Override CSV output path")
    parser.add_argument("--snapshots-dir", type=str, default=None, help="Override snapshots directory")
    parser.add_argument("--annotated-video", type=str, default=None, help="Override annotated video path")
    parser.add_argument("--no-gpt", action="store_true", help="Disable GPT AI vision post-processing in excel mode")

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

    if args.csv is not None:
        cfg["paths"]["csv_log"] = args.csv

    if args.annotated_video is not None:
        cfg["paths"]["annotated_video"] = args.annotated_video


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    apply_cli_overrides(cfg, args)

    setup_logging(cfg)
    logger = logging.getLogger(__name__)

    logger.info("=================================================================")
    logger.info("Starting Traffic Monitoring Pipeline [Mode: %s]", args.mode.upper())
    logger.info("Input source: %s", cfg["input"]["source"])
    logger.info("=================================================================")

    if args.mode == "all":
        from plate_speed_pipeline import PlateSpeedPipeline

        pipeline = PlateSpeedPipeline(cfg, enable_lane=True)
        pipeline.run(max_frames=args.max_frames)

    elif args.mode == "lane":
        from wrong_lane_violation import LaneViolationPipeline

        pipeline = LaneViolationPipeline(cfg)
        pipeline.run(max_frames=args.max_frames)

    elif args.mode == "speed":
        from speed_violation import SpeedViolationPipeline

        pipeline = SpeedViolationPipeline(cfg)
        pipeline.run(max_frames=args.max_frames)

    elif args.mode == "plate":
        from license_plate_capture import LicensePlateCapturePipeline

        pipeline = LicensePlateCapturePipeline(cfg)
        pipeline.run(max_frames=args.max_frames)

    elif args.mode == "traffic_light":
        import traffic_light_violation

        if args.csv is None:
            args.csv = cfg["paths"].get("csv_log", "outputs/traffic_light_violations.csv")
        if args.snapshots_dir is None:
            args.snapshots_dir = "outputs/traffic_light_snapshots"
        if args.annotated_video is None:
            args.annotated_video = cfg["paths"].get("annotated_video", "outputs/annotated_traffic_light_violation.mp4")

        pipeline = traffic_light_violation.TrafficLightViolationPipeline(cfg, args)
        pipeline.run(max_frames=args.max_frames)

    elif args.mode == "excel":
        from process_crops import process_all_crops

        speed_cfg = cfg.get("speed", {})
        speed_limit = float(speed_cfg.get("speed_limit_kmh", 60.0))
        speed_tol = float(speed_cfg.get("speed_tolerance_kmh", 5.0))
        gpt_key = cfg.get("ai", {}).get("openai_api_key") or None

        excel_path = str(Path(cfg.get("paths", {}).get("output_dir", "outputs")) / "traffic_violations_alpr.xlsx")
        process_all_crops(
            crops_dir="outputs/plate_crops",
            db_path=str(cfg.get("paths", {}).get("db_log", "outputs/detections.db")),
            excel_path=excel_path,
            speed_limit_kmh=speed_limit,
            speed_tolerance_kmh=speed_tol,
            use_gpt=not args.no_gpt,
            gpt_api_key=gpt_key,
        )

    logger.info("Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()
