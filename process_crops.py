"""
High-Accuracy Offline License Plate Processor & Excel Exporter
==============================================================
Runs deep, multi-scale Bengali/English ALPR on harvested license plate snapshots,
optionally leverages GPT-4o Vision for next-generation AI plate extraction,
and exports executive-ready structured reports to Excel with embedded image thumbnails.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import logging
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Configure console utf-8 encoding for Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.excel_exporter import export_to_excel
from src.plate_detector import BanglaPlateOCR, PlatePreprocessor
from src.gpt_plate_reader import post_process_violations_with_gpt
from src.utils import load_config, setup_logging

logger = logging.getLogger(__name__)


def deep_recognize_plate(
    ocr: BanglaPlateOCR, plate_img: np.ndarray
) -> Tuple[str, str, float]:
    """
    Performs multi-scale, high-accuracy deep recognition on a single license plate snapshot.
    Bypasses real-time shortcuts to achieve maximum character and digit precision.
    """
    if plate_img is None or plate_img.size == 0 or ocr._bn_reader is None:
        return "", "", 0.0

    h, w = plate_img.shape[:2]

    # Test scales: standard, 2x, and 3x with border reflection padding
    scales = [
        ("scaled_260", cv2.resize(plate_img, (max(240, w * 2), max(68, h * 2)), interpolation=cv2.INTER_CUBIC)),
        ("raw", plate_img),
        ("scaled_320", cv2.resize(plate_img, (max(300, w * 3), max(80, h * 3)), interpolation=cv2.INTER_CUBIC)),
    ]

    best_bn = ""
    best_en = ""
    best_conf = 0.0

    for s_name, img_variant in scales:
        vh, vw = img_variant.shape[:2]

        # Line 1: Upper 54% (City & Class)
        # Line 2: Lower 58% (Series & Registration Digits)
        h_list = [
            [0, vw, 0, int(vh * 0.54)],
            [0, vw, int(vh * 0.42), vh],
        ]

        try:
            res = ocr._bn_reader.recognize(img_variant, horizontal_list=h_list, free_list=[])
            l1_text = res[0][1].strip() if len(res) > 0 else ""
            l1_conf = float(res[0][2]) if len(res) > 0 else 0.0

            l2_text = res[1][1].strip() if len(res) > 1 else ""
            l2_conf = float(res[1][2]) if len(res) > 1 else 0.0

            bn_text, en_text, syntax_score = ocr.parse_bangla_plate(l1_text, l2_text)
            conf = (max(l1_conf, 0.1) * 0.3 + max(l2_conf, 0.1) * 0.7) * syntax_score

            if conf > best_conf and len(bn_text) >= 3:
                best_bn = bn_text
                best_en = en_text
                best_conf = conf

            if best_conf >= 0.75:
                break
        except Exception as e:
            logger.debug("[DeepALPR] Error on scale %s: %s", s_name, e)

    return best_bn, best_en, best_conf


def process_all_crops(
    crops_dir: str = "outputs/plate_crops",
    db_path: str = "outputs/detections.db",
    excel_path: str = "outputs/traffic_violations_alpr.xlsx",
    speed_limit_kmh: float = 60.0,
    speed_tolerance_kmh: float = 5.0,
    use_gpt: bool = True,
    gpt_api_key: Optional[str] = None,
) -> str:
    """Processes all harvested plate crops and exports structured results to Excel."""
    logger.info("=================================================================")
    logger.info("Starting High-Accuracy Offline Plate Recognition & Excel Export")
    logger.info("Harvested Crops Directory: %s", crops_dir)
    logger.info("=================================================================")

    crop_files = sorted(glob.glob(os.path.join(crops_dir, "*.png")))
    if not crop_files:
        logger.warning("No plate crops found in %s", crops_dir)
        return ""

    logger.info("Found %d plate snapshots to process with deep OCR.", len(crop_files))

    ocr = BanglaPlateOCR()

    # Load speeds and metadata from SQLite if available
    db_info: Dict[int, Dict[str, Any]] = {}
    db_p = Path(db_path)
    if db_p.exists():
        try:
            conn = sqlite3.connect(str(db_p))
            cur = conn.cursor()
            
            # Check table columns
            cur.execute("PRAGMA table_info(detections)")
            cols = [row[1] for row in cur.fetchall()]
            
            query = "SELECT track_id, timestamp_sec, speed_kmh, peak_speed_kmh, snapshot_path"
            has_pc_date = "pc_date" in cols
            has_pc_time = "pc_time" in cols
            if has_pc_date and has_pc_time:
                query += ", pc_date, pc_time"
            query += " FROM detections"

            rows = cur.execute(query).fetchall()
            for r in rows:
                tid = int(r[0])
                info = {
                    "timestamp_sec": float(r[1] or 0.0),
                    "speed_kmh": float(r[2] or 0.0),
                    "peak_speed_kmh": float(r[3] or r[2] or 0.0),
                    "snapshot_path": r[4] or "",
                }
                if has_pc_date and has_pc_time:
                    info["pc_date"] = r[5] or ""
                    info["pc_time"] = r[6] or ""
                db_info[tid] = info
            conn.close()
        except Exception as e:
            logger.warning("Could not read detections.db: %s", e)

    records: List[Dict[str, Any]] = []
    now_dt = datetime.datetime.now()

    for crop_path in crop_files:
        img = cv2.imread(crop_path)
        if img is None:
            continue

        base_name = os.path.basename(crop_path)
        # Extract track id from file name (e.g. track_5_plate.png or track_5_frame_605.png)
        tid_match = re.search(r"track_(\d+)", base_name)
        track_id = int(tid_match.group(1)) if tid_match else 0

        # Generate enhanced version for crystal-clear visual quality if not yet enhanced
        enhanced_crop = PlatePreprocessor.enhance_for_display(img)
        enhanced_path = crop_path.replace(".png", "_enhanced.png")
        if not os.path.exists(enhanced_path):
            cv2.imwrite(enhanced_path, enhanced_crop)

        # Run deep recognition
        bn_text, en_text, conf = deep_recognize_plate(ocr, img)

        meta = db_info.get(track_id, {})
        ts_sec = meta.get("timestamp_sec", 0.0)
        speed = meta.get("speed_kmh", 0.0)
        peak_speed = meta.get("peak_speed_kmh", speed)
        snap_path = meta.get("snapshot_path", "")
        pc_date = meta.get("pc_date") or now_dt.strftime("%Y-%m-%d")
        pc_time = meta.get("pc_time") or now_dt.strftime("%H:%M:%S")

        logger.info(
            "Track %d: '%s' (%s) [conf: %.1f%%] Speed: %.1f km/h",
            track_id,
            bn_text or "UNREADABLE",
            en_text or "N/A",
            conf * 100,
            max(speed, peak_speed),
        )

        records.append(
            {
                "track_id": track_id,
                "timestamp_sec": ts_sec,
                "pc_date": pc_date,
                "pc_time": pc_time,
                "plate_bn": bn_text,
                "plate_en": en_text,
                "plate_conf": conf,
                "recognition_source": "EasyOCR Neural / BRTA",
                "speed_kmh": speed,
                "peak_speed_kmh": peak_speed,
                "plate_crop_path": enhanced_path if os.path.exists(enhanced_path) else crop_path,
                "snapshot_path": snap_path,
            }
        )

    # Sort records by track_id
    records.sort(key=lambda r: r["track_id"])

    # Optional AI Post-Processing with GPT-4o Vision
    if use_gpt:
        records = post_process_violations_with_gpt(
            records=records,
            api_key=gpt_api_key,
            db_path=str(db_p) if db_p.exists() else None,
        )

    # Update SQLite database with deep OCR / GPT results
    if db_p.exists():
        try:
            conn = sqlite3.connect(str(db_p))
            for r in records:
                if r.get("plate_bn"):
                    conn.execute(
                        "UPDATE detections SET plate_bn = ?, plate_en = ?, plate_conf = ? WHERE track_id = ?",
                        (r["plate_bn"], r["plate_en"], r["plate_conf"], r["track_id"]),
                    )
            conn.commit()
            conn.close()
            logger.info("[Database] Synchronized %d plate records into detections.db", len(records))
        except Exception as e:
            logger.warning("Could not sync detections.db: %s", e)

    # Export to Excel with embedded image thumbnails and relational mapping
    out_excel = export_to_excel(
        records=records,
        output_path=excel_path,
        speed_limit_kmh=speed_limit_kmh,
        speed_tolerance_kmh=speed_tolerance_kmh,
    )

    logger.info("=================================================================")
    logger.info("Successfully exported %d vehicle records to: %s", len(records), out_excel)
    logger.info("=================================================================")
    return out_excel


def main():
    parser = argparse.ArgumentParser(
        description="High-Accuracy Offline License Plate Recognition & Excel Report Generator"
    )
    parser.add_argument(
        "--crops-dir",
        type=str,
        default="outputs/plate_crops",
        help="Directory containing harvested plate crops (default: outputs/plate_crops)",
    )
    parser.add_argument(
        "--excel",
        type=str,
        default="outputs/traffic_violations_alpr.xlsx",
        help="Output Excel report path (default: outputs/traffic_violations_alpr.xlsx)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="outputs/detections.db",
        help="Path to detections.db (default: outputs/detections.db)",
    )
    parser.add_argument(
        "--speed-limit",
        type=float,
        default=60.0,
        help="Speed limit in km/h for violation status (default: 60.0)",
    )
    parser.add_argument(
        "--speed-tolerance",
        type=float,
        default=5.0,
        help="Speed tolerance buffer in km/h (default: 5.0)",
    )
    parser.add_argument(
        "--gpt",
        action="store_true",
        help="Enable GPT-4o Vision AI plate extraction",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key for GPT vision processing",
    )

    args = parser.parse_args()
    cfg = load_config("config.yaml") if Path("config.yaml").exists() else {}
    setup_logging(cfg)

    process_all_crops(
        crops_dir=args.crops_dir,
        db_path=args.db,
        excel_path=args.excel,
        speed_limit_kmh=args.speed_limit,
        speed_tolerance_kmh=args.speed_tolerance,
        use_gpt=args.gpt or bool(args.api_key),
        gpt_api_key=args.api_key,
    )


if __name__ == "__main__":
    main()
