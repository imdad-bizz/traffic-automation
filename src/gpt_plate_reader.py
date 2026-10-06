"""
OpenAI GPT-4o Vision Post-Processing Module for Bangladeshi License Plates
===========================================================================
Performs state-of-the-art multimodal vision recognition on harvested license
plate and vehicle evidence snapshots. Designed as an offline/post-processing
system to achieve near 100% accuracy on complex, faded, or embossed plates.

Works automatically when an OpenAI API key is provided via:
  1. CLI argument: --api-key <KEY>
  2. Environment variable: OPENAI_API_KEY
  3. config.yaml: ai.openai_api_key
If no key is present, it gracefully logs instructions and allows local OCR to proceed.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def encode_image_base64(image: np.ndarray) -> str:
    """Encodes an OpenCV BGR image into a base64 JPEG string."""
    success, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not success:
        raise ValueError("Could not encode image to JPEG")
    return base64.b64encode(buffer).decode("utf-8")


class GPTPlateReader:
    """
    OpenAI Vision AI extractor for Bangladeshi ALPR.
    Extracts BRTA-compliant Bengali & English license plates with structured JSON.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        config_path: str = "config.yaml",
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model

        # Fallback to config.yaml if API key is not yet set
        if not self.api_key and Path(config_path).exists():
            try:
                import yaml
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                    ai_cfg = cfg.get("ai", {})
                    self.api_key = ai_cfg.get("openai_api_key") or None
                    self.model = ai_cfg.get("gpt_model", self.model)
            except Exception as e:
                logger.debug("Could not read API key from config: %s", e)

        self._client = None
        if self.api_key and self.api_key.strip():
            self._init_client()

    def _init_client(self) -> None:
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key.strip())
            logger.info("[GPTPlateReader] OpenAI client initialized with model: %s", self.model)
        except Exception as e:
            logger.warning("[GPTPlateReader] Could not initialize OpenAI client: %s", e)
            self._client = None

    @property
    def is_available(self) -> bool:
        """Returns True if OpenAI API key is configured and client is ready."""
        return self._client is not None

    def recognize_plate(
        self,
        plate_image: np.ndarray,
        vehicle_context_image: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Submits plate crop (and optional vehicle context) to GPT-4o Vision.
        Returns parsed Bengali & English plate strings and confidence score.
        """
        if not self.is_available:
            return {
                "plate_bn": "",
                "plate_en": "",
                "confidence": 0.0,
                "error": "OpenAI API key not configured",
            }

        if plate_image is None or plate_image.size == 0:
            return {"plate_bn": "", "plate_en": "", "confidence": 0.0}

        try:
            plate_b64 = encode_image_base64(plate_image)

            prompt_text = (
                "You are an expert Bangladeshi Vehicle License Plate Recognition system (BRTA standard).\n"
                "Analyze this license plate crop and extract the exact registration details.\n\n"
                "Bangladeshi License Plate Rules:\n"
                "- Line 1 contains the City/Metro name followed by vehicle class letter "
                "(e.g., 'ঢাকা মেট্রো-ড', 'ঢাকা মেট্রো-ন', 'ঢাকা মেট্রো-ক', 'চট্টগ্রাম মেট্রো-খ', 'সিলেট মেট্রো-গ').\n"
                "- Line 2 contains a 2-digit series number, a hyphen, and a 4-digit number (e.g., '১২-৬০১৫', '৩৫-৪৭৪১', '৩৪-৭২৭৫').\n"
                "- Commercial vehicles/trucks usually have green plates with white/black embossed digits.\n\n"
                "Respond ONLY with a valid JSON object in this exact schema without markdown backticks:\n"
                "{\n"
                '  "plate_bn": "সম্পূর্ণ বাংলা প্লেট নম্বর (যেমন: ঢাকা মেট্রো-ড ১২-৬০১৫)",\n'
                '  "plate_en": "Standard English transliteration (e.g. DHAKA METRO-DA 12-6015)",\n'
                '  "city": "City/Metro name in English",\n'
                '  "vehicle_class": "Class letter in English (e.g. DA, KA, NA)",\n'
                '  "series": "2-digit series in English",\n'
                '  "number": "4-digit number in English",\n'
                '  "confidence": float between 0.0 and 1.0,\n'
                '  "notes": "Brief observation"\n'
                "}"
            )

            messages_content: List[Dict[str, Any]] = [
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{plate_b64}",
                        "detail": "high",
                    },
                },
            ]

            if vehicle_context_image is not None and vehicle_context_image.size > 0:
                v_b64 = encode_image_base64(vehicle_context_image)
                messages_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{v_b64}",
                            "detail": "low",
                        },
                    }
                )

            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a specialized Computer Vision OCR assistant for Bangladeshi traffic ALPR. Always return pure JSON.",
                    },
                    {"role": "user", "content": messages_content},
                ],
                max_tokens=250,
                temperature=0.1,
            )

            raw_reply = response.choices[0].message.content.strip()
            # Clean possible markdown fencing
            cleaned = re.sub(r"^```(?:json)?", "", raw_reply).rstrip("`").strip()
            data = json.loads(cleaned)

            return {
                "plate_bn": data.get("plate_bn", "").strip(),
                "plate_en": data.get("plate_en", "").strip(),
                "confidence": float(data.get("confidence", 0.90)),
                "source": "GPT-4o Vision",
                "notes": data.get("notes", ""),
            }

        except Exception as e:
            logger.warning("[GPTPlateReader] Recognition error: %s", e)
            return {
                "plate_bn": "",
                "plate_en": "",
                "confidence": 0.0,
                "error": str(e),
            }


def post_process_violations_with_gpt(
    records: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    db_path: Optional[str] = "outputs/detections.db",
) -> List[Dict[str, Any]]:
    """
    Runs GPT-4o Vision post-processing on all records with captured plate crops.
    Updates the records dictionary and syncs the SQLite database.
    """
    reader = GPTPlateReader(api_key=api_key)

    if not reader.is_available:
        logger.info(
            "\n"
            "========================================================================\n"
            "ℹ️ [AI Post-Processing Notice]\n"
            "OpenAI API key was not detected. Local OCR results will be retained.\n"
            "To activate Deep GPT-4o Plate Recognition:\n"
            "  1. Set environment variable: set OPENAI_API_KEY=your_key_here\n"
            "  2. Or add to config.yaml under ai.openai_api_key: 'your_key'\n"
            "  3. Or run: python process_crops.py --gpt --api-key <YOUR_KEY>\n"
            "========================================================================"
        )
        return records

    logger.info("=================================================================")
    logger.info("Starting GPT-4o Vision Post-Processing on %d Vehicle Records", len(records))
    logger.info("=================================================================")

    conn = None
    if db_path and Path(db_path).exists():
        try:
            conn = sqlite3.connect(db_path)
        except Exception:
            conn = None

    processed_count = 0
    for record in records:
        plate_crop_path = record.get("plate_crop_path", "")
        if not plate_crop_path or not Path(plate_crop_path).exists():
            continue

        plate_img = cv2.imread(plate_crop_path)
        if plate_img is None:
            continue

        snap_path = record.get("snapshot_path", "")
        vehicle_img = cv2.imread(snap_path) if snap_path and Path(snap_path).exists() else None

        track_id = record.get("track_id", 0)
        logger.info("[GPT AI] Processing Track #%d plate crop (%s)...", track_id, plate_crop_path)

        gpt_result = reader.recognize_plate(plate_img, vehicle_img)
        if gpt_result.get("plate_bn"):
            record["plate_bn"] = gpt_result["plate_bn"]
            record["plate_en"] = gpt_result["plate_en"]
            record["plate_conf"] = gpt_result["confidence"]
            record["recognition_source"] = "GPT-4o Vision"
            processed_count += 1
            logger.info(
                "✅ [GPT AI Success] Track #%d: '%s' (%s) | Conf: %.2f",
                track_id,
                record["plate_bn"],
                record["plate_en"],
                record["plate_conf"],
            )

            if conn:
                try:
                    conn.execute(
                        "UPDATE detections SET plate_bn = ?, plate_en = ?, plate_conf = ? WHERE track_id = ?",
                        (record["plate_bn"], record["plate_en"], record["plate_conf"], track_id),
                    )
                except Exception as db_err:
                    logger.debug("DB update error: %s", db_err)

    if conn:
        conn.commit()
        conn.close()

    logger.info(
        "=================================================================\n"
        "GPT Post-Processing Complete: Successfully refined %d records.\n"
        "=================================================================",
        processed_count,
    )
    return records


def main():
    parser = argparse.ArgumentParser(description="GPT-4o Vision License Plate Recognition Tool")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API key")
    parser.add_argument("--crops-dir", type=str, default="outputs/plate_crops", help="Plate crops folder")
    parser.add_argument("--db", type=str, default="outputs/detections.db", help="SQLite database path")
    parser.add_argument("--excel", type=str, default="outputs/traffic_violations_alpr.xlsx", help="Excel output")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    from process_crops import process_all_crops
    process_all_crops(
        crops_dir=args.crops_dir,
        db_path=args.db,
        excel_path=args.excel,
        use_gpt=True,
        gpt_api_key=args.api_key,
    )


if __name__ == "__main__":
    main()
