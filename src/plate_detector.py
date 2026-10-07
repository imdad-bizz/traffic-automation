"""
License Plate Detection & Recognition Module (Bangladeshi ALPR)
==============================================================
Provides YOLO-based number plate localization and custom Bengali/English
OCR recognition tailored for Bangladeshi vehicle registration plates.
Includes BRTA syntax validation, canonicalization, and multi-frame temporal voting.
"""

from __future__ import annotations

import logging
import os
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports for heavy packages
_ultralytics_yolo = None
_easyocr_module = None


def get_yolo_class():
    global _ultralytics_yolo
    if _ultralytics_yolo is None:
        from ultralytics import YOLO
        _ultralytics_yolo = YOLO
    return _ultralytics_yolo


def get_easyocr_module():
    global _easyocr_module
    if _easyocr_module is None:
        import easyocr
        _easyocr_module = easyocr
    return _easyocr_module


# ---------------------------------------------------------------------------
# Bengali character mappings and BRTA standard syntax
# ---------------------------------------------------------------------------
BN_DIGITS = "০১২৩৪৫৬৭৮৯"
EN_DIGITS = "0123456789"
BN_TO_EN_DIGITS = dict(zip(BN_DIGITS, EN_DIGITS))
EN_TO_BN_DIGITS = dict(zip(EN_DIGITS, BN_DIGITS))

BN_CLASSES = {
    "ক": "KA", "খ": "KHA", "গ": "GA", "ঘ": "GHA", "ঙ": "NGA",
    "চ": "CHA", "ছ": "CHHA", "জ": "JA", "ঝ": "JHA",
    "ট": "TA", "ঠ": "THA", "ড": "DA", "ঢ": "DHA",
    "ত": "TA", "থ": "THA", "দ": "DA", "ধ": "DHA", "ন": "NA",
    "প": "PA", "ফ": "PHA", "ব": "BA", "ভ": "BHA", "ম": "MA",
    "য": "YA", "র": "RA", "ল": "LA", "শ": "SHA", "ষ": "SSA",
    "স": "SA", "হ": "HA", "ড়": "RA", "ঢ়": "RHA", "য়": "YA",
    "ই": "E", "অ": "A"
}

CITIES = [
    ("ঢাকা মেট্রো", ["ঢাকা", "মেট্রো", "ঢা", "মেট", "নেট্", "ঢারা", "ঢরা", "মেট্"]),
    ("চট্টগ্রাম মেট্রো", ["চট্টগ্রাম", "চট্ট", "চিটাগং", "চট্টগ"]),
    ("সিলেট মেট্রো", ["সিলেট"]),
    ("রাজশাহী মেট্রো", ["রাজশাহী", "রাজ"]),
    ("খুলনা মেট্রো", ["খুলনা"]),
    ("বরিশাল মেট্রো", ["বরিশাল"]),
    ("রংপুর মেট্রো", ["রংপুর"]),
    ("গাজীপুর", ["গাজীপুর"]),
    ("নারায়ণগঞ্জ", ["নারায়ণগঞ্জ", "নারায়ন"]),
    ("কুমিল্লা", ["কুমিল্লা"]),
    ("বগুড়া", ["বগুড়া", "বগুড়া"]),
    ("ময়মনসিংহ", ["ময়মনসিংহ", "ময়মনসিংহ"]),
    ("পাবনা", ["পাবনা"]),
    ("যশোর", ["যশোর"]),
    ("টাঙ্গাইল", ["টাঙ্গাইল"]),
    ("ফরিদপুর", ["ফরিদপুর"]),
]


@dataclass
class PlateDetectionResult:
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in frame coordinates
    confidence: float
    plate_text: str = ""
    plate_text_en: str = ""
    ocr_confidence: float = 0.0
    cropped_plate: Optional[np.ndarray] = None


class PlatePreprocessor:
    """Preprocesses cropped license plate images to maximize OCR readability."""

    @classmethod
    def enhance_for_ocr(cls, plate_crop: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """Generates the optimal enhanced variations of the plate crop for high-accuracy OCR."""
        if plate_crop is None or plate_crop.size == 0:
            return []

        h, w = plate_crop.shape[:2]
        target_w = 260
        scale = max(1.0, target_w / float(w))
        target_h = max(64, int(h * scale))
        resized = cv2.resize(plate_crop, (int(w * scale), target_h), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # CLAHE + Unsharp masking: optimal for Bengali numeral contrast against green background
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)

        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharp = cv2.filter2D(clahe_img, -1, kernel)

        return [
            ("raw_rgb", cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)),
            ("sharp_clahe", cv2.cvtColor(sharp, cv2.COLOR_GRAY2RGB)),
        ]

    @classmethod
    def enhance_for_display(cls, plate_crop: np.ndarray) -> np.ndarray:
        """
        Enhances a license plate crop to maximum visual clarity (super-resolution upscaling,
        denoising, and contrast-stretched CLAHE) for clear snapshot evidence and AI vision.
        """
        if plate_crop is None or plate_crop.size == 0:
            return plate_crop

        h, w = plate_crop.shape[:2]
        target_w = max(280, w * 2)
        scale = target_w / float(w)
        target_h = max(72, int(h * scale))

        # High quality bicubic interpolation
        upscaled = cv2.resize(plate_crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        # Bilateral filter to smooth background noise while keeping character edges crisp
        denoised = cv2.bilateralFilter(upscaled, d=5, sigmaColor=50, sigmaSpace=50)

        # Contrast enhancement in LAB space
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(6, 6))
        l_clahe = clahe.apply(l)
        enhanced_bgr = cv2.cvtColor(cv2.merge([l_clahe, a, b]), cv2.COLOR_LAB2BGR)

        # Unsharp mask for crisp character outline
        gaussian = cv2.GaussianBlur(enhanced_bgr, (0, 0), 2.0)
        sharpened = cv2.addWeighted(enhanced_bgr, 1.35, gaussian, -0.35, 0)
        return sharpened

    @classmethod
    def compute_plate_sharpness(cls, plate_crop: np.ndarray) -> float:
        """Calculates Laplacian variance to measure sharpness and focus."""
        if plate_crop is None or plate_crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY) if len(plate_crop.shape) == 3 else plate_crop
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class LicensePlateDetector:
    """YOLO-based license plate detector with vehicle-geometry spatial filtering."""

    def __init__(
        self,
        model_path: str = "models/plate_detector.pt",
        confidence_threshold: float = 0.20,
    ) -> None:
        self.model_path = model_path
        self.conf = confidence_threshold
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        p = Path(self.model_path)
        if not p.exists():
            raise FileNotFoundError(f"Plate detection model not found: {self.model_path}")
        logger.info("[LicensePlateDetector] Loading model from %s", self.model_path)
        yolo_cls = get_yolo_class()
        self._model = yolo_cls(str(p))

    def detect_in_vehicle_crop(
        self, vehicle_crop: np.ndarray, vehicle_bbox: Tuple[int, int, int, int]
    ) -> List[PlateDetectionResult]:
        """Detect plates within a cropped vehicle image and map coords back to the full frame."""
        if vehicle_crop is None or vehicle_crop.size == 0:
            return []

        vx1, vy1, _, _ = vehicle_bbox
        vh, vw = vehicle_crop.shape[:2]

        # Vehicle must be large enough to contain a detectable plate
        if vw < 50 or vh < 50:
            return []

        results = self._model(vehicle_crop, conf=self.conf, verbose=False)[0]
        detections: List[PlateDetectionResult] = []

        if results.boxes is None or len(results.boxes) == 0:
            return detections

        for box in results.boxes:
            px1, py1, px2, py2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])

            pw, ph = px2 - px1, py2 - py1
            if pw < 18 or ph < 10:
                continue

            # Aspect ratio check: typical license plates are wider than tall (ratio between 1.1 and 4.5)
            aspect = pw / float(ph)
            if aspect < 1.0 or aspect > 4.8:
                continue

            # Spatial filter: plates are mounted on the lower 75% of vehicles (exclude roof signs/logos)
            if py1 < int(vh * 0.18) and py2 < int(vh * 0.40):
                continue

            # Add gentle margin around plate for full character retention
            pad_x = max(3, int(pw * 0.08))
            pad_y = max(3, int(ph * 0.10))

            crop_px1 = max(0, px1 - pad_x)
            crop_py1 = max(0, py1 - pad_y)
            crop_px2 = min(vw, px2 + pad_x)
            crop_py2 = min(vh, py2 + pad_y)

            plate_crop = vehicle_crop[crop_py1:crop_py2, crop_px1:crop_px2].copy()
            if plate_crop.size == 0:
                continue

            abs_bbox = (vx1 + crop_px1, vy1 + crop_py1, vx1 + crop_px2, vy1 + crop_py2)

            detections.append(
                PlateDetectionResult(
                    bbox=abs_bbox,
                    confidence=conf,
                    cropped_plate=plate_crop,
                )
            )

        return detections


class BanglaPlateOCR:
    """Optical Character Recognition for Bangladeshi License Plates with BRTA syntax parser."""

    def __init__(
        self,
        custom_model_dir: str = "models/EasyOCR/models",
        user_network_dir: str = "models/EasyOCR/user_network",
        use_gpu: bool = False,
    ) -> None:
        self.custom_model_dir = custom_model_dir
        self.user_network_dir = user_network_dir
        self.use_gpu = use_gpu

        self._bn_reader = None
        self._en_reader = None
        self._init_readers()

    def _init_readers(self) -> None:
        easyocr = get_easyocr_module()
        custom_model_p = Path(self.custom_model_dir)
        user_net_p = Path(self.user_network_dir)

        if custom_model_p.exists() and user_net_p.exists():
            try:
                logger.info("[BanglaPlateOCR] Initializing specialized bn_license_tps network...")
                self._bn_reader = easyocr.Reader(
                    ["bn"],
                    gpu=self.use_gpu,
                    model_storage_directory=str(custom_model_p.resolve()),
                    user_network_directory=str(user_net_p.resolve()),
                    recog_network="bn_license_tps",
                )
                logger.info("[BanglaPlateOCR] Specialized bn_license_tps model loaded!")
            except Exception as e:
                logger.warning("[BanglaPlateOCR] Fallback to standard bn: %s", e)
                self._bn_reader = easyocr.Reader(["bn"], gpu=self.use_gpu)
        else:
            self._bn_reader = easyocr.Reader(["bn"], gpu=self.use_gpu)

        try:
            self._en_reader = easyocr.Reader(["en"], gpu=self.use_gpu)
        except Exception as e:
            logger.warning("[BanglaPlateOCR] Could not load English reader: %s", e)

    @staticmethod
    def parse_bangla_plate(line1: str, line2: str) -> Tuple[str, str, float]:
        """
        Parses raw OCR output lines according to BRTA license plate standards.
        Rejects noise fragments and returns: (bangla_text, english_text, confidence_score_multiplier)
        """
        detected_city = ""
        detected_cls = ""

        # 1. Match City / Metro
        for canon, patterns in CITIES:
            for p in patterns:
                if p in line1:
                    detected_city = canon
                    break
            if detected_city:
                break

        # 2. Match Vehicle Class (letter)
        # Strip city and metro keywords (including OCR variants like নেটরো, ঢারা, মেট্, etc.)
        rem = re.sub(
            r"ঢা[কার]*|মেট্?[র্রো]*|নেট্?[র্রো]*|চট্টগ্রাম|সিলেট|খুলনা|বরিশাল|রাজশাহী|বগুড়া|কুমিল্লা",
            "",
            line1,
        ).strip()

        for char in reversed(rem):
            if char in BN_CLASSES:
                detected_cls = char
                break

        # 3. Clean line 2 (digits)
        digits = []
        for c in (line2 + " " + line1):  # Also check if digits were on line 1
            if c in BN_DIGITS:
                digits.append(c)
            elif c in EN_DIGITS:
                digits.append(EN_TO_BN_DIGITS[c])
            elif c == "-" and digits and digits[-1] != "-":
                digits.append("-")

        d_str = "".join(digits).strip("-")
        pure_d = [c for c in d_str if c in BN_DIGITS]

        # Require at least 2 digits to be considered a registration number
        if len(pure_d) < 2:
            d_str = ""
            pure_d = []
        elif len(pure_d) >= 4 and "-" not in d_str:
            if len(pure_d) >= 6:
                d_str = "".join(pure_d[:2]) + "-" + "".join(pure_d[2:6])
            elif len(pure_d) == 5:
                d_str = "".join(pure_d[:2]) + "-" + "".join(pure_d[2:5])
            else:
                d_str = "".join(pure_d[:2]) + "-" + "".join(pure_d[2:4])

        # If neither city nor digits found, reject as noise
        if not detected_city and not d_str:
            return "", "", 0.0

        plate_parts = []
        if detected_city:
            if detected_cls:
                plate_parts.append(f"{detected_city}-{detected_cls}")
            else:
                plate_parts.append(detected_city)
        elif detected_cls and d_str:
            plate_parts.append(f"মেট্রো-{detected_cls}")

        if d_str:
            plate_parts.append(d_str)

        full_bn = " ".join(plate_parts)

        # English transliteration
        en_city = (
            detected_city.replace("ঢাকা মেট্রো", "DHAKA METRO")
            .replace("চট্টগ্রাম মেট্রো", "CHATTOGRAM METRO")
            .replace("কুমিল্লা", "CUMILLA")
            .replace("সিলেট মেট্রো", "SYLHET METRO")
        )
        en_cls = BN_CLASSES.get(detected_cls, detected_cls)
        en_digits = "".join(BN_TO_EN_DIGITS.get(c, c) for c in d_str)

        en_parts = []
        if en_city:
            if en_cls:
                en_parts.append(f"{en_city}-{en_cls}")
            else:
                en_parts.append(en_city)
        elif en_cls and en_digits:
            en_parts.append(f"METRO-{en_cls}")

        if en_digits:
            en_parts.append(en_digits)

        full_en = " ".join(en_parts)

        score = 0.3
        if detected_city:
            score += 0.35
        if detected_cls:
            score += 0.15
        if len(pure_d) >= 4:
            score += 0.40
        elif len(pure_d) >= 2:
            score += 0.20

        return full_bn, full_en, min(1.0, score)

    def recognize(self, plate_crop: np.ndarray) -> Tuple[str, str, float]:
        """
        Recognizes plate text from a crop using direct multi-line neural network recognition.
        Returns: (plate_text_bn, plate_text_en, confidence)
        """
        if plate_crop is None or plate_crop.size == 0 or self._bn_reader is None:
            return "", "", 0.0

        variations = PlatePreprocessor.enhance_for_ocr(plate_crop)
        # Prioritize sharp_clahe first as it provides superior Bengali numeral contrast on green plates
        variations.sort(key=lambda x: 0 if x[0] == "sharp_clahe" else 1)

        best_bn = ""
        best_en = ""
        best_conf = 0.0

        for var_name, var_img in variations:
            vh, vw = var_img.shape[:2]

            # Dual-line direct box segmentation (Bangladeshi standard 2-line layout)
            line1_box = [0, vw, 0, int(vh * 0.54)]
            line2_box = [0, vw, int(vh * 0.42), vh]
            h_list = [line1_box, line2_box]

            try:
                rec_res = self._bn_reader.recognize(var_img, horizontal_list=h_list, free_list=[])
                line1_text = rec_res[0][1].strip() if len(rec_res) > 0 else ""
                line1_conf = float(rec_res[0][2]) if len(rec_res) > 0 else 0.0

                line2_text = rec_res[1][1].strip() if len(rec_res) > 1 else ""
                line2_conf = float(rec_res[1][2]) if len(rec_res) > 1 else 0.0

                # If both lines failed to detect digits, test full-box fallback
                if not line1_text and not line2_text:
                    full_res = self._bn_reader.recognize(
                        var_img, horizontal_list=[[0, vw, 0, vh]], free_list=[]
                    )
                    full_text = full_res[0][1].strip() if len(full_res) > 0 else ""
                    full_conf = float(full_res[0][2]) if len(full_res) > 0 else 0.0
                    if any(c in BN_DIGITS or c in EN_DIGITS for c in full_text):
                        line2_text = full_text
                        line2_conf = full_conf

                bn_text, en_text, syntax_score = self.parse_bangla_plate(line1_text, line2_text)
                combined_conf = (max(line1_conf, 0.1) * 0.3 + max(line2_conf, 0.1) * 0.7) * syntax_score

                if combined_conf > best_conf and len(bn_text) >= 3:
                    best_bn = bn_text
                    best_en = en_text
                    best_conf = combined_conf

                # Fast exit: if confident reading obtained, skip redundant variations
                if best_conf >= 0.30:
                    break

            except Exception as e:
                logger.debug("[BanglaPlateOCR] Direct recognize error on %s: %s", var_name, e)

        return best_bn, best_en, best_conf


class TrackPlateManager:
    """
    Manages multi-frame temporal voting, plate accumulation, and non-blocking asynchronous
    OCR recognition per vehicle track. Prevents GUI/video freezing on CPU architectures.
    """

    @dataclass
    class TrackRecord:
        track_id: int
        best_plate_text: str = ""
        best_plate_text_en: str = ""
        best_ocr_conf: float = 0.0
        best_plate_crop: Optional[np.ndarray] = None
        best_plate_crop_enhanced: Optional[np.ndarray] = None
        best_plate_sharpness: float = 0.0
        best_plate_area: int = 0
        best_vehicle_snapshot: Optional[np.ndarray] = None
        frame_idx: int = 0
        timestamp_sec: float = 0.0
        last_speed_kmh: float = 0.0
        readings_count: int = 0
        saved: bool = False

    def __init__(
        self,
        plate_detector: LicensePlateDetector,
        plate_ocr: BanglaPlateOCR,
        ocr_interval_frames: int = 3,
        min_ocr_conf: float = 0.18,
        async_ocr: bool = True,
    ) -> None:
        self.detector = plate_detector
        self.ocr = plate_ocr
        self.ocr_interval_frames = max(1, int(ocr_interval_frames))
        self.min_ocr_conf = min_ocr_conf
        self.records: Dict[int, TrackPlateManager.TrackRecord] = {}
        self._lock = threading.Lock()
        self.async_ocr = async_ocr

        if self.async_ocr:
            self._ocr_queue: queue.Queue = queue.Queue(maxsize=32)
            self._stop_event = threading.Event()
            self._worker_thread = threading.Thread(
                target=self._ocr_worker, daemon=True, name="PlateOCRWorker"
            )
            self._worker_thread.start()
        else:
            self._ocr_queue = None
            self._stop_event = None
            self._worker_thread = None

    def _ocr_worker(self) -> None:
        """Background worker thread consuming plate crops for non-blocking neural OCR."""
        while not self._stop_event.is_set():
            try:
                task = self._ocr_queue.get(timeout=0.25)
            except queue.Empty:
                continue

            if task is None:
                self._ocr_queue.task_done()
                break

            track_id, plate_crop, frame_idx = task
            try:
                bn_text, en_text, conf = self.ocr.recognize(plate_crop)
                with self._lock:
                    if track_id in self.records:
                        record = self.records[track_id]
                        record.readings_count += 1

                        is_better_text = False
                        if conf > record.best_ocr_conf and conf >= self.min_ocr_conf:
                            is_better_text = True
                        elif bn_text and not record.best_plate_text:
                            is_better_text = True
                        elif ("-" in bn_text) and ("-" not in record.best_plate_text):
                            is_better_text = True

                        if is_better_text:
                            record.best_plate_text = bn_text
                            record.best_plate_text_en = en_text
                            record.best_ocr_conf = max(conf, record.best_ocr_conf)
            except Exception as e:
                logger.debug("[TrackPlateManager] Async OCR error on track %d: %s", track_id, e)
            finally:
                self._ocr_queue.task_done()

    def update_track(
        self,
        track_id: int,
        frame: np.ndarray,
        vehicle_bbox: Tuple[int, int, int, int],
        frame_idx: int,
        timestamp_sec: float,
        speed_kmh: float = 0.0,
    ) -> Optional[PlateDetectionResult]:
        """
        Updates plate reading for a vehicle track.
        Executes fast YOLO plate localization on-thread and dispatches heavy OCR asynchronously.
        """
        with self._lock:
            if track_id not in self.records:
                self.records[track_id] = self.TrackRecord(track_id=track_id)
            record = self.records[track_id]
            record.last_speed_kmh = speed_kmh

        h, w = frame.shape[:2]
        vx1, vy1, vx2, vy2 = vehicle_bbox
        vw, vh = vx2 - vx1, vy2 - vy1

        # Skip vehicles that are too small or far near horizon for clear plate detection
        if vw < 55 or vh < 55 or vy2 < 240:
            return None

        # Determine if we should run YOLO plate detection on this frame
        should_run = (
            (record.best_plate_crop is None)
            or (frame_idx % self.ocr_interval_frames == 0)
        )
        if not should_run:
            return None

        pad = 14
        crop_y1, crop_y2 = max(0, vy1 - pad), min(h, vy2 + pad)
        crop_x1, crop_x2 = max(0, vx1 - pad), min(w, vx2 + pad)
        vehicle_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        if vehicle_crop.size == 0:
            return None

        detected_plates = self.detector.detect_in_vehicle_crop(
            vehicle_crop, (crop_x1, crop_y1, crop_x2, crop_y2)
        )

        best_det = None
        for det in detected_plates:
            plate_crop = det.cropped_plate
            if plate_crop is None or plate_crop.size == 0:
                continue

            ch, cw = plate_crop.shape[:2]
            current_area = cw * ch
            sharpness = PlatePreprocessor.compute_plate_sharpness(plate_crop)

            # Quality metrics: composite score of resolution and edge sharpness
            current_quality = current_area * np.sqrt(max(1.0, sharpness))
            prev_quality = record.best_plate_area * np.sqrt(max(1.0, record.best_plate_sharpness))

            # Upgrade plate crop if larger / sharper as vehicle approaches camera
            is_better_crop = (
                record.best_plate_crop is None
                or current_area > int(record.best_plate_area * 1.25)
                or (current_quality > prev_quality and current_area >= int(record.best_plate_area * 0.85))
            )

            if is_better_crop:
                with self._lock:
                    record.best_plate_crop = plate_crop.copy()
                    record.best_plate_crop_enhanced = PlatePreprocessor.enhance_for_display(plate_crop)
                    record.best_plate_sharpness = sharpness
                    record.best_plate_area = current_area
                    record.best_vehicle_snapshot = vehicle_crop.copy()
                    record.frame_idx = frame_idx
                    record.timestamp_sec = timestamp_sec

                # Enqueue for OCR only if crop is large enough to decipher (>= 400 px² and >= 30x12)
                if current_area >= 400 and cw >= 30 and ch >= 12:
                    if self.async_ocr and self._ocr_queue is not None:
                        try:
                            self._ocr_queue.put_nowait((track_id, plate_crop.copy(), frame_idx))
                        except queue.Full:
                            pass
                    else:
                        bn_text, en_text, conf = self.ocr.recognize(plate_crop)
                        with self._lock:
                            record.readings_count += 1
                            if conf > record.best_ocr_conf and conf >= self.min_ocr_conf:
                                record.best_plate_text = bn_text
                                record.best_plate_text_en = en_text
                                record.best_ocr_conf = max(conf, record.best_ocr_conf)
                            elif bn_text and not record.best_plate_text:
                                record.best_plate_text = bn_text
                                record.best_plate_text_en = en_text
                                record.best_ocr_conf = max(conf, record.best_ocr_conf)

            with self._lock:
                det.plate_text = record.best_plate_text
                det.plate_text_en = record.best_plate_text_en
                det.ocr_confidence = record.best_ocr_conf

            best_det = det

        return best_det

    def flush(self, timeout: float = 3.0) -> None:
        """Waits for pending OCR tasks to finish processing."""
        if self._ocr_queue is not None:
            end_t = time.time() + timeout
            while not self._ocr_queue.empty() and time.time() < end_t:
                time.sleep(0.05)

    def shutdown(self, wait_seconds: float = 2.0) -> None:
        """Stops the asynchronous OCR background worker gracefully."""
        if self.async_ocr and self._worker_thread is not None:
            self.flush(timeout=wait_seconds)
            if self._stop_event is not None:
                self._stop_event.set()
            try:
                if self._ocr_queue is not None:
                    self._ocr_queue.put_nowait(None)
            except Exception:
                pass
            self._worker_thread.join(timeout=1.0)

    def get_clearest_plate_crop(self, track_id: int) -> Optional[np.ndarray]:
        """Returns the sharpest, highest-clarity enhanced plate snapshot available for track."""
        with self._lock:
            record = self.records.get(track_id)
            if not record:
                return None
            return record.best_plate_crop_enhanced if record.best_plate_crop_enhanced is not None else record.best_plate_crop

    def get_record(self, track_id: int) -> Optional[TrackPlateManager.TrackRecord]:
        with self._lock:
            return self.records.get(track_id)

    def remove_track(self, track_id: int) -> Optional[TrackPlateManager.TrackRecord]:
        with self._lock:
            return self.records.pop(track_id, None)
