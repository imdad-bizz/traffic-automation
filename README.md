# Bangladeshi Traffic Monitoring & Automatic Number Plate Recognition (ANPR / ALPR)

A computer vision pipeline tailored for Bangladeshi traffic surveillance, vehicle tracking, license plate localization, Bengali/English OCR recognition, speed estimation, lane discipline enforcement, and red-light violation detection.

> 📘 **Detailed Architectural Guide**: For full architecture diagrams, mathematical formulations, database schemas, and engineering specifications, refer to [SYSTEM_ARCHITECTURE_AND_SETUP.md](SYSTEM_ARCHITECTURE_AND_SETUP.md).

---

## 🌟 Key Capabilities

1. **Vehicle Detection & Tracking**:
   - YOLOv8 vehicle detection (cars, motorbikes, buses, trucks).
   - ByteTrack multi-object tracking with persistent track IDs.
2. **Dedicated License Plate Localization**:
   - YOLO license plate model (`models/plate_detector.pt`) specifically trained to localize vehicle registration plates in high-resolution and CCTV footage.
3. **Bangladeshi Plate Character Recognition (Bengali OCR)**:
   - Customized EasyOCR model (`bn_license_tps`) trained on Bangladeshi license plate syntax (Dhaka Metro, Chittagong, Sylhet, Bengali numerals `০-৯`, class letters `ক, খ, গ, ঘ, চ, ছ`, etc.).
   - Bilingual fallback (Bengali + English alphanumeric recognition).
   - Image enhancement: multi-scale upscaling, CLAHE contrast equalization, bilateral denoising, sharpening, and perspective rectification.
4. **Speed Estimation**:
   - Pixel displacement tracking calibrated to camera geometry with customizable speed limits.
5. **Violation Enforcement**:
   - **Lane Discipline**: Detects wrong-way driving and forbidden-side lane entries.
   - **Traffic Light Enforcement**: Detects red-light running with dynamic stop-strip crossing detection.
6. **Multi-Channel Evidence Logging**:
   - Annotated HD video output with vehicle boxes and live Picture-in-Picture (PiP) plate preview.
   - High-resolution snapshots featuring the violating vehicle and an inset zoomed crop of the license plate with recognized text.
   - Real-time logging to both CSV files and SQLite databases (`outputs/detections.db`, `outputs/violations.db`).

---

## 📁 Repository Structure

```text
├── config.yaml                       # Master configuration (paths, models, thresholds)
├── main.py                           # Unified CLI entry point
├── plate_speed_pipeline.py           # ALPR & speed estimation pipeline
├── pipeline.py                       # Lane violation pipeline
├── traffic_light_violation.py        # Red-light violation pipeline
├── requirements.txt                  # Python dependencies
├── cctv_footage.mp4                  # Sample surveillance footage
├── yolov8n.pt                        # Base vehicle detection weights
├── models/
│   ├── plate_detector.pt             # Trained YOLO license plate detector
│   └── EasyOCR/                      # Specialized Bangladeshi OCR models
│       ├── models/bn_license_tps.pth # Bengali plate recognition network
│       └── user_network/             # Custom network architecture & vocabulary
└── src/
    ├── plate_detector.py             # LicensePlateDetector, BanglaPlateOCR, TrackPlateManager
    ├── tracker.py                    # Vehicle detector & ByteTrack tracker
    ├── motion_tracker.py             # Centroid history & trajectory tracking
    ├── lane_zone.py                  # Lane polygon geometry & coordinate mapping
    ├── violation_detector.py         # Directional vector & speed violation logic
    ├── violation_reporter.py         # ALPR evidence snapshots, CSV & SQLite logging
    └── utils.py                      # Frame ingestion, Unicode Bengali rendering, HUD
```

---

## 🚀 Installation

Ensure you have Python 3.10+ installed.

```bash
pip install -r requirements.txt
```

---

## 💻 How to Run

### 1. Plate Recognition & Speed Monitoring (Default)

Run on the default video (`cctv_footage.mp4`) configured in `config.yaml`:

```bash
python main.py
```

Override with any video footage or camera stream:

```bash
# Run on custom footage with live display
python main.py path/to/your_footage.mp4

# Run in headless mode (no GUI window)
python main.py path/to/your_footage.mp4 --no-display

# Test only the first 300 frames
python main.py path/to/your_footage.mp4 --max-frames 300 --no-display
```

### 2. Lane Discipline & Wrong-Way Monitoring (with ALPR)

```bash
python main.py --mode lane
```

### 3. Traffic Light / Red-Light Violation Monitoring (with ALPR)

```bash
python main.py --mode traffic_light
```

---

### 4. Offline High-Accuracy AI Post-Processing (OpenAI GPT-4o Vision)

For maximum accuracy on challenging, blurred, or degraded license plates, process harvested clear snapshots offline using OpenAI GPT-4o Vision:

```bash
# Run local neural OCR post-processing and generate the executive Excel workbook
python process_crops.py

# Run with OpenAI GPT-4o Vision (pass key directly)
python process_crops.py --gpt --api-key "your_openai_api_key_here"

# Or set your environment variable or config.yaml:
export OPENAI_API_KEY="your_api_key"
python process_crops.py --gpt
```

---

## 📊 Outputs & Evidence Logs

All outputs are saved in the `outputs/` directory:

- **Executive ALPR & Violation Excel Report**: [`outputs/traffic_violations_alpr.xlsx`](file:///c:/Users/Hp/Desktop/traffic-automation/outputs/traffic_violations_alpr.xlsx)
  - Executive KPI summary cards (Total Vehicles, Active Violations, Compliance Rate, Speed Limits, PC Generation Time).
  - 18 relational columns including PC clock date/time, vehicle class, speed metrics, BRTA Bengali plate, English transliteration, OCR confidence, and violation status.
  - **Embedded Visual Image Thumbnails** for both the clear plate crop and the full vehicle scene.
  - **Clickable Hyperlinks** directly opening high-resolution snapshot files.
- **Annotated Full Footage**: [`outputs/annotated_plate_speed.mp4`](file:///c:/Users/Hp/Desktop/traffic-automation/outputs/annotated_plate_speed.mp4) (Complete surveillance video with bilingual HUD, PiP plate preview, audio alert on violations, and speed labels).
- **Clearest Plate Crops**: `outputs/plate_crops/track_<id>_plate.png` (High-resolution, contrast-enhanced plate crops harvested at closest optical approach).
- **Composite Evidence Snapshots**: `outputs/plate_snapshots/track_<id>_frame_<frame>.png` (Vehicle context + zoomed license plate inset + speed banner).
- **CSV Data Log**: [`outputs/detections.csv`](file:///c:/Users/Hp/Desktop/traffic-automation/outputs/detections.csv) (Includes PC Date and PC Time columns).
- **SQLite Database**: [`outputs/detections.db`](file:///c:/Users/Hp/Desktop/traffic-automation/outputs/detections.db) (Structured relational database synchronized with all detections and violations).

---

## ⚙️ Key Enhancements

1. **Clear Number Plate Snapshots**: Dynamically evaluates plate resolution and sharpness across vehicle tracks, retaining the highest-resolution snapshot when the vehicle is closest to the camera. Enhances crops with bicubic upscaling, bilateral denoising, and CLAHE contrast equalization.
2. **PC Clock Time Logging**: Automatically stamps the host computer's real-time clock (`YYYY-MM-DD` and `HH:MM:SS`) on all violation records, snapshots, CSV logs, SQLite entries, and Excel rows.
3. **OpenAI GPT-4o Vision Post-Processing**: Offline vision AI module specialized in Bangladesh BRTA vehicle registration formats (metro names, class letters, and Bengali digits). Activates seamlessly when an API key is provided via CLI, environment variable, or `config.yaml`.
4. **Relational Excel Reporting**: Auto-generates styled Excel workbooks with embedded thumbnails, direct links to evidence files, conditional formatting, and executive KPI cards.
5. **Violation Audio & Visual Alerts**: Triggers a non-blocking audio beep alert (`winsound.Beep` on Windows in a daemon thread) and displays a prominent on-screen HUD warning banner whenever a speeding, lane, or red-light violation occurs.
6. **Finetuned Speeding Validation**: Recalibrated homography IPM parameters (30m road stretch), horizon jitter filtering ($y < 260$), median trajectory speed calculation, and a configurable 5 km/h tolerance buffer to prevent false-positive speeding flags on normal traffic.

