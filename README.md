
# Lane Violation Detection Pipeline

This project now follows the architecture in lane_violation_pipeline_architecture.svg:

1. FrameIngester (decimation + FPS control)
2. VehicleDetector (YOLOv8n)
3. VehicleTracker (ByteTrack IDs)
4. MotionTracker (centroid history)
5. ViolationDetector (direction score + side-entry rule)
6. ViolationReporter (snapshot PNG + annotated video + CSV)

The previous c_ALPR folder is preserved as reference and the detector/tracker pattern was reused from it.

## Lane Rule

Lane polygon points:

- P1: (95, 402)
- P2: (571, 529)
- P3: (856, 192)
- P4: (658, 133)

Allowed direction: from P3-P4 side to P1-P2 side.

Violation triggers:

- Opposite-direction movement inside the lane polygon.
- Suspicious entry from the forbidden side (P1-P2 side) with no forward progress.

## Project Structure

- main.py
- pipeline.py
- config.yaml
- src/utils.py
- src/tracker.py
- src/lane_zone.py
- src/motion_tracker.py
- src/violation_detector.py
- src/violation_reporter.py

## Install

```bash
pip install -r requirements.txt
```

## Run

Default source is set in config.yaml.

```bash
python main.py
```

Override source and options:

```bash
python main.py Traffic_3_lane_cut.mp4 --save-video
python main.py 0 --no-display
python main.py --decimation 2 --fps 30
```

## Output

Generated under outputs/:

- outputs/annotated_lane_violation.mp4
- outputs/violations.csv
- outputs/snapshots/track_<id>_frame_<frame>.png

Each CSV row includes the track ID (bounding box ID), reason, direction score, and snapshot path.
