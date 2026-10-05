# YOLO26 Detection

Object detection experiments built on [Ultralytics](https://docs.ultralytics.com/) YOLO26. The repo has two use cases:

1. **Traffic jam detection**: a model trained on VisDrone detects vehicles in road footage and labels each frame `FREE_FLOW`, `CONGESTED` or `TRAFFIC_JAM`.
2. **Volleyball tracking**: a model trained on a Roboflow volleyball dataset detects the ball and works out when a rally is in play.

## Project structure

```
src/
├── train/
│   ├── main.py         # Train on VisDrone (resumes from latest runs/detect/train*/weights/last.pt)
│   └── volley.py       # Train on the volleyball dataset (resumes from runs/detect/volleyball)
├── validation/
│   └── volley.py       # Val + test metrics for the volleyball model
├── predict/
│   ├── main.py         # Traffic jam detection on a video
│   └── volley.py       # Ball detection + rally detection on a video
└── dataset/            # Videos and datasets (git-ignored)
runs/                   # Ultralytics training output (git-ignored)
models/                 # Exported / final weights (git-ignored)
yolo26n.pt              # Pretrained base weights
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Training picks a device automatically: CUDA GPU first, then Apple Silicon (MPS), then CPU. Batch size and worker count are set to match.

### Data

Datasets, videos and weights are git-ignored, so you need to supply them:

- **VisDrone**: Ultralytics downloads it on first run through `VisDrone.yaml`.
- **Volleyball**: [volleyball_v2 v3](https://universe.roboflow.com/shukur-sabzaliev1/volleyball_v2/dataset/3) from Roboflow (CC BY 4.0), YOLO format, 1024×768, one class (`volleyball`). Extract it to `src/dataset/Volleyball_v2.v3-1024x768.yolo26/`.
- **Videos**: put input videos in `src/dataset/`, e.g. `traffic_jakarta_footage_2.mp4` and `Volleyball_3.mp4`.

## Usage

Run every script from the repo root. Paths and parameters are set as constants at the top of each script, so edit them there.

### Traffic jam detection

```bash
python src/train/main.py      # train on VisDrone (100 epochs, imgsz 640)
python src/predict/main.py    # annotate a traffic video
```

The prediction script loads `runs/detect/train/weights/best.pt`. It only analyses vehicles in the bottom third of the frame, because perspective makes distant cars look closer together than they are. For each frame it measures:

- **Vehicle count** in the zone. Fewer than 5 vehicles is always `FREE_FLOW`.
- **Spacing**: the vertical gap between vehicles as a fraction of frame height.
- **Speed**: how far vehicles move between frames, in px/s.

| State | Rule (speed **or** spacing) | Colour |
|---|---|---|
| `TRAFFIC_JAM` | speed < 3 px/s or spacing < 8% | red |
| `CONGESTED` | speed < 12 px/s or spacing < 15% | orange |
| `FREE_FLOW` | otherwise | green |

You can tune the thresholds in `classify_traffic()`. The annotated video is written to `src/predict/VisDrone/`.

### Volleyball

```bash
python src/train/volley.py       # train (100 epochs, imgsz 1024, checkpoint every 5 epochs)
python src/validation/volley.py  # val + test set metrics
python src/predict/volley.py     # annotate a match video
```

The prediction script loads `runs/detect/volleyball/weights/best.pt` and writes:

- `src/predict/volley/results/frames/`: one annotated image per frame
- `src/predict/volley/results/video/<name>_annotated.mp4`: the annotated video

**Rally detection** counts a rally only while the ball is *moving*, so a ball that is being held or is lying still does not count:

- A rally **starts** once the ball has moved at least `RALLY_MIN_SPEED_PX` (4 px) per frame in `RALLY_START_MIN_MOVING` (5) frames within the last `RALLY_START_WINDOW_SEC` (0.5 s).
- A rally **ends** once the ball has not moved for `RALLY_END_IDLE_SEC` (1.5 s).

The rally status is drawn in the top-right corner of each frame. All of these thresholds are constants at the top of `src/predict/volley.py`.

## Outputs

Ultralytics saves training runs to `runs/detect/<name>/`. Each run holds `weights/best.pt`, `weights/last.pt`, the metric plots and `results.csv`. If a run is interrupted, re-running the training script resumes from `last.pt`.
