# Face-YOLO

Face detection built with **YOLOv8** (Ultralytics). Includes training/inference scripts and an interactive **Streamlit** demo supporting image, video, and camera input, plus a dataset management tab for adding new data and retraining.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-orange)
![Streamlit](https://img.shields.io/badge/Demo-Streamlit-red)

<img width="1920" height="1920" alt="train_batch0" src="https://github.com/user-attachments/assets/635abb0d-1fea-4e29-b224-18735733a3c9" />


## Features

- Detects faces in images, videos, and browser camera snapshots
- Draws bounding boxes with class labels (standard object detection output)
- Interactive demo app with configurable confidence/IoU thresholds, image size, and device (GPU/CPU, auto-detected)
- Dataset tab: validate and merge externally labeled data into the existing train/val split, keeping the YOLOv8 folder structure intact
- One-click retraining from the demo app

## Project structure

```
Face-YOLO/
├── app.py                          # Streamlit demo (test + data/train tabs)
├── yolo_utils.py                   # Inference and dataset utility functions
├── Train.py                        # Standalone training script
├── Test.py                         # Standalone batch inference script
├── requirements.txt
├── yolov8n.pt                      # Base pretrained weights (nano)
├── Face/
│   ├── data.yaml                   # Dataset config (classes, split paths)
│   ├── train/{images,labels}/
│   ├── valid/{images,labels}/
│   └── test/images/
└── runs/detect/train/weights/
    ├── best.pt                     # Trained weights (best mAP)
    └── last.pt
```

> Note: `Face/*/images`, `Face/*/labels`, and `runs/` are excluded from this repo via `.gitignore` (dataset images and training logs are too large for GitHub). Only `data.yaml` and `best.pt` are tracked. See [Dataset](#dataset) below for how to get the full data.

## Setup

```bash
git clone <your-repo-url>
cd Face-YOLO

python -m venv YOLO_venv
YOLO_venv\Scripts\activate        # Windows
# source YOLO_venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

Requires a CUDA-capable GPU for fast training/inference (scripts default to `device=0`); the demo app auto-detects GPU availability and falls back to CPU if unavailable.

## Dataset

Format: standard YOLOv8 detection dataset (`images/` + matching `labels/*.txt`, one `.txt` per image with `class_id x_center y_center width height` normalized to [0,1]).

`data.yaml` is included in this repo. Download the full image set here:

```
<!-- TODO: add your dataset source link, e.g. Roboflow Universe / Kaggle / Google Drive -->
```

After downloading, place the images/labels under `Face/` matching the structure above.

## Usage

**Train:**
```bash
python Train.py
```

**Batch inference on test set:**
```bash
python Test.py
```

**Real-time inference via CLI:**
```bash
yolo predict model=runs/detect/train/weights/best.pt source=0 imgsz=416 device=0 show=True
```

**Interactive demo app:**
```bash
streamlit run app.py
```
- **Test tab** — upload an image or video, or take a snapshot through your browser camera
- **Data & Train tab** — view current dataset stats, upload a `.zip` of new labeled images to merge into the dataset, and trigger retraining

## Model

Base model: `yolov8n.pt` (YOLOv8 nano — fast, lightweight, suited for real-time inference on modest hardware). Fine-tuned for face detection; see `Face/data.yaml` for the full class list.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Dataset: <!-- TODO: credit dataset source -->
