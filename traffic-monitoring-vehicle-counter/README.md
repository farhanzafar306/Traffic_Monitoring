# Traffic Monitoring & Vehicle Counter

Real-time vehicle detection, tracking, and line-crossing counter built on a fine-tuned YOLOv8n model. Built as part of my internship at NECOP (National Electronics Complex of Pakistan) AI lab.

![demo placeholder](docs/demo.gif)
*(replace with a short GIF or screen recording of the Gradio app or output video)*

## Overview

The system detects and tracks vehicles (car, bus, truck, motorcycle) in traffic footage, assigns persistent IDs across frames, and counts each vehicle exactly once as it crosses a defined line — a standard approach for automated traffic monitoring.

## Results

Fine-tuned YOLOv8n on a custom Roboflow vehicle dataset (Bus, Motorcycle, car, truck), 50 epochs @ 640px, trained on Colab (T4 GPU).

**Test set:**

| Metric | Score |
|---|---|
| mAP50 | 0.911 |
| mAP50-95 | 0.662 |
| Precision | 0.945 |
| Recall | 0.821 |

**Per-class mAP50:** Bus 0.889 · Motorcycle 0.969 · Car 0.920 · Truck (see full metrics in training notebook)

Inference throughput: ~2.75 FPS on CPU (Intel i5-1035G1) at 640px with tracking enabled — CPU-only, no GPU acceleration used at inference time.

## Dataset 
https://app.roboflow.com/farhan-zafar/vehicles-k83q3-lbfwf/1

## How it works

1. **Training** (`train.py` / `notebooks/training.ipynb`) — fine-tunes YOLOv8n on a custom-labeled vehicle dataset from Roboflow.
2. **Detection + Tracking** (`inference.py`) — runs `model.track()` frame-by-frame, assigns persistent object IDs, and draws bounding boxes with class + ID labels.
3. **Counting logic** — a horizontal line is defined at `line_y`. Each tracked object is counted exactly once, the first time its center point crosses below that line, using a `counted_ids` set to prevent double-counting.
4. **Live demo** (`app.py`) — a Gradio web app that streams the annotated video frame-by-frame with a live running tally, for easy demoing without needing to run scripts.

## Project structure

```
├── train.py            # YOLOv8n fine-tuning script
├── inference.py         # OpenCV-based detection, tracking, and counting
├── app.py               # Gradio web demo (live frame-by-frame streaming)
├── weights/
│   └── best.pt           # trained model weights
├── notebooks/
│   ├── training.ipynb     # original Colab training notebook
│   └── main.ipynb        # original inference notebook
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/<your-username>/traffic-monitoring-vehicle-counter.git
cd traffic-monitoring-vehicle-counter
pip install -r requirements.txt
```

## Usage

**Run on a video file (OpenCV window + saved output):**
```bash
python inference.py --source path/to/video.mp4 --weights weights/best.pt --line-y 200
```

**Run the live Gradio demo:**
```bash
python app.py
```
Then open the local URL it prints and upload a traffic video.

**Retrain on your own data:**
```bash
python train.py --data path/to/data.yaml --epochs 50 --imgsz 640
```

## Notes / limitations

- The counting line position (`line_y`) is scene-specific and needs to be tuned per camera angle.
- CPU inference is well under real-time (~2.75 FPS); a GPU is recommended for live deployment.
- Trained on a relatively small custom dataset — accuracy may drop on camera angles or lighting conditions very different from the training data.

## License

MIT
