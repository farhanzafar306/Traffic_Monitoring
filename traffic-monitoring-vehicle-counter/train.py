"""
Fine-tune YOLOv8n on a custom vehicle detection dataset.

Usage:
    python train.py --data path/to/data.yaml --epochs 50 --imgsz 640 --batch 16
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8n vehicle detector")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", type=str, default="traffic_custom")
    parser.add_argument("--base-model", type=str, default="yolov8n.pt")
    args = parser.parse_args()

    model = YOLO(args.base_model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
    )

    # Evaluate on the test split once training is done
    metrics = model.val(data=args.data, split="test")
    print("\n--- Test Set Results ---")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()
