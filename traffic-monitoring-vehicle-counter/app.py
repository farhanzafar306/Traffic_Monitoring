"""
Gradio web demo: upload a traffic video and see live detection, tracking,
and vehicle counts streamed frame-by-frame.

Usage:
    python app.py
"""

import os
import cv2
import gradio as gr
from ultralytics import YOLO

WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", "weights/best.pt")
model = YOLO(WEIGHTS_PATH)


def stream_video_gradio(video_path, line_y=200):
    if not video_path or not os.path.exists(video_path):
        yield None, "Error: Invalid video file."
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        yield None, "Error: Could not open video stream."
        return

    target_w, target_h = 1280, 720
    counts = {cls: 0 for cls in model.names.values()}
    counted_ids = set()

    results = model.track(
        source=video_path, imgsz=640, conf=0.15, persist=True, stream=True, verbose=False
    )

    for r in results:
        frame = cv2.resize(r.orig_img, (target_w, target_h))
        cv2.line(frame, (0, line_y), (target_w, line_y), (0, 255, 255), 2)

        if r.boxes is not None and r.boxes.id is not None:
            orig_h, orig_w = r.orig_img.shape[:2]
            scale_x, scale_y = target_w / orig_w, target_h / orig_h

            for box, track_id, cls_id in zip(r.boxes.xyxy, r.boxes.id, r.boxes.cls):
                x1, y1, x2, y2 = box.tolist()
                x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                y1, y2 = int(y1 * scale_y), int(y2 * scale_y)

                cls_name = model.names[int(cls_id)]
                uid = int(track_id)
                cy = (y1 + y2) // 2

                if uid not in counted_ids and cy > line_y:
                    counted_ids.add(uid)
                    counts[cls_name] += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(frame, f"{cls_name} #{uid}", (x1, max(y1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)

        y_offset = 30
        for cls, n in counts.items():
            cv2.putText(frame, f"{cls}: {n}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 25

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        summary = "--- LIVE VEHICLE COUNT ---\n" + "\n".join(
            f"{k.capitalize()}: {v}" for k, v in counts.items()
        )

        yield frame_rgb, summary

    cap.release()


demo = gr.Interface(
    fn=stream_video_gradio,
    inputs=gr.Video(label="Upload Traffic Video"),
    outputs=[
        gr.Image(label="Real-Time Detection Feed"),
        gr.Textbox(label="Live Count Tally"),
    ],
    title="Real-Time Traffic Analytics",
    description="Upload a video to see live bounding boxes, object tracking, and line counts updating frame-by-frame.",
)

if __name__ == "__main__":
    demo.launch()
