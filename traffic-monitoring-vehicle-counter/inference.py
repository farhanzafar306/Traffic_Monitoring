"""
Run vehicle detection, tracking, and line-crossing counting on a video file.

Usage:
    python inference.py --source path/to/video.mp4 --weights weights/best.pt --line-y 200
"""

import argparse
import cv2
from ultralytics import YOLO


def process_video(model, video_path, output_path="output.mp4", line_y=200, target_w=1280, target_h=720):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file at {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (target_w, target_h))

    counts = {cls: 0 for cls in model.names.values()}
    counted_ids = set()

    results = model.track(source=video_path, imgsz=640, persist=True, stream=True, verbose=False)

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

        cv2.imshow("Traffic Monitoring", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return output_path, counts


def main():
    parser = argparse.ArgumentParser(description="Vehicle detection, tracking, and counting")
    parser.add_argument("--source", type=str, required=True, help="Path to input video")
    parser.add_argument("--weights", type=str, default="weights/best.pt")
    parser.add_argument("--output", type=str, default="output.mp4")
    parser.add_argument("--line-y", type=int, default=200, help="Y-coordinate of the counting line")
    args = parser.parse_args()

    model = YOLO(args.weights)
    output_path, final_counts = process_video(
        model, args.source, output_path=args.output, line_y=args.line_y
    )

    print(f"\nSaved annotated video to: {output_path}")
    print("Final Vehicle Counts:", final_counts)


if __name__ == "__main__":
    main()
