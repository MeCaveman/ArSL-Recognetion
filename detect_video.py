"""
To Run: python detect_video.py --video path/to/video.mp4
"""

import argparse
import os
import sys

WEIGHTS  = "runs/train/arsl21l/weights/best.pt"
IMG_SIZE = 640
CONF     = 0.4

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",   required=True)
    parser.add_argument("--weights", default=WEIGHTS)
    parser.add_argument("--conf",    default=CONF, type=float)
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Run: pip install ultralytics")
        sys.exit(1)

    if not os.path.exists(args.video):
        print(f"Video not found: {args.video}")
        sys.exit(1)
    if not os.path.exists(args.weights):
        print(f"Weights not found: {args.weights}\n   Run 1_train.py first.")
        sys.exit(1)

    model = YOLO(args.weights)

    print(f"🎬 Processing: {args.video}")
    model.predict(
        source  = args.video,
        imgsz   = IMG_SIZE,
        conf    = args.conf,
        save    = True,
        project = "runs/detect",
        name    = "video",
        stream  = True,
    )
    print("\nAnnotated video saved in runs/detect/video/")

if __name__ == "__main__":
    main()
