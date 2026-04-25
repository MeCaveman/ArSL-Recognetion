"""
To Run: python detect_image.py --image path/to/your/image.jpg
"""

import argparse
import os
import sys

WEIGHTS = "runs/train/arsl21l/weights/best.pt"
IMG_SIZE = 640
CONF     = 0.4

CLASSES = ['ain','al','aleff','bb','dal','dha','dhad','fa','gaaf','ghain',
           'ha','haa','jeem','kaaf','khaa','la','laam','meem','nun','ra',
           'saad','seen','sheen','ta','taa','thaa','thal','toot','waw','ya','yaa','zay']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",   required=True, help="Path to input image")
    parser.add_argument("--weights", default=WEIGHTS)
    parser.add_argument("--conf",    default=CONF, type=float)
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Run: pip install ultralytics")
        sys.exit(1)

    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        sys.exit(1)
    if not os.path.exists(args.weights):
        print(f"Weights not found: {args.weights}\n   Run 1_train.py first.")
        sys.exit(1)

    model = YOLO(args.weights)

    print(f"🔍 Detecting on: {args.image}")
    results = model.predict(
        source  = args.image,
        imgsz   = IMG_SIZE,
        conf    = args.conf,
        save    = True,
        project = "runs/detect",
        name    = "image",
    )

    for r in results:
        if len(r.boxes) == 0:
            print("   No signs detected.")
        for box in r.boxes:
            name = model.names[int(box.cls)]
            print(f"   → {name}  ({float(box.conf):.0%})")

    print("\nResult saved in runs/detect/image/")

if __name__ == "__main__":
    main()
