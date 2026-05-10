"""
To Run: python detect_webcam.py
Press Q to quit.
"""

import sys
import os
import time
import cv2

WEIGHTS  = "runs/train/arsl21l/weights/best.pt"
IMG_SIZE = 640
CONF     = 0.45

def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Run: pip install ultralytics")
        sys.exit(1)

    if not os.path.exists(WEIGHTS):
        print(f"Weights not found: {WEIGHTS}\n   Run train.py first.")
        sys.exit(1)

    model = YOLO(WEIGHTS)
    cap   = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam. Try changing VideoCapture(0) to (1).")
        sys.exit(1)

    print("Webcam running — press Q to quit")
    prev = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results   = model.predict(frame, imgsz=IMG_SIZE, conf=CONF, verbose=False)
        annotated = results[0].plot()

        now = time.time()
        fps = 1 / (now - prev + 1e-6)
        prev = now
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 229, 170), 2)

        cv2.imshow("ArSL Recognition (YOLO26) — Q to quit", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
