import os
import sys

# ── CONFIG ────────────────────────────────────────────────────────────────────
EPOCHS     = 100          # increase to 100 for better accuracy
BATCH_SIZE = 16          # reduce to 8 if GPU runs out of memory
IMG_SIZE   = 640         # YOLO26 default
MODEL      = "yolo26s"   # yolo26n=fastest, yolo26s=balanced, yolo26m/l/x=larger
DEVICE     = 0           # 0 = first GPU; "cpu" to force CPU
PROJECT    = "runs/train"
RUN_NAME   = "arsl21l"
# ──────────────────────────────────────────────────────────────────────────────

def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed. Run: uv add ultralytics")
        sys.exit(1)

    if not os.path.exists("data.yaml"):
        print("data.yaml not found. Make sure it's in the same folder as this script.")
        sys.exit(1)

    # Read YAML using an explicit encoding so Windows does not default to cp1252.
    try:
        with open("data.yaml", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        print("Could not read data.yaml as UTF-8.")
        print("Please re-save data.yaml using UTF-8 encoding, then run training again.")
        sys.exit(1)

    if "CHANGE THIS" in content:
        print("Please open data.yaml and set 'path' to your dataset folder first!")
        sys.exit(1)

    print(f"Starting YOLO26 training...")
    print(f"   Model:   {MODEL}")
    print(f"   Epochs:  {EPOCHS}")
    print(f"   Batch:   {BATCH_SIZE}")
    print(f"   Device:  {DEVICE}\n")

    model = YOLO(f"{MODEL}.pt")   # weights auto-downloaded on first run

    model.train(
        data     = "data.yaml",
        epochs   = EPOCHS,
        batch    = BATCH_SIZE,
        imgsz    = IMG_SIZE,
        device   = DEVICE,
        project  = PROJECT,
        name     = RUN_NAME,
        cache    = True,
        workers  = 4,
        exist_ok = True,
    )

    weights = f"{PROJECT}/{RUN_NAME}/weights/best.pt"
    print(f"\nTraining complete!")
    print(f"   Best model: {weights}")
    print(f"\n   Next step: run python app_gui.py")

if __name__ == "__main__":
    main()
