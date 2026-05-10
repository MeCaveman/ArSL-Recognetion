# ArSL Recognition — Arabic Sign Language Recognition

A real-time Arabic Sign Language (ArSL) recognition system using **YOLO26** object detection with a **PySide6** GUI.

## Dataset

- **Classes**: 32 Arabic sign language letters
- **Images**: ~150+ photos per letter (~4,800 total)
- **Annotation**: YOLO-format bounding boxes
- **Source**: ArSL21L dataset

## Model

- **Architecture**: YOLO26s (small, ~22M params)
- **Input size**: 640×640
- **Trained for**: 50 epochs
- **Weights**: `runs/train/arsl21l/weights/best.pt`
- **Confidence threshold**: 0.45

## Features

- **Live webcam recognition** — detects signs in real-time with FPS display
- **Image upload** — detect signs from static images
- **Letter panel** — shows Arabic/English name, description, hand sign photo, and pronunciation
- **Auto-play audio** — hold a sign for 1.5 seconds to hear its pronunciation automatically, or click the play button
- **Dark UI** — modern dark-themed interface with Arabic and English text

## Project Structure

```
ArSL-Recognetion/
├── App.py                     # Main GUI application (PySide6)
├── runs/train/arsl21l/        # Trained model weights
├── assets/
│   ├── sounds/                # Pronunciation audio files (key_arsl.mp3)
│   └── fonts/                 # Inter variable font + NotoSansArabic
├── hand_signs/                 # Hand sign reference images (key_arsl.png)
├── src/
│   ├── recognition/           # Recognition utilities
│   └── ui/                    # UI components
├── config/                     # Configuration files
├── models/                     # Pre-trained YOLO models
└── pyproject.toml              # Python project config
```

## Quick Start

### 1. Install dependencies

```bash
pip install ultralytics PySide6 opencv-python
```

Or using uv (if your project uses it):

```bash
uv sync
```

### 2. Run the GUI

```bash
python App.py
```

### 3. Using the app

1. Press **▶** to start the webcam
2. Show a sign language letter to the camera
3. The right panel displays the detected letter with its details
4. Hold the sign for **1.5 seconds** to hear pronunciation automatically
5. Or click the **play button** to hear it manually
6. Use the **folder button** to upload an image instead

## Performance

| Mode | Speed | Notes |
|------|-------|-------|
| CPU (default) | ~15–30 FPS | Depends on CPU |
| GPU | ~60+ FPS | Set `DEVICE = 0` in App.py |

## Configuration

Edit `App.py` to adjust:

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `"cpu"` | Inference device (`"cpu"` or `0` for GPU) |
| `CONF` | `0.45` | Detection confidence threshold (0–1) |
| `IMG_SIZE` | `640` | Input image size |

## Adding New Letters

1. Add hand sign images to `hand_signs/` as `{key}_arsl.png`
2. Add pronunciation audio to `assets/sounds/` as `{key}_arsl.mp3`
3. Add the letter to `LETTER_DATA` in `App.py`
4. Retrain the model with the new class

## Built With

- **YOLO26** (ultralytics) — object detection
- **PySide6** — GUI framework
- **OpenCV** — camera capture & image processing
- **NotoSansArabic** — Arabic text rendering
