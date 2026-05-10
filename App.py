"""
ArSL Recognition — Main GUI Application (Integrated)
Drop hand signs → hand_signs/{key}_arsl.png
Drop sounds    → sounds/{key}_arsl.wav  (or .mp3)
Drop logo      → logo.png / logo.svg
"""

import sys
import os
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.multimedia*=false"
os.environ["AV_LOG_LEVEL"] = "quiet"
import time
import cv2
import urllib.request
import zipfile
import io

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QDialog,
    QFrame, QSizePolicy, QSlider, QStackedWidget,
    QGraphicsOpacityEffect, QMessageBox, QComboBox,
)
from PySide6.QtCore import (
    Qt, QThread, Signal, QTimer, QPropertyAnimation, QEasingCurve,
    QRect, QRectF, QUrl,
)
from PySide6.QtGui import (
    QFont, QPixmap, QColor, QPainter, QPen, QImage,
    QFontMetrics, QRadialGradient, QPainterPath, QPalette,
    QFontDatabase,
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

# ─── Config ───────────────────────────────────────────────────────────────────
WEIGHTS        = "runs/train/arsl21l/weights/best.pt"
IMG_SIZE       = 640
CONF           = 0.45
DEVICE         = "cpu"
HAND_SIGNS_DIR = "hand_signs"
SOUNDS_DIR     = "assets/sounds"
FONTS_DIR      = "assets/fonts"
MONO_FONT      = "JetBrains Mono"
MONO_FALLBACK  = "Consolas"
SANS_FONT      = "Segoe UI"   # updated at runtime after Google Sans is loaded
ARABIC_FONT    = "Noto Sans Arabic"

# ─── Palette ──────────────────────────────────────────────────────────────────
BG          = "#0a0a0a"
SURFACE     = "#111111"
SURFACE2    = "#171717"
SURFACE3    = "#1e1e1e"
ACCENT      = "#00e5aa"
ACCENT_DIM  = "#00a877"
ACCENT_GLOW = "#00e5aa33"
ACCENT_BG   = "#00e5aa11"
RED         = "#e55353"
TEXT        = "#f0f0f0"
TEXT_DIM    = "#888888"
TEXT_MUTED  = "#3a3a3a"
BORDER      = "#252525"
BORDER2     = "#1e1e1e"

# ─── Letter data (32 ArSL21L classes, keys match YOLO class names) ────────────
LETTER_DATA = {
    "ain":   {"arabic":"ع",  "name_en":"Ain",   "name_ar":"العين",         "phonetic":"Ai – n",                   "desc_en":"Curve the index finger and thumb together\nto form a C-shape.",                    "desc_ar":"تقويس السبابة والإبهام معاً لتشكيل حرف C"},
    "al":    {"arabic":"ال", "name_en":"Al",    "name_ar":"ال",             "phonetic":"A – l",                    "desc_en":"Point the index and middle fingers\nstraight downward side-by-side.",           "desc_ar":"مد السبابة والوسطى للأسفل معاً بشكل مستقيم"},
    "aleff": {"arabic":"ا",  "name_en":"Alef",  "name_ar":"الألف",          "phonetic":"A – lif",                  "desc_en":"Close all fingers and extend the\nindex finger straight upward.",               "desc_ar":"إغلاق جميع الأصابع مع رفع السبابة فقط وتوجيهها للأعلى"},
    "bb":    {"arabic":"ب",  "name_en":"Ba",    "name_ar":"الباء",          "phonetic":"B – aa",                   "desc_en":"Extend the index finger horizontally\nwith the thumb pointing downward.",           "desc_ar":"مد السبابة بشكل أفقي مع توجيه الإبهام للأسفل لإظهار النقطة وإغلاق باقي الأصابع"},
    "dal":   {"arabic":"د",  "name_en":"Dal",   "name_ar":"الدال",          "phonetic":"D – aal",                  "desc_en":"Curve the index finger and thumb\nto form a half-circle.",                       "desc_ar":"تقويس السبابة والإبهام لتشكيل نصف دائرة مع إغلاق باقي الأصابع"},
    "dha":   {"arabic":"ظ",  "name_en":"Dha",   "name_ar":"الظاء",          "phonetic":"Zh – aa",                  "desc_en":"Form the Taa shape and apply\na slight upward movement.",                       "desc_ar":"مد السبابة للأمام وطي باقي الأصابع مع تحريك اليد للأعلى قليلاً"},
    "dhad":  {"arabic":"ض",  "name_en":"Dhad",  "name_ar":"الضاد",          "phonetic":"D – aad",                  "desc_en":"Form the Saad shape and apply\na slight outward movement.",                      "desc_ar":"إغلاق قبضة اليد على شكل حلقة مع تحريكها قليلاً للخارج"},
    "fa":    {"arabic":"ف",  "name_en":"Fa",    "name_ar":"الفاء",          "phonetic":"F – aa",                   "desc_en":"Form a circle with the thumb and index finger\nwhile keeping the middle finger raised.",      "desc_ar":"تكوين حلقة بالسبابة والإبهام مع رفع الوسطى للأعلى وطي باقي الأصابع"},
    "gaaf":  {"arabic":"ق",  "name_en":"Qaf",   "name_ar":"القاف",          "phonetic":"Q – aaf",                  "desc_en":"Form a circle with the thumb and index finger\nwhile keeping both the middle and ring fingers raised.", "desc_ar":"تكوين حلقة بالسبابة والإبهام مع رفع الوسطى والبنصر للأعلى"},
    "ghain": {"arabic":"غ",  "name_en":"Ghain", "name_ar":"الغين",          "phonetic":"Gh – ain",                 "desc_en":"Curve the hand with\na slight wrist rotation.",                          "desc_ar":"تقويس اليد لتشكيل قوس مع حركة دائرية بسيطة في المعصم"},
    "ha":    {"arabic":"ه",  "name_en":"Ha",    "name_ar":"الهاء",          "phonetic":"H – aa",                   "desc_en":"Bring the tips of all five fingers\ntogether to form a teardrop shape.",             "desc_ar":"جمع أطراف الأصابع الخمسة لتتلامس معاً مشكلة دائرة أو قطرة"},
    "haa":   {"arabic":"ح",  "name_en":"Haa",   "name_ar":"الحاء",          "phonetic":"H – aa",                   "desc_en":"Curve the hand downward with\nthe thumb extended outward.",                     "desc_ar":"تقويس اليد للأسفل مع إبقاء الإبهام ممدوداً للخارج"},
    "jeem":  {"arabic":"ج",  "name_en":"Jeem",  "name_ar":"الجيم",          "phonetic":"J – eem",                  "desc_en":"Curve the hand downward with\nthe thumb tucked inside.",                        "desc_ar":"تقويس اليد للأسفل مع إدخال الإبهام بداخلها"},
    "kaaf":  {"arabic":"ك",  "name_en":"Kaaf",  "name_ar":"الكاف",          "phonetic":"K – aaf",                  "desc_en":"Open the hand and slightly curve the fingers\nto form a forward-facing semi-cup shape.", "desc_ar":"فتح اليد وتقويس الأصابع قليلاً للأمام لتشكل نصف كوب"},
    "khaa":  {"arabic":"خ",  "name_en":"Khaa",  "name_ar":"الخاء",          "phonetic":"Kh – aa",                  "desc_en":"Curve the hand downward with\nthe thumb pointing straight upward.",               "desc_ar":"تقويس اليد للأسفل مع توجيه الإبهام للأعلى"},
    "la":    {"arabic":"لا", "name_en":"La",    "name_ar":"لا",             "phonetic":"L – aa",                   "desc_en":"Lam-Alef ligature.\nL-shape formed with the thumb and index finger.",          "desc_ar":"لاَم أَلِف.\nتشكيل حرف L بالإبهام والسبابة."},
    "laam":  {"arabic":"ل",  "name_en":"Lam",   "name_ar":"اللام",          "phonetic":"L – aam",                  "desc_en":"Extend the thumb and index finger downward\nto form an inverted L-shape.",               "desc_ar":"مد السبابة والإبهام للأسفل وتشكيل زاوية قائمة مائلة"},
    "meem":  {"arabic":"م",  "name_en":"Meem",  "name_ar":"الميم",          "phonetic":"M – eem",                  "desc_en":"Close the hand into a fist and point\nthe index finger straight downward.",             "desc_ar":"إغلاق قبضة اليد بالكامل وتوجيه السبابة بشكل مستقيم للأسفل"},
    "nun":   {"arabic":"ن",  "name_en":"Nun",   "name_ar":"النون",          "phonetic":"N – oon",                  "desc_en":"Form an upward-facing semi-circle cup shape\nand point the thumb downward into the cup.", "desc_ar":"تقويس الأصابع للأعلى مع توجيه الإبهام لأسفل داخلها"},
    "ra":    {"arabic":"ر",  "name_en":"Ra",    "name_ar":"الراء",          "phonetic":"R – aa",                   "desc_en":"Point the index and middle fingers\ndiagonally downward and cross them.",              "desc_ar":"تقاطع السبابة والوسطى مع توجيههما للأسفل مائلاً"},
    "saad":  {"arabic":"ص",  "name_en":"Saad",  "name_ar":"الصاد",          "phonetic":"S – aad",                  "desc_en":"Close the fist and rest the index finger\nover the thumb to form a closed loop.",         "desc_ar":"إغلاق قبضة اليد وتمرير السبابة فوق الإبهام لتشكيل حلقة مغلقة"},
    "seen":  {"arabic":"س",  "name_en":"Seen",  "name_ar":"السين",          "phonetic":"S – een",                  "desc_en":"Extend the index, middle, and ring\nfingers straight upward.",                         "desc_ar":"رفع السبابة والوسطى والبنصر للأعلى مع إغلاق باقي الأصابع"},
    "sheen": {"arabic":"ش",  "name_en":"Sheen", "name_ar":"الشين",          "phonetic":"Sh – een",                 "desc_en":"Extend the index, middle, and ring fingers\nstraight upward with a slight side-to-side wrist movement.", "desc_ar":"رفع السبابة والوسطى والبنصر للأعلى مع تحريك اليد من اليمين لليسار"},
    "ta":    {"arabic":"ت",  "name_en":"Ta",    "name_ar":"التاء",          "phonetic":"T – aa",                   "desc_en":"Extend both the index and\nmiddle fingers horizontally.",                          "desc_ar":"مد السبابة والوسطى معاً بشكل أفقي مع إغلاق باقي الأصابع"},
    "taa":   {"arabic":"ط",  "name_en":"Taa",   "name_ar":"الطاء",          "phonetic":"T – aa",                   "desc_en":"Extend the index finger straight forward\nwhile curling the remaining fingers.",           "desc_ar":"مد السبابة للأمام بشكل مستقيم وطي باقي الأصابع بحيث يلامس الإبهام الوسطى"},
    "thaa":  {"arabic":"ث",  "name_en":"Thaa",  "name_ar":"الثاء",          "phonetic":"Th – aa",                  "desc_en":"Extend the index, middle, and ring\nfingers horizontally.",                          "desc_ar":"مد السبابة والوسطى والبنصر بشكل أفقي مع إغلاق باقي الأصابع"},
    "thal":  {"arabic":"ذ",  "name_en":"Thal",  "name_ar":"الذال",          "phonetic":"Th – aal",                 "desc_en":"Curve the index finger and thumb into\na half-circle with a slight shaking movement.",     "desc_ar":"تقويس السبابة والإبهام لتشكيل نصف دائرة مع تحريك اليد قليلاً"},
    "toot":  {"arabic":"ة",  "name_en":"Toot",  "name_ar":"التاء المربوطة", "phonetic":"T – oot",                  "desc_en":"Form the Haa shape and apply\na slight shaking movement.",                          "desc_ar":"جمع أطراف الأصابع الخمسة لتتلامس معاً مع تحريك اليد قليلاً"},
    "waw":   {"arabic":"و",  "name_en":"Waw",   "name_ar":"الواو",          "phonetic":"W – aaw",                  "desc_en":"Form a circle with the thumb and index finger\nwhile extending the remaining three fingers straight upward.", "desc_ar":"تكوين دائرة بالسبابة والإبهام مع مد الأصابع الثلاثة الأخرى للأعلى"},
    "ya":    {"arabic":"ي",  "name_en":"Ya",    "name_ar":"الياء",          "phonetic":"Y – aa",                   "desc_en":"Close the fist and extend\nonly the pinky finger straight upward.",                 "desc_ar":"إغلاق قبضة اليد ورفع الخنصر فقط للأعلى"},
    "yaa":   {"arabic":"ى",  "name_en":"Yaa",   "name_ar":"الألف المقصورة", "phonetic":"Y – aa",                   "desc_en":"Close the fist and extend the pinky\nfinger, moving it slightly downward.",              "desc_ar":"إغلاق قبضة اليد ورفع الخنصر مع حركة خفيفة للأسفل"},
    "zay":   {"arabic":"ز",  "name_en":"Zay",   "name_ar":"الزاي",          "phonetic":"Z – aa",                   "desc_en":"Point the index and middle fingers diagonally\ndownward, crossing them with a slight shaking movement.", "desc_ar":"تقاطع السبابة والوسطى للأسفل مائلاً مع تحريك اليد قليلاً"},
}


# ─── Font setup ───────────────────────────────────────────────────────────────
def _download_font_zip(family: str):
    os.makedirs(FONTS_DIR, exist_ok=True)
    url = f"https://fonts.google.com/download?family={family.replace(' ', '+')}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for name in zf.namelist():
                if name.lower().endswith(".ttf") or name.lower().endswith(".otf"):
                    dest = os.path.join(FONTS_DIR, os.path.basename(name))
                    with open(dest, "wb") as f:
                        f.write(zf.read(name))
    except Exception as e:
        print(f"[Font] Could not download {family}: {e}")


def _load_or_download(subfamily: str, fallback: str) -> str:
    os.makedirs(FONTS_DIR, exist_ok=True)
    key = subfamily.lower().replace(" ", "")
    cached = [
        f for f in os.listdir(FONTS_DIR)
        if key in f.lower().replace(" ", "").replace("_", "").replace("-", "")
        and (f.lower().endswith(".ttf") or f.lower().endswith(".otf"))
    ]
    if not cached:
        _download_font_zip(subfamily)
        cached = [
            f for f in os.listdir(FONTS_DIR)
            if key in f.lower().replace(" ", "").replace("_", "").replace("-", "")
            and (f.lower().endswith(".ttf") or f.lower().endswith(".otf"))
        ]
    if cached:
        for fname in cached:
            fid = QFontDatabase.addApplicationFont(os.path.join(FONTS_DIR, fname))
            if fid != -1:
                families = QFontDatabase.applicationFontFamilies(fid)
                if families:
                    return families[0]
    return fallback


# Material Icons codepoints used in this UI
MI_FOLDER   = ""   # folder_open
MI_SETTINGS = ""   # settings
MI_PLAY     = ""   # play_arrow
MI_PAUSE    = ""   # pause
MI_STOP     = ""   # stop

ICON_FONT = "Material Icons"   # updated after loading


def mono(size: int, bold: bool = False) -> QFont:
    f = QFont(MONO_FONT, size, QFont.Bold if bold else QFont.Normal)
    f.setStyleHint(QFont.Monospace)
    return f


def gsans(size: int, bold: bool = False) -> QFont:
    f = QFont(SANS_FONT, size, QFont.Bold if bold else QFont.Normal)
    f.setStyleHint(QFont.SansSerif)
    return f


def micon(size: int) -> QFont:
    """Material Icons font for icon glyphs."""
    f = QFont(ICON_FONT, size)
    f.setStyleHint(QFont.AnyStyle)
    return f


# ─── Helpers ──────────────────────────────────────────────────────────────────
def load_hand_photo(key: str, size: int = 110) -> QPixmap:
    path = os.path.join(HAND_SIGNS_DIR, f"{key}_arsl.png")
    if os.path.exists(path):
        px = QPixmap(path)
        if not px.isNull():
            side = min(px.width(), px.height())
            x = (px.width() - side) // 2
            y = (px.height() - side) // 2
            px = px.copy(x, y, side, side)
            return px.scaled(size, size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
    return _make_hand_placeholder(size)


def find_sound_path(key: str) -> str | None:
    for ext in ("wav", "mp3", "ogg"):
        p = os.path.join(SOUNDS_DIR, f"{key}_arsl.{ext}")
        if os.path.exists(p):
            return os.path.abspath(p)
    return None


def _make_letter_pixmap(letter: str, size: int = 116) -> QPixmap:
    px = QPixmap(size, size)
    px.fill(Qt.transparent)
    p = QPainter(px)
    p.setRenderHint(QPainter.Antialiasing)
    path = QPainterPath()
    path.addRoundedRect(0, 0, size, size, 16, 16)
    p.fillPath(path, QColor("#181818"))
    p.setPen(QPen(QColor(ACCENT_DIM), 1.5))
    p.drawPath(path)
    p.setFont(QFont("Arial", int(size * 0.40), QFont.Bold))
    p.setPen(QColor(ACCENT))
    p.drawText(QRect(0, 0, size, size), Qt.AlignCenter, letter)
    p.end()
    return px


def _make_hand_placeholder(size: int = 110) -> QPixmap:
    px = QPixmap(size, size)
    px.fill(Qt.transparent)
    p = QPainter(px)
    p.fillRect(0, 0, size, size, QColor(SURFACE3))
    p.setPen(QPen(QColor(TEXT_MUTED), 1))
    p.drawRect(0, 0, size - 1, size - 1)
    p.setFont(gsans(7))
    p.setPen(QColor(TEXT_MUTED))
    p.drawText(QRect(0, 0, size, size), Qt.AlignCenter, "no photo")
    p.end()
    return px


def make_webcam_placeholder(w: int = 700, h: int = 480) -> QPixmap:
    px = QPixmap(w, h)
    px.fill(QColor(BG))
    p = QPainter(px)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(QPen(QColor("#161616"), 1))
    for x in range(0, w, 40):
        p.drawLine(x, 0, x, h)
    for y in range(0, h, 40):
        p.drawLine(0, y, w, y)
    cx, cy, r = w // 2, h // 2, 48
    glow = QRadialGradient(cx, cy, r * 2)
    glow.setColorAt(0, QColor(ACCENT_GLOW))
    glow.setColorAt(1, Qt.transparent)
    p.fillRect(cx - r * 2, cy - r * 2, r * 4, r * 4, glow)
    p.setPen(QPen(QColor(ACCENT_DIM), 2, Qt.DashLine))
    p.drawEllipse(cx - r, cy - r, r * 2, r * 2)
    p.setFont(gsans(12))
    p.setPen(QColor(TEXT_MUTED))
    p.drawText(QRect(0, cy + r + 16, w, 30), Qt.AlignCenter, "Press  ▶  to start webcam")
    p.end()
    return px


def _divider_h() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setFixedHeight(1)
    f.setStyleSheet(f"background: {BORDER}; border: none;")
    return f


# ─── Inference threads ────────────────────────────────────────────────────────
class WebcamThread(QThread):
    frame_ready = Signal(QPixmap)
    detection   = Signal(str)
    fps_signal  = Signal(float)
    error       = Signal(str)

    def __init__(self, model, cam_index: int, device, conf, img_size, parent=None):
        super().__init__(parent)
        self._model     = model
        self._cam_index = cam_index
        self._device    = device
        self._conf      = conf
        self._img_size  = img_size
        self._running   = False

    def run(self):
        self._running = True
        cap = cv2.VideoCapture(self._cam_index)
        if not cap.isOpened():
            self.error.emit(f"Cannot open camera {self._cam_index}")
            return
        prev = time.time()
        while self._running:
            ret, frame = cap.read()
            if not ret:
                break
            results   = self._model.predict(frame, imgsz=self._img_size,
                                            conf=self._conf, device=self._device,
                                            verbose=False)
            annotated = results[0].plot()
            now  = time.time()
            fps  = 1.0 / (now - prev + 1e-6)
            prev = now

            rgb  = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self.frame_ready.emit(QPixmap.fromImage(qimg))
            self.fps_signal.emit(fps)

            if results[0].boxes:
                top = max(results[0].boxes, key=lambda b: float(b.conf))
                self.detection.emit(self._model.names[int(top.cls)])
            else:
                self.detection.emit("")
        cap.release()

    def stop(self):
        self._running = False
        self.wait(3000)


class ImageInferenceThread(QThread):
    done  = Signal(QPixmap, str)
    error = Signal(str)

    def __init__(self, model, path, device, conf, img_size, parent=None):
        super().__init__(parent)
        self._model    = model
        self._path     = path
        self._device   = device
        self._conf     = conf
        self._img_size = img_size

    def run(self):
        try:
            results   = self._model.predict(self._path, imgsz=self._img_size,
                                            conf=self._conf, device=self._device,
                                            verbose=False)
            annotated = results[0].plot()
            rgb  = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            px   = QPixmap.fromImage(qimg)
            key  = ""
            if results[0].boxes:
                top = max(results[0].boxes, key=lambda b: float(b.conf))
                key = self._model.names[int(top.cls)]
            self.done.emit(px, key)
        except Exception as e:
            self.error.emit(str(e))


# ─── Settings dialog ──────────────────────────────────────────────────────────
class SettingsDialog(QDialog):
    def __init__(self, current_cam: int = 0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setFixedSize(480, 520)
        self.setStyleSheet(f"""
            QDialog {{ background: {SURFACE}; border: 1px solid {BORDER}; }}
            QLabel  {{ color: {TEXT}; background: transparent; }}
        """)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(32, 32, 32, 32)
        lay.setSpacing(0)

        # Title
        title = QLabel("Settings")
        title.setFont(gsans(18, bold=True))
        title.setStyleSheet(f"color: {ACCENT};")
        lay.addWidget(title)
        lay.addSpacing(6)

        subtitle = QLabel("Configure camera and view application info")
        subtitle.setFont(gsans(9))
        subtitle.setStyleSheet(f"color: {TEXT_DIM};")
        lay.addWidget(subtitle)
        lay.addSpacing(20)
        lay.addWidget(_divider_h())
        lay.addSpacing(20)

        # ── Camera section ────────────────────────────────────────────────────
        cam_hdr = QLabel("Camera")
        cam_hdr.setFont(gsans(11, bold=True))
        cam_hdr.setStyleSheet(f"color: {TEXT};")
        lay.addWidget(cam_hdr)
        lay.addSpacing(4)

        cam_desc = QLabel("Select the input camera device for live recognition.")
        cam_desc.setFont(gsans(9))
        cam_desc.setStyleSheet(f"color: {TEXT_DIM};")
        lay.addWidget(cam_desc)
        lay.addSpacing(12)

        cam_row = QHBoxLayout()
        cam_row.setSpacing(12)
        cam_lbl = QLabel("Input device")
        cam_lbl.setFont(gsans(9))
        cam_lbl.setStyleSheet(f"color: {TEXT_DIM};")
        cam_lbl.setFixedWidth(90)

        self._cam_combo = QComboBox()
        self._cam_combo.setFixedHeight(36)
        self._cam_combo.setFont(gsans(9))
        for i in range(4):
            self._cam_combo.addItem(f"Camera {i}", i)
        self._cam_combo.setCurrentIndex(current_cam)
        self._cam_combo.setStyleSheet(f"""
            QComboBox {{
                background: {SURFACE3}; color: {TEXT};
                border: 1px solid {BORDER}; border-radius: 8px;
                padding-left: 12px;
            }}
            QComboBox::drop-down {{ border: none; width: 24px; }}
            QComboBox::down-arrow {{
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid {TEXT_DIM};
                width: 0; height: 0; margin-right: 6px;
            }}
            QComboBox QAbstractItemView {{
                background: {SURFACE3}; color: {TEXT};
                border: 1px solid {BORDER};
                selection-background-color: {ACCENT_DIM};
                selection-color: {BG};
                outline: none;
            }}
        """)

        cam_row.addWidget(cam_lbl)
        cam_row.addWidget(self._cam_combo, stretch=1)
        lay.addLayout(cam_row)
        lay.addSpacing(24)
        lay.addWidget(_divider_h())
        lay.addSpacing(20)

        # ── About section ─────────────────────────────────────────────────────
        about_hdr = QLabel("About")
        about_hdr.setFont(gsans(11, bold=True))
        about_hdr.setStyleSheet(f"color: {TEXT};")
        lay.addWidget(about_hdr)
        lay.addSpacing(4)

        t2 = QLabel("ArSL Recognition  ·  YOLO26  ·  ArSL21L Dataset  ·  32 classes")
        t2.setFont(gsans(9))
        t2.setStyleSheet(f"color: {TEXT_DIM};")
        lay.addWidget(t2)
        lay.addSpacing(14)

        for k, v in [
            ("Model",    "YOLO26 via ultralytics"),
            ("Dataset",  "ArSL21L — 32 classes"),
            ("Input",    "Webcam  ·  Image"),
            ("Device",   "CPU inference"),
            ("Img size", "640 × 640 px"),
            ("GUI",      "PySide6"),
            ("Python",   "3.12+"),
        ]:
            row = QHBoxLayout()
            row.setSpacing(0)
            lk = QLabel(k)
            lk.setFont(gsans(9, bold=True))
            lk.setStyleSheet(f"color: {ACCENT_DIM};")
            lk.setFixedWidth(80)
            lv = QLabel(v)
            lv.setFont(gsans(9))
            lv.setStyleSheet(f"color: {TEXT_DIM};")
            row.addWidget(lk)
            row.addWidget(lv)
            row.addStretch()
            lay.addLayout(row)
            lay.addSpacing(4)

        lay.addStretch()

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        btn_row.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedSize(100, 38)
        cancel_btn.setCursor(Qt.PointingHandCursor)
        cancel_btn.setFont(gsans(9))
        cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background: {SURFACE3}; color: {TEXT_DIM};
                border: 1px solid {BORDER}; border-radius: 8px;
            }}
            QPushButton:hover {{
                background: {SURFACE2}; color: {TEXT}; border-color: {ACCENT_DIM};
            }}
        """)
        cancel_btn.clicked.connect(self.reject)

        apply_btn = QPushButton("Apply")
        apply_btn.setFixedSize(100, 38)
        apply_btn.setCursor(Qt.PointingHandCursor)
        apply_btn.setFont(gsans(9, bold=True))
        apply_btn.setStyleSheet(f"""
            QPushButton {{
                background: {ACCENT_DIM}; color: {BG};
                border: none; border-radius: 8px;
            }}
            QPushButton:hover {{
                background: {ACCENT}; color: {BG};
            }}
        """)
        apply_btn.clicked.connect(self.accept)

        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(apply_btn)
        lay.addLayout(btn_row)

    def selected_camera_index(self) -> int:
        return self._cam_combo.currentData() or 0


# ─── Letter panel ─────────────────────────────────────────────────────────────
class LetterPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(320)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        self._has_detection   = False
        self._current_key     = ""
        self._pending_key     = ""
        self._animating       = False
        self._auto_played_key = ""

        self._hold_timer = QTimer(self)
        self._hold_timer.setSingleShot(True)
        self._hold_timer.setInterval(500)
        self._hold_timer.timeout.connect(self._auto_play_current)

        self._player    = QMediaPlayer()
        self._audio_out = QAudioOutput()
        self._player.setAudioOutput(self._audio_out)
        self._audio_out.setVolume(1.0)
        self._player.positionChanged.connect(self._on_audio_pos)
        self._player.durationChanged.connect(self._on_audio_dur)

        self._build()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(5, 5, self.width() - 10, self.height() - 10)
        path = QPainterPath()
        path.addRoundedRect(r, 14, 14)
        p.fillPath(path, QColor(SURFACE))
        p.setPen(QPen(QColor(BORDER), 1))
        p.drawPath(path)
        p.end()

    # ── Layout build ──────────────────────────────────────────────────────────
    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._stack = QStackedWidget()
        self._stack.setStyleSheet("background: transparent;")

        self._stack.addWidget(self._page_placeholder())   # 0
        content = self._page_content()
        self._stack.addWidget(content)                    # 1
        self._stack.setCurrentIndex(0)
        root.addWidget(self._stack)

        self._fx = QGraphicsOpacityEffect(content)
        self._fx.setOpacity(1.0)
        content.setGraphicsEffect(self._fx)

        self._anim = QPropertyAnimation(self._fx, b"opacity")
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim_finished_connected = False

    def _page_placeholder(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet("background: transparent;")
        w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay = QVBoxLayout(w)
        lay.setAlignment(Qt.AlignCenter)
        lay.setSpacing(12)

        symbol = QLabel("◆")
        symbol.setFont(gsans(26))
        symbol.setAlignment(Qt.AlignCenter)
        symbol.setStyleSheet(f"color: {TEXT_MUTED}; background: transparent;")

        msg = QLabel("make a sign\nto begin")
        msg.setFont(gsans(11))
        msg.setAlignment(Qt.AlignCenter)
        msg.setStyleSheet(f"color: {TEXT_MUTED}; background: transparent;")

        lay.addWidget(symbol)
        lay.addWidget(msg)
        return w

    def _page_content(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet("background: transparent;")
        w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        lay = QVBoxLayout(w)
        lay.setContentsMargins(22, 22, 22, 22)
        lay.setSpacing(0)

        # ── Images ────────────────────────────────────────────────────────────
        img_row = QHBoxLayout()
        img_row.setSpacing(14)
        img_row.addStretch()

        self.letter_img = QLabel()
        self.letter_img.setFixedSize(116, 116)
        self.letter_img.setAlignment(Qt.AlignCenter)
        self.letter_img.setStyleSheet("background: transparent;")
        img_row.addWidget(self.letter_img)

        right_col = QVBoxLayout()
        right_col.setSpacing(6)
        right_col.addStretch()

        self.hand_img = QLabel()
        self.hand_img.setFixedSize(110, 110)
        self.hand_img.setAlignment(Qt.AlignCenter)
        self.hand_img.setStyleSheet(f"""
            background: {SURFACE3};
            border-radius: 10px;
        """)

        self.hand_label = QLabel("")
        self.hand_label.setFont(gsans(8))
        self.hand_label.setAlignment(Qt.AlignCenter)
        self.hand_label.setStyleSheet(f"color: {TEXT_DIM}; background: transparent;")

        right_col.addWidget(self.hand_img)
        right_col.addWidget(self.hand_label)
        right_col.addStretch()
        img_row.addLayout(right_col)
        img_row.addStretch()
        lay.addLayout(img_row)
        lay.addSpacing(20)

        lay.addWidget(_divider_h())
        lay.addSpacing(18)

        # ── Arabic ────────────────────────────────────────────────────────────
        self.name_ar = QLabel()
        self.name_ar.setFont(QFont("Arial", 15, QFont.Bold))
        self.name_ar.setAlignment(Qt.AlignCenter)
        self.name_ar.setStyleSheet(f"color: {TEXT}; background: transparent;")
        lay.addWidget(self.name_ar)
        lay.addSpacing(6)

        self.desc_ar = QLabel()
        self.desc_ar.setFont(QFont("Arial", 9))
        self.desc_ar.setAlignment(Qt.AlignCenter)
        self.desc_ar.setWordWrap(True)
        self.desc_ar.setStyleSheet(f"color: {TEXT_DIM}; background: transparent;")
        self.desc_ar.setLayoutDirection(Qt.RightToLeft)
        lay.addWidget(self.desc_ar)
        lay.addSpacing(18)

        lay.addWidget(_divider_h())
        lay.addSpacing(18)

        # ── English ───────────────────────────────────────────────────────────
        self.name_en = QLabel()
        self.name_en.setFont(gsans(14, bold=True))
        self.name_en.setAlignment(Qt.AlignCenter)
        self.name_en.setStyleSheet(f"color: {ACCENT}; background: transparent;")
        lay.addWidget(self.name_en)
        lay.addSpacing(6)

        self.phonetic = QLabel()
        self.phonetic.setFont(gsans(9))
        self.phonetic.setAlignment(Qt.AlignCenter)
        self.phonetic.setStyleSheet(f"color: {ACCENT_DIM}; background: transparent;")
        lay.addWidget(self.phonetic)
        lay.addSpacing(6)

        self.desc_en = QLabel()
        self.desc_en.setFont(gsans(9))
        self.desc_en.setAlignment(Qt.AlignCenter)
        self.desc_en.setWordWrap(True)
        self.desc_en.setStyleSheet(f"color: {TEXT_DIM}; background: transparent;")
        lay.addWidget(self.desc_en)
        lay.addSpacing(18)

        lay.addStretch()

        # ── Pronunciation ─────────────────────────────────────────────────────
        lay.addWidget(_divider_h())
        lay.addSpacing(16)

        audio_hdr = QLabel("Pronunciation  ·  النطق")
        audio_hdr.setFont(gsans(8, bold=True))
        audio_hdr.setAlignment(Qt.AlignCenter)
        audio_hdr.setStyleSheet(f"color: {ACCENT_DIM}; background: transparent;")
        lay.addWidget(audio_hdr)
        lay.addSpacing(12)

        self.audio_slider = QSlider(Qt.Horizontal)
        self.audio_slider.setRange(0, 1000)
        self.audio_slider.setValue(0)
        self.audio_slider.setFixedHeight(6)
        self.audio_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                background: {SURFACE3}; height: 4px; border-radius: 2px;
            }}
            QSlider::sub-page:horizontal {{
                background: {ACCENT}; height: 4px; border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {ACCENT}; width: 10px; height: 10px;
                margin: -3px 0; border-radius: 5px;
            }}
        """)
        self.audio_slider.sliderMoved.connect(self._on_slider_moved)
        lay.addWidget(self.audio_slider)
        lay.addSpacing(12)

        audio_ctrl = QHBoxLayout()
        audio_ctrl.setSpacing(10)
        audio_ctrl.addStretch()

        self.play_audio_btn = self._audio_btn(MI_PLAY, 44)

        self.play_audio_btn.clicked.connect(self._play_pronunciation)

        audio_ctrl.addWidget(self.play_audio_btn)
        audio_ctrl.addStretch()
        lay.addLayout(audio_ctrl)
        lay.addSpacing(18)

        lay.addWidget(_divider_h())

        return w

    # ── Widget factories ──────────────────────────────────────────────────────
    def _audio_btn(self, label: str, w: int) -> QPushButton:
        btn = QPushButton(label)
        btn.setFixedSize(w, 34)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFont(micon(18))
        btn.setStyleSheet(f"""
            QPushButton {{
                background: {SURFACE3}; color: {TEXT_DIM};
                border: 1px solid {BORDER}; border-radius: 8px;
            }}
            QPushButton:hover {{
                background: {SURFACE2}; color: {TEXT}; border-color: {ACCENT_DIM};
            }}
        """)
        return btn

    # ── Audio slots ───────────────────────────────────────────────────────────
    def _on_audio_pos(self, pos: int):
        dur = self._player.duration()
        if dur > 0:
            self.audio_slider.setValue(int(pos * 1000 / dur))

    def _on_audio_dur(self, dur: int):
        pass

    def _on_slider_moved(self, val: int):
        dur = self._player.duration()
        if dur > 0:
            self._player.setPosition(int(val * dur / 1000))

    def _play_pronunciation(self):
        self._player.stop()
        self._player.play()

    def _auto_play_current(self):
        self._auto_played_key = self._current_key
        self._play_pronunciation()

    def _load_audio(self, key: str):
        path = find_sound_path(key)
        if path:
            self._player.setSource(QUrl.fromLocalFile(path))
        else:
            self._player.setSource(QUrl())

    # ── Populate ──────────────────────────────────────────────────────────────
    def _populate(self, key: str):
        data = LETTER_DATA[key]
        self.letter_img.setPixmap(_make_letter_pixmap(data["arabic"], 116))
        self.hand_img.setPixmap(load_hand_photo(key, 110))
        self.hand_label.setText(data["name_en"])
        self.name_ar.setText(data["name_ar"])
        self.desc_ar.setText(data["desc_ar"])
        self.name_en.setText("Letter  " + data["name_en"])
        self.phonetic.setText(data["phonetic"])
        self.desc_en.setText(data["desc_en"])
        self._load_audio(key)

    # ── Fade helpers ──────────────────────────────────────────────────────────
    def _fade(self, start: float, end: float, ms: int, on_done=None):
        self._anim.stop()
        if self._anim_finished_connected:
            self._anim.finished.disconnect()
            self._anim_finished_connected = False
        if on_done:
            self._anim.finished.connect(on_done)
            self._anim_finished_connected = True
        self._anim.setDuration(ms)
        self._anim.setStartValue(start)
        self._anim.setEndValue(end)
        self._fx.setOpacity(start)
        self._anim.start()

    # ── Public API ────────────────────────────────────────────────────────────
    def show_letter(self, key: str):
        if key not in LETTER_DATA:
            return

        if not self._has_detection:
            self._has_detection   = True
            self._current_key     = key
            self._auto_played_key = ""
            self._populate(key)
            self._stack.setCurrentIndex(1)
            self._fade(0.0, 1.0, 320)
            self._hold_timer.start()
            return

        if key == self._current_key:
            return

        self._hold_timer.stop()
        self._auto_played_key = ""

        if self._animating:
            self._pending_key = key
            return

        self._animating   = True
        self._current_key = key

        def _mid():
            actual = self._pending_key if self._pending_key else key
            self._pending_key = ""
            self._current_key = actual
            self._auto_played_key = ""
            self._populate(actual)
            self._fade(0.0, 1.0, 220, _done)

        def _done():
            self._animating = False
            self._hold_timer.start()
            if self._pending_key:
                next_key = self._pending_key
                self._pending_key = ""
                self.show_letter(next_key)

        self._fade(1.0, 0.0, 140, _mid)

    def show_placeholder(self):
        self._hold_timer.stop()
        self._player.stop()
        self._has_detection = False
        self._current_key   = ""
        self._stack.setCurrentIndex(0)


# ─── Webcam canvas ────────────────────────────────────────────────────────────
class WebcamCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(700, 480)
        self._pixmap   = None
        self._fps_text = ""
        self._show_placeholder()

    def _show_placeholder(self):
        self._pixmap = make_webcam_placeholder(700, 480)
        self.update()

    def set_frame(self, px: QPixmap):
        self._pixmap = px.scaled(700, 480, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.update()

    def set_fps(self, fps: float):
        self._fps_text = f"FPS: {fps:.1f}"
        self.update()

    def clear_fps(self):
        self._fps_text = ""
        self.update()

    def reset(self):
        self._show_placeholder()
        self.clear_fps()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, self.width(), self.height()), 12, 12)
        p.setClipPath(path)

        p.fillRect(self.rect(), QColor(BG))

        if self._pixmap and not self._pixmap.isNull():
            x = (self.width()  - self._pixmap.width())  // 2
            y = (self.height() - self._pixmap.height()) // 2
            p.drawPixmap(x, y, self._pixmap)

        p.setClipping(False)

        if self._fps_text:
            f = mono(8)
            p.setFont(f)
            p.setPen(QColor("#2e2e2e"))
            tw = QFontMetrics(f).horizontalAdvance(self._fps_text)
            p.drawText(self.width() - tw - 8, self.height() - 8, self._fps_text)

        p.end()


# ─── Status bar ───────────────────────────────────────────────────────────────
class StatusBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(38)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setStyleSheet(f"""
            QWidget {{
                background: #0d0d0d;
                border-top: 2px solid {BORDER};
            }}
        """)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(20, 0, 20, 0)
        lay.setSpacing(8)

        self._dot = QLabel("◆")
        self._dot.setFont(mono(7))
        self._dot.setFixedWidth(12)
        self._dot.setStyleSheet(f"color: {TEXT_MUTED}; background: transparent; border: none;")
        lay.addWidget(self._dot)

        self._label = QLabel("Status: Ready")
        self._label.setFont(gsans(9))
        self._label.setStyleSheet(f"color: {TEXT_DIM}; background: transparent; border: none;")
        lay.addWidget(self._label)
        lay.addStretch()

        self.set_idle()

    def _apply(self, text: str, dot_color: str):
        self._label.setText(text)
        self._dot.setStyleSheet(
            f"color: {dot_color}; background: transparent; border: none;"
        )

    def set_idle(self):          self._apply("Status:  Ready",                    TEXT_MUTED)
    def set_webcam_on(self):     self._apply("Status:  Webcam  ·  Recognizing",  ACCENT)
    def set_webcam_paused(self): self._apply("Status:  Webcam paused",            TEXT_DIM)
    def set_stopped(self):       self._apply("Status:  Stopped",                 RED)

    def set_recognizing_image(self, filename: str = ""):
        name = filename.replace("\\", "/").split("/")[-1] or "image"
        self._apply(f"Status:  Image  ·  {name}", "#e5aa00")

    def set_error(self, msg: str):
        self._apply(f"Error:  {msg}", RED)


# ─── Top bar ──────────────────────────────────────────────────────────────────
class TopBar(QWidget):
    settings_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(56)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setStyleSheet(f"background: {SURFACE}; border-bottom: 1px solid {BORDER};")

        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 0, 16, 0)
        lay.setSpacing(0)

        self._logo_lbl = QLabel()
        self._logo_lbl.setFixedSize(34, 34)
        self._logo_lbl.setAlignment(Qt.AlignCenter)
        self._load_logo()
        lay.addWidget(self._logo_lbl)
        lay.addSpacing(10)

        lay.addStretch()

        title_row = QHBoxLayout()
        title_row.setSpacing(10)
        title_row.setAlignment(Qt.AlignCenter)

        main_title = QLabel("Arabic Sign Language Recognition")
        main_title.setFont(gsans(13, bold=True))
        main_title.setAlignment(Qt.AlignCenter)
        main_title.setStyleSheet(f"color: {TEXT}; background: transparent;")
        title_row.addWidget(main_title)

        sep = QLabel("·")
        sep.setFont(gsans(13, bold=True))
        sep.setAlignment(Qt.AlignCenter)
        sep.setStyleSheet(f"color: {ACCENT_DIM}; background: transparent;")
        title_row.addWidget(sep)

        ar_title = QLabel("لغة الإشارة العربية")
        ar_title.setFont(QFont(ARABIC_FONT, 12, QFont.Bold))
        ar_title.setAlignment(Qt.AlignCenter)
        ar_title.setStyleSheet(f"color: {ACCENT}; background: transparent;")
        title_row.addWidget(ar_title)

        lay.addLayout(title_row)

        lay.addStretch()

        settings_btn = QPushButton(MI_SETTINGS)
        settings_btn.setFixedSize(36, 32)
        settings_btn.setCursor(Qt.PointingHandCursor)
        settings_btn.setToolTip("Settings")
        settings_btn.setFont(micon(18))
        settings_btn.setStyleSheet(f"""
            QPushButton {{
                background: {SURFACE3}; color: {TEXT_DIM};
                border: 1px solid {BORDER}; border-radius: 8px;
            }}
            QPushButton:hover {{
                background: {ACCENT_BG}; color: {ACCENT}; border-color: {ACCENT_DIM};
            }}
        """)
        settings_btn.clicked.connect(self.settings_clicked)
        lay.addWidget(settings_btn)

    def _load_logo(self):
        for fname in ("logo.png", "logo.svg", "logo.ico"):
            px = QPixmap(fname)
            if not px.isNull():
                self._logo_lbl.setPixmap(
                    px.scaled(34, 34, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
                return
        self._logo_lbl.setText("◆")
        self._logo_lbl.setFont(mono(20))
        self._logo_lbl.setStyleSheet(f"color: {ACCENT}; background: transparent;")


# ─── Main window ──────────────────────────────────────────────────────────────
class ArSLMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ArSL Recognition — YOLO26")
        self.setFixedSize(1080, 700)
        self.setStyleSheet(f"background: {BG};")

        self.model = None
        self._cam_thread: WebcamThread | None = None
        self._img_thread: ImageInferenceThread | None = None
        self._mode      = "idle"
        self._cam_index = 0

        self._build_ui()
        self._load_model()

    def _build_ui(self):
        central = QWidget()
        central.setStyleSheet(f"background: {BG};")
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Top bar — full width
        self.topbar = TopBar()
        self.topbar.settings_clicked.connect(self._open_settings)
        root.addWidget(self.topbar)

        # Middle
        mid_widget = QWidget()
        mid_widget.setStyleSheet(f"background: {BG};")
        mid_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        mid_layout = QHBoxLayout(mid_widget)
        mid_layout.setContentsMargins(0, 0, 0, 0)
        mid_layout.setSpacing(0)

        # ── Left: webcam area ─────────────────────────────────────────────────
        left_wrapper = QWidget()
        left_wrapper.setStyleSheet(f"background: {BG};")
        left_col = QVBoxLayout(left_wrapper)
        left_col.setContentsMargins(20, 16, 20, 0)
        left_col.setSpacing(0)

        # Header row: label + folder upload button
        header_row = QHBoxLayout()
        header_row.setSpacing(0)

        section_lbl = QLabel("Webcam Window")
        section_lbl.setFont(gsans(10, bold=True))
        section_lbl.setStyleSheet(f"color: {TEXT_DIM}; background: transparent;")
        header_row.addWidget(section_lbl)
        header_row.addStretch()

        self.upload_btn = QPushButton(MI_FOLDER)
        self.upload_btn.setFixedSize(30, 30)
        self.upload_btn.setCursor(Qt.PointingHandCursor)
        self.upload_btn.setToolTip("Upload an image from computer")
        self.upload_btn.setFont(micon(16))
        self.upload_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {TEXT_DIM};
                border: 1px solid {BORDER}; border-radius: 7px;
            }}
            QPushButton:hover {{
                background: {ACCENT_BG}; color: {ACCENT}; border-color: {ACCENT_DIM};
            }}
        """)
        self.upload_btn.clicked.connect(self._open_image)
        header_row.addWidget(self.upload_btn)

        left_col.addLayout(header_row)
        left_col.addSpacing(8)

        # Webcam canvas
        self.canvas = WebcamCanvas()
        left_col.addWidget(self.canvas)
        left_col.addSpacing(12)

        # Play / Pause buttons under canvas
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(10)
        ctrl_row.addStretch()

        self.play_btn = self._webcam_btn(MI_PLAY)
        self.play_btn.setToolTip("Start webcam")
        self.play_btn.clicked.connect(self._start_webcam)

        self.pause_btn = self._webcam_btn(MI_PAUSE)
        self.pause_btn.setToolTip("Pause webcam")
        self.pause_btn.clicked.connect(self._pause_webcam)

        ctrl_row.addWidget(self.play_btn)
        ctrl_row.addWidget(self.pause_btn)
        ctrl_row.addStretch()
        left_col.addLayout(ctrl_row)
        left_col.addStretch()

        mid_layout.addWidget(left_wrapper, stretch=1)

        vsep = QFrame()
        vsep.setFrameShape(QFrame.VLine)
        vsep.setFixedWidth(1)
        vsep.setStyleSheet(f"background: {BORDER}; border: none;")
        mid_layout.addWidget(vsep)

        # ── Right: letter panel ───────────────────────────────────────────────
        self.letter_panel = LetterPanel()
        mid_layout.addWidget(self.letter_panel)

        root.addWidget(mid_widget, stretch=1)

        # Status bar — full width
        self.status_bar = StatusBar()
        root.addWidget(self.status_bar)

    def _webcam_btn(self, label: str) -> QPushButton:
        btn = QPushButton(label)
        btn.setFixedSize(50, 38)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFont(micon(20))
        btn.setStyleSheet(f"""
            QPushButton {{
                background: {SURFACE3}; color: {TEXT_DIM};
                border: 1px solid {BORDER}; border-radius: 10px;
            }}
            QPushButton:hover {{
                background: {ACCENT_DIM}; color: {BG}; border-color: {ACCENT};
            }}
        """)
        return btn

    # ── Model ─────────────────────────────────────────────────────────────────
    def _load_model(self):
        if not os.path.exists(WEIGHTS):
            QMessageBox.critical(self, "Model not found",
                f"Weights not found:\n{WEIGHTS}\n\nRun train.py first.")
            self._set_inference_enabled(False)
            return
        try:
            from ultralytics import YOLO
            self.model = YOLO(WEIGHTS)
        except Exception as e:
            QMessageBox.critical(self, "Model error", str(e))
            self._set_inference_enabled(False)

    def _set_inference_enabled(self, enabled: bool):
        for w in (self.play_btn, self.pause_btn, self.upload_btn):
            w.setEnabled(enabled)

    # ── Slots ─────────────────────────────────────────────────────────────────
    def _open_settings(self):
        dlg = SettingsDialog(current_cam=self._cam_index, parent=self)
        if dlg.exec() == QDialog.Accepted:
            self._cam_index = dlg.selected_camera_index()

    def _start_webcam(self):
        if self.model is None:
            return
        self._stop_threads()
        self._mode = "webcam"
        self.status_bar.set_webcam_on()

        self._cam_thread = WebcamThread(
            self.model, self._cam_index, DEVICE, CONF, IMG_SIZE
        )
        self._cam_thread.frame_ready.connect(self.canvas.set_frame)
        self._cam_thread.fps_signal.connect(self.canvas.set_fps)
        self._cam_thread.detection.connect(self._on_detection)
        self._cam_thread.error.connect(self.status_bar.set_error)
        self._cam_thread.start()

    def _pause_webcam(self):
        self._stop_threads()
        self.status_bar.set_webcam_paused()
        self.canvas.reset()
        self.letter_panel.show_placeholder()
        self._mode = "idle"

    def _open_image(self):
        if self.model is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp)"
        )
        if not path:
            return
        self._stop_threads()
        self._mode = "image"
        self.canvas.clear_fps()
        self.status_bar.set_recognizing_image(path)

        self._img_thread = ImageInferenceThread(
            self.model, path, DEVICE, CONF, IMG_SIZE
        )
        self._img_thread.done.connect(self._on_image_done)
        self._img_thread.error.connect(self.status_bar.set_error)
        self._img_thread.start()

    def _on_image_done(self, px: QPixmap, key: str):
        self.canvas.set_frame(px)
        if key:
            self._on_detection(key)

    def _on_detection(self, key: str):
        if key:
            if key in LETTER_DATA:
                self.letter_panel.show_letter(key)
        else:
            self.letter_panel._hold_timer.stop()

    def _stop_threads(self):
        if self._cam_thread and self._cam_thread.isRunning():
            self._cam_thread.stop()
            self._cam_thread = None
        if self._img_thread and self._img_thread.isRunning():
            self._img_thread.wait(3000)
            self._img_thread = None

    def closeEvent(self, event):
        self._stop_threads()
        self.letter_panel._player.stop()
        event.accept()


# ─── Entry point ──────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Load fonts before any widgets are created
    global SANS_FONT, ICON_FONT, ARABIC_FONT
    SANS_FONT   = _load_or_download("Inter",          "Segoe UI")
    ICON_FONT   = _load_or_download("Material Icons", "Material Icons")
    ARABIC_FONT = _load_or_download("Noto Sans Arabic", "Segoe UI")

    palette = QPalette()
    palette.setColor(QPalette.Window,          QColor(BG))
    palette.setColor(QPalette.WindowText,      QColor(TEXT))
    palette.setColor(QPalette.Base,            QColor(SURFACE))
    palette.setColor(QPalette.AlternateBase,   QColor(SURFACE2))
    palette.setColor(QPalette.Text,            QColor(TEXT))
    palette.setColor(QPalette.Button,          QColor(SURFACE3))
    palette.setColor(QPalette.ButtonText,      QColor(TEXT))
    palette.setColor(QPalette.Highlight,       QColor(ACCENT_DIM))
    palette.setColor(QPalette.HighlightedText, QColor(BG))
    app.setPalette(palette)

    win = ArSLMainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
