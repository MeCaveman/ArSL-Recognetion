"""
ArSL Recognition — Main GUI Application (Integrated)
Drop hand signs → hand_signs/{key}_arsl.png
Drop sounds    → sounds/{key}_arsl.wav  (or .mp3)
Drop logo      → logo.png / logo.svg
"""

import sys
import os
import time
import cv2

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QDialog,
    QFrame, QSizePolicy, QSlider, QStackedWidget,
    QGraphicsOpacityEffect, QMessageBox, QComboBox,
)
from PySide6.QtCore import (
    Qt, QThread, Signal, QPropertyAnimation, QEasingCurve,
    QRect, QUrl,
)
from PySide6.QtGui import (
    QFont, QPixmap, QColor, QPainter, QPen, QImage,
    QFontMetrics, QRadialGradient, QPainterPath, QPalette,
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

# ─── Config ───────────────────────────────────────────────────────────────────
WEIGHTS        = "runs/train/arsl21l/weights/best.pt"
IMG_SIZE       = 640
CONF           = 0.45
DEVICE         = "cpu"
HAND_SIGNS_DIR = "hand_signs"
SOUNDS_DIR     = "sounds"
MONO_FONT      = "JetBrains Mono"          # falls back to Consolas if not installed
MONO_FALLBACK  = "Consolas"

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
    "ain":   {"arabic":"ع",  "name_en":"Ain",   "name_ar":"العين",         "phonetic":"ʕ – ain",                  "desc_en":"Hand forms a curved shape\nresembling the letter.",              "desc_ar":"اليد تأخذ شكلاً منحنياً يشبه شكل الحرف"},
    "al":    {"arabic":"ال", "name_en":"Al",    "name_ar":"ال التعريف",    "phonetic":"Al  (definite article)",   "desc_en":"Lam-Alef ligature.\nCombined single gesture.",                   "desc_ar":"حركة مدمجة لحرفي اللام والألف"},
    "aleff": {"arabic":"ا",  "name_en":"Alef",  "name_ar":"الألف",         "phonetic":"A – lef",                  "desc_en":"Pointer finger up,\nrest of fingers closed.",                    "desc_ar":"رفع السبابة مع إغلاق بقية الأصابع"},
    "bb":    {"arabic":"ب",  "name_en":"Ba",    "name_ar":"الباء",         "phonetic":"B – aa",                   "desc_en":"Four fingers extended flat,\nthumb tucked underneath.",           "desc_ar":"أربعة أصابع ممدودة مع الإبهام تحتها"},
    "dal":   {"arabic":"د",  "name_en":"Dal",   "name_ar":"الدال",         "phonetic":"D – al",                   "desc_en":"Index finger bent,\nthumb touching its tip.",                    "desc_ar":"السبابة منحنية مع لمس الإبهام لطرفها"},
    "dha":   {"arabic":"ظ",  "name_en":"Dha",   "name_ar":"الظاء",         "phonetic":"Dh – aa  (emphatic)",      "desc_en":"Finger points forward\nwith a downward tilt.",                   "desc_ar":"إشارة أمامية مع ميل للأسفل"},
    "dhad":  {"arabic":"ض",  "name_en":"Dhad",  "name_ar":"الضاد",         "phonetic":"Dh – aad  (emphatic)",     "desc_en":"Fist with index bent,\nthumb over fingers.",                     "desc_ar":"قبضة مع انحناء السبابة والإبهام فوقها"},
    "fa":    {"arabic":"ف",  "name_en":"Fa",    "name_ar":"الفاء",         "phonetic":"F – aa",                   "desc_en":"Index and thumb form a circle,\nother fingers extended up.",      "desc_ar":"السبابة والإبهام يشكلان دائرة والأصابع للأعلى"},
    "gaaf":  {"arabic":"ق",  "name_en":"Qaf",   "name_ar":"القاف",         "phonetic":"Q – aaf",                  "desc_en":"Two fingers bent,\nthumb pointing up.",                          "desc_ar":"إصبعان منحنيتان مع الإبهام للأعلى"},
    "ghain": {"arabic":"غ",  "name_en":"Ghain", "name_ar":"الغين",         "phonetic":"Gh – ain",                 "desc_en":"Similar to Ain with\na slight wrist rotation.",                  "desc_ar":"مشابه للعين مع دوران طفيف للمعصم"},
    "ha":    {"arabic":"ه",  "name_en":"Ha",    "name_ar":"الهاء",         "phonetic":"H – aa",                   "desc_en":"Open hand, all fingers\nspread and slightly curved.",             "desc_ar":"يد مفتوحة مع نشر الأصابع وانحناءها قليلاً"},
    "haa":   {"arabic":"ح",  "name_en":"Haa",   "name_ar":"الحاء",         "phonetic":"H – aa  (pharyngeal)",     "desc_en":"Two fingers in a V-shape,\npointing sideways.",                  "desc_ar":"إصبعان بشكل V مع الإشارة إلى الجانب"},
    "jeem":  {"arabic":"ج",  "name_en":"Jeem",  "name_ar":"الجيم",         "phonetic":"J – eem",                  "desc_en":"Index curved inward,\nother fingers closed.",                    "desc_ar":"السبابة منحنية للداخل وبقية الأصابع مغلقة"},
    "kaaf":  {"arabic":"ك",  "name_en":"Kaaf",  "name_ar":"الكاف",         "phonetic":"K – aaf",                  "desc_en":"Index and middle fingers in V,\npalm facing forward.",            "desc_ar":"السبابة والوسطى بشكل V مع الكف للأمام"},
    "khaa":  {"arabic":"خ",  "name_en":"Khaa",  "name_ar":"الخاء",         "phonetic":"Kh – aa",                  "desc_en":"Fist with index bent,\nknuckle forward.",                        "desc_ar":"قبضة مع انحناء السبابة ومد مفصلها للأمام"},
    "la":    {"arabic":"لا", "name_en":"La",    "name_ar":"لا",            "phonetic":"L – aa",                   "desc_en":"Lam-Alef ligature.\nL-shape or crossed fingers.",                "desc_ar":"شكل L أو تقاطع الإصبعين"},
    "laam":  {"arabic":"ل",  "name_en":"Lam",   "name_ar":"اللام",         "phonetic":"L – aam",                  "desc_en":"Index pointing up,\nthumb extended sideways.",                   "desc_ar":"السبابة للأعلى والإبهام للجانب"},
    "meem":  {"arabic":"م",  "name_en":"Meem",  "name_ar":"الميم",         "phonetic":"M – eem",                  "desc_en":"All four fingers closed\nover the thumb.",                       "desc_ar":"الأصابع الأربعة مغلقة فوق الإبهام"},
    "nun":   {"arabic":"ن",  "name_en":"Nun",   "name_ar":"النون",         "phonetic":"N – oon",                  "desc_en":"Index bent downward,\nthumb tucked under it.",                   "desc_ar":"السبابة منحنية للأسفل والإبهام تحتها"},
    "ra":    {"arabic":"ر",  "name_en":"Ra",    "name_ar":"الراء",         "phonetic":"R – aa",                   "desc_en":"Index finger curved,\npointing forward.",                        "desc_ar":"السبابة منحنية مع الإشارة للأمام"},
    "saad":  {"arabic":"ص",  "name_en":"Saad",  "name_ar":"الصاد",         "phonetic":"S – aad  (emphatic)",      "desc_en":"Thumb and index form a loop,\npalm facing down.",                "desc_ar":"الإبهام والسبابة يشكلان حلقة مع الكف للأسفل"},
    "seen":  {"arabic":"س",  "name_en":"Seen",  "name_ar":"السين",         "phonetic":"S – een",                  "desc_en":"Three fingers extended\nand slightly spread.",                   "desc_ar":"ثلاثة أصابع ممدودة ومنتشرة قليلاً"},
    "sheen": {"arabic":"ش",  "name_en":"Sheen", "name_ar":"الشين",         "phonetic":"Sh – een",                 "desc_en":"Three fingers spread\nwith a slight shake motion.",               "desc_ar":"ثلاثة أصابع منتشرة مع حركة طفيفة"},
    "ta":    {"arabic":"ت",  "name_en":"Ta",    "name_ar":"التاء",         "phonetic":"T – aa",                   "desc_en":"Four fingers up, thumb\nfolded across the palm.",                "desc_ar":"أربعة أصابع للأعلى مع ثني الإبهام على الكف"},
    "taa":   {"arabic":"ط",  "name_en":"Taa",   "name_ar":"الطاء",         "phonetic":"T – aa  (emphatic)",       "desc_en":"Fist with thumb tucked inside,\npressing downward.",             "desc_ar":"قبضة مع الإبهام بالداخل والضغط للأسفل"},
    "thaa":  {"arabic":"ث",  "name_en":"Thaa",  "name_ar":"الثاء",         "phonetic":"Th – aa",                  "desc_en":"Three middle fingers\nextended upward.",                         "desc_ar":"ثلاثة أصابع وسطى ممدودة للأعلى"},
    "thal":  {"arabic":"ذ",  "name_en":"Thal",  "name_ar":"الذال",         "phonetic":"Dh – al",                  "desc_en":"Index pointing up,\ntip near the lower lip.",                   "desc_ar":"السبابة للأعلى مع طرفها قرب الشفة السفلى"},
    "toot":  {"arabic":"ة",  "name_en":"Toot",  "name_ar":"التاء المربوطة","phonetic":"T – oot  (Ta Marbuta)",    "desc_en":"Four fingers closed,\nthen opened slightly.",                    "desc_ar":"أربعة أصابع مغلقة ثم تفتح قليلاً"},
    "waw":   {"arabic":"و",  "name_en":"Waw",   "name_ar":"الواو",         "phonetic":"W – aw",                   "desc_en":"Pinky extended up,\nother fingers in a fist.",                   "desc_ar":"الخنصر ممدود للأعلى وبقية الأصابع في قبضة"},
    "ya":    {"arabic":"ي",  "name_en":"Ya",    "name_ar":"الياء",         "phonetic":"Y – aa",                   "desc_en":"Pinky pointing down,\nwrist slightly rotated.",                  "desc_ar":"الخنصر للأسفل مع دوران طفيف للمعصم"},
    "yaa":   {"arabic":"ى",  "name_en":"Yaa",   "name_ar":"الألف المقصورة","phonetic":"Y – aa  (Alef Maqsura)",   "desc_en":"Pinky down with a\nsmaller wrist motion.",                       "desc_ar":"الخنصر للأسفل مع حركة معصم أصغر"},
    "zay":   {"arabic":"ز",  "name_en":"Zay",   "name_ar":"الزاي",         "phonetic":"Z – ay",                   "desc_en":"Index pointing forward,\nthen drawn downward.",                  "desc_ar":"السبابة تشير للأمام ثم تنسحب للأسفل"},
}


# ─── Helpers ──────────────────────────────────────────────────────────────────
def mono(size: int, bold: bool = False) -> QFont:
    """Return JetBrains Mono font, falling back to Consolas."""
    f = QFont(MONO_FONT, size, QFont.Bold if bold else QFont.Normal)
    f.setStyleHint(QFont.Monospace)
    return f


def load_hand_photo(key: str, w: int = 130, h: int = 95) -> QPixmap:
    path = os.path.join(HAND_SIGNS_DIR, f"{key}_arsl.png")
    if os.path.exists(path):
        px = QPixmap(path)
        if not px.isNull():
            return px.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return _make_hand_placeholder(w, h)


def find_sound_path(key: str) -> str | None:
    for ext in ("wav", "mp3", "ogg"):
        p = os.path.join(SOUNDS_DIR, f"{key}_arsl.{ext}")
        if os.path.exists(p):
            return os.path.abspath(p)
    return None


def _make_letter_pixmap(letter: str, size: int = 120) -> QPixmap:
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


def _make_hand_placeholder(w: int, h: int) -> QPixmap:
    px = QPixmap(w, h)
    px.fill(Qt.transparent)
    p = QPainter(px)
    p.fillRect(0, 0, w, h, QColor(SURFACE3))
    p.setPen(QPen(QColor(TEXT_MUTED), 1))
    p.drawRect(0, 0, w - 1, h - 1)
    p.setFont(mono(7))
    p.setPen(QColor(TEXT_MUTED))
    p.drawText(QRect(0, 0, w, h), Qt.AlignCenter, "no photo")
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
    p.setFont(mono(11))
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


# ─── About dialog ─────────────────────────────────────────────────────────────
class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About ArSL")
        self.setFixedSize(480, 380)
        self.setStyleSheet(f"""
            QDialog {{ background: {SURFACE}; border: 1px solid {BORDER}; }}
            QLabel  {{ color: {TEXT}; background: transparent; }}
        """)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(32, 32, 32, 32)
        lay.setSpacing(12)

        t1 = QLabel("ArSL Recognition")
        t1.setFont(mono(16, bold=True))
        t1.setStyleSheet(f"color: {ACCENT};")
        t2 = QLabel("YOLO26  ·  ArSL21L Dataset  ·  32 classes")
        t2.setFont(mono(9))
        t2.setStyleSheet(f"color: {TEXT_DIM};")
        lay.addWidget(t1)
        lay.addWidget(t2)
        lay.addWidget(_divider_h())
        lay.addSpacing(4)

        for k, v in [
            ("Model",    "YOLO26 via ultralytics"),
            ("Dataset",  "ArSL21L — 32 classes"),
            ("Input",    "Webcam  ·  Image"),
            ("Device",   "CPU inference"),
            ("Conf.",    "45 % threshold"),
            ("Img size", "640 × 640 px"),
            ("GUI",      "PySide6"),
            ("Python",   "3.12+"),
        ]:
            row = QHBoxLayout()
            lk = QLabel(k)
            lk.setFont(mono(9, bold=True))
            lk.setStyleSheet(f"color: {ACCENT_DIM};")
            lk.setFixedWidth(80)
            lv = QLabel(v)
            lv.setFont(mono(9))
            lv.setStyleSheet(f"color: {TEXT_DIM};")
            row.addWidget(lk)
            row.addWidget(lv)
            row.addStretch()
            lay.addLayout(row)

        lay.addStretch()
        btn = QPushButton("Close")
        btn.setFixedHeight(36)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFont(mono(9))
        btn.setStyleSheet(f"""
            QPushButton {{
                background: {SURFACE3}; color: {TEXT_DIM};
                border: 1px solid {BORDER}; border-radius: 6px;
            }}
            QPushButton:hover {{
                background: {SURFACE2}; color: {TEXT}; border-color: {ACCENT_DIM};
            }}
        """)
        btn.clicked.connect(self.accept)
        lay.addWidget(btn)


# ─── Letter panel ─────────────────────────────────────────────────────────────
class LetterPanel(QWidget):
    """
    Solid #111 background via paintEvent.
    Placeholder on launch → fade-in on first detection.
    Cross-fade animation between different letters.
    Audio via QMediaPlayer.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(320)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        self._has_detection = False
        self._current_key   = ""
        self._pending_key   = ""
        self._animating     = False

        # Audio
        self._player    = QMediaPlayer()
        self._audio_out = QAudioOutput()
        self._player.setAudioOutput(self._audio_out)
        self._audio_out.setVolume(1.0)
        self._player.positionChanged.connect(self._on_audio_pos)
        self._player.durationChanged.connect(self._on_audio_dur)

        self._build()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(SURFACE))
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

        # Opacity effect on content page
        self._fx = QGraphicsOpacityEffect(content)
        self._fx.setOpacity(1.0)
        content.setGraphicsEffect(self._fx)

        self._anim = QPropertyAnimation(self._fx, b"opacity")
        self._anim.setEasingCurve(QEasingCurve.OutCubic)

    def _page_placeholder(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet("background: transparent;")
        w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay = QVBoxLayout(w)
        lay.setAlignment(Qt.AlignCenter)
        lay.setSpacing(12)

        symbol = QLabel("◆")
        symbol.setFont(mono(26))
        symbol.setAlignment(Qt.AlignCenter)
        symbol.setStyleSheet(f"color: {TEXT_MUTED}; background: transparent;")

        msg = QLabel("make a sign\nto begin")
        msg.setFont(mono(10))
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
        lay.setContentsMargins(26, 22, 26, 18)
        lay.setSpacing(0)

        # ── Images ────────────────────────────────────────────────────────────
        img_row = QHBoxLayout()
        img_row.setSpacing(12)

        self.letter_img = QLabel()
        self.letter_img.setFixedSize(116, 116)
        self.letter_img.setAlignment(Qt.AlignCenter)
        self.letter_img.setStyleSheet("background: transparent;")
        img_row.addWidget(self.letter_img)

        right_col = QVBoxLayout()
        right_col.setSpacing(6)

        self.hand_img = QLabel()
        self.hand_img.setFixedSize(126, 90)
        self.hand_img.setAlignment(Qt.AlignCenter)
        self.hand_img.setStyleSheet("background: transparent;")

        self.hand_label = QLabel("")
        self.hand_label.setFont(mono(8))
        self.hand_label.setAlignment(Qt.AlignCenter)
        self.hand_label.setStyleSheet(f"color: {TEXT_DIM}; background: transparent;")

        right_col.addWidget(self.hand_img)
        right_col.addWidget(self.hand_label)
        img_row.addLayout(right_col)
        lay.addLayout(img_row)
        lay.addSpacing(18)

        lay.addWidget(_divider_h())
        lay.addSpacing(16)

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
        lay.addSpacing(16)

        lay.addWidget(_divider_h())
        lay.addSpacing(16)

        # ── English ───────────────────────────────────────────────────────────
        self.name_en = QLabel()
        self.name_en.setFont(mono(14, bold=True))
        self.name_en.setAlignment(Qt.AlignCenter)
        self.name_en.setStyleSheet(f"color: {ACCENT}; background: transparent;")
        lay.addWidget(self.name_en)
        lay.addSpacing(4)

        self.phonetic = QLabel()
        self.phonetic.setFont(mono(9))
        self.phonetic.setAlignment(Qt.AlignCenter)
        self.phonetic.setStyleSheet(f"color: {ACCENT_DIM}; background: transparent;")
        lay.addWidget(self.phonetic)
        lay.addSpacing(8)

        self.desc_en = QLabel()
        self.desc_en.setFont(mono(9))
        self.desc_en.setAlignment(Qt.AlignCenter)
        self.desc_en.setWordWrap(True)
        self.desc_en.setStyleSheet(f"color: {TEXT_DIM}; background: transparent;")
        lay.addWidget(self.desc_en)

        lay.addStretch()

        # ── Pronunciation / Audio ─────────────────────────────────────────────
        lay.addWidget(_divider_h())
        lay.addSpacing(14)

        audio_hdr = QLabel("Pronunciation")
        audio_hdr.setFont(mono(8))
        audio_hdr.setAlignment(Qt.AlignCenter)
        audio_hdr.setStyleSheet(f"color: {ACCENT_DIM}; background: transparent;")
        lay.addWidget(audio_hdr)
        lay.addSpacing(10)

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
        lay.addSpacing(10)

        audio_ctrl = QHBoxLayout()
        audio_ctrl.setSpacing(8)
        audio_ctrl.addStretch()

        self.play_audio_btn  = self._audio_btn("▶", 40)
        self.pause_audio_btn = self._audio_btn("‖", 40)
        self.stop_audio_btn  = self._audio_btn("■", 40)

        self.play_audio_btn.clicked.connect(self._player.play)
        self.pause_audio_btn.clicked.connect(self._player.pause)
        self.stop_audio_btn.clicked.connect(self._player.stop)

        for b in (self.play_audio_btn, self.pause_audio_btn, self.stop_audio_btn):
            audio_ctrl.addWidget(b)
        audio_ctrl.addStretch()
        lay.addLayout(audio_ctrl)
        lay.addSpacing(14)

        lay.addWidget(_divider_h())
        lay.addSpacing(14)

        # ── Actions ───────────────────────────────────────────────────────────
        action_hdr = QLabel("Recognize from image")
        action_hdr.setFont(mono(8))
        action_hdr.setAlignment(Qt.AlignCenter)
        action_hdr.setStyleSheet(f"color: {ACCENT_DIM}; background: transparent;")
        lay.addWidget(action_hdr)
        lay.addSpacing(10)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.cam_btn    = self._text_btn("Image")
        self.folder_btn = self._text_btn("Browse")
        btn_row.addWidget(self.cam_btn)
        btn_row.addWidget(self.folder_btn)
        lay.addLayout(btn_row)
        lay.addSpacing(8)

        self.back_webcam_btn = QPushButton("←  Webcam")
        self.back_webcam_btn.setFixedHeight(32)
        self.back_webcam_btn.setCursor(Qt.PointingHandCursor)
        self.back_webcam_btn.setFont(mono(9))
        self.back_webcam_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {ACCENT_DIM};
                border: 1px solid {BORDER}; border-radius: 6px;
            }}
            QPushButton:hover {{
                background: {ACCENT_BG}; color: {ACCENT}; border-color: {ACCENT_DIM};
            }}
        """)
        lay.addWidget(self.back_webcam_btn)

        return w

    # ── Widget factories ──────────────────────────────────────────────────────
    def _audio_btn(self, label: str, w: int) -> QPushButton:
        btn = QPushButton(label)
        btn.setFixedSize(w, 32)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFont(mono(10))
        btn.setStyleSheet(f"""
            QPushButton {{
                background: {SURFACE3}; color: {TEXT_DIM};
                border: 1px solid {BORDER}; border-radius: 6px;
            }}
            QPushButton:hover {{
                background: {SURFACE2}; color: {TEXT}; border-color: {ACCENT_DIM};
            }}
        """)
        return btn

    def _text_btn(self, label: str) -> QPushButton:
        btn = QPushButton(label)
        btn.setFixedSize(100, 32)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFont(mono(9))
        btn.setStyleSheet(f"""
            QPushButton {{
                background: {SURFACE3}; color: {TEXT};
                border: 1px solid {BORDER}; border-radius: 6px;
            }}
            QPushButton:hover {{
                background: {SURFACE2}; border-color: {ACCENT_DIM};
            }}
        """)
        return btn

    # ── Audio slots ───────────────────────────────────────────────────────────
    def _on_audio_pos(self, pos: int):
        dur = self._player.duration()
        if dur > 0:
            self.audio_slider.setValue(int(pos * 1000 / dur))

    def _on_audio_dur(self, dur: int):
        pass  # slider range already 0-1000

    def _on_slider_moved(self, val: int):
        dur = self._player.duration()
        if dur > 0:
            self._player.setPosition(int(val * dur / 1000))

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
        self.hand_img.setPixmap(load_hand_photo(key, 126, 90))
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
        try:
            self._anim.finished.disconnect()
        except RuntimeError:
            pass
        if on_done:
            self._anim.finished.connect(on_done)
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
            # First detection: populate and fade in from placeholder
            self._has_detection = True
            self._current_key   = key
            self._populate(key)
            self._stack.setCurrentIndex(1)
            self._fade(0.0, 1.0, 320)
            return

        if key == self._current_key:
            return  # same letter — skip

        if self._animating:
            self._pending_key = key   # only keep latest
            return

        self._animating   = True
        self._current_key = key

        def _mid():
            # Apply pending key if another came in during fade-out
            actual = self._pending_key if self._pending_key else key
            self._pending_key = ""
            self._current_key = actual
            self._populate(actual)
            self._fade(0.0, 1.0, 220, _done)

        def _done():
            self._animating = False
            if self._pending_key:
                next_key = self._pending_key
                self._pending_key = ""
                self.show_letter(next_key)

        self._fade(1.0, 0.0, 140, _mid)

    def show_placeholder(self):
        self._player.stop()
        self._has_detection = False
        self._current_key   = ""
        self._stack.setCurrentIndex(0)


# ─── Webcam canvas ────────────────────────────────────────────────────────────
class WebcamCanvas(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(700, 480)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(f"background: {BG}; border: 1px solid {BORDER};")
        self._fps_text = ""
        self._show_placeholder()

    def _show_placeholder(self):
        self.setPixmap(make_webcam_placeholder(700, 480))

    def set_frame(self, px: QPixmap):
        self.setPixmap(px.scaled(700, 480, Qt.KeepAspectRatio, Qt.SmoothTransformation))

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
        super().paintEvent(event)
        if not self._fps_text:
            return
        p = QPainter(self)
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
        # Darker bg + stronger top border distinguishes it from controls bar above
        self.setStyleSheet(f"""
            QWidget {{
                background: #0d0d0d;
                border-top: 2px solid {BORDER};
            }}
        """)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(20, 0, 20, 0)
        lay.setSpacing(8)

        # State dot
        self._dot = QLabel("◆")
        self._dot.setFont(mono(7))
        self._dot.setFixedWidth(12)
        self._dot.setStyleSheet(f"color: {TEXT_MUTED}; background: transparent; border: none;")
        lay.addWidget(self._dot)

        self._label = QLabel("Status: Ready")
        self._label.setFont(mono(9))
        self._label.setStyleSheet(f"color: {TEXT_DIM}; background: transparent; border: none;")
        lay.addWidget(self._label)
        lay.addStretch()

        # Right: static device info
        info = QLabel(f"CPU  ·  YOLO26  ·  conf {int(CONF*100)}%")
        info.setFont(mono(8))
        info.setStyleSheet(f"color: {TEXT_MUTED}; background: transparent; border: none;")
        lay.addWidget(info)

        self.set_idle()

    def _apply(self, text: str, dot_color: str):
        self._label.setText(text)
        self._dot.setStyleSheet(
            f"color: {dot_color}; background: transparent; border: none;"
        )

    def set_idle(self):          self._apply("Status:  Ready",                      TEXT_MUTED)
    def set_webcam_on(self):     self._apply("Status:  Webcam  ·  Recognizing",    ACCENT)
    def set_webcam_paused(self): self._apply("Status:  Webcam paused",              TEXT_DIM)
    def set_stopped(self):       self._apply("Status:  Stopped",                   RED)

    def set_recognizing_image(self, filename: str = ""):
        name = filename.replace("\\", "/").split("/")[-1] or "image"
        self._apply(f"Status:  Image  ·  {name}", "#e5aa00")

    def set_error(self, msg: str):
        self._apply(f"Error:  {msg}", RED)


# ─── Top bar ──────────────────────────────────────────────────────────────────
class TopBar(QWidget):
    about_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(56)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setStyleSheet(f"background: {SURFACE}; border-bottom: 1px solid {BORDER};")

        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 0, 16, 0)
        lay.setSpacing(0)

        # ── Logo (left) ───────────────────────────────────────────────────────
        self._logo_lbl = QLabel()
        self._logo_lbl.setFixedSize(34, 34)
        self._logo_lbl.setAlignment(Qt.AlignCenter)
        self._load_logo()
        lay.addWidget(self._logo_lbl)
        lay.addSpacing(10)

        # ── Centered title ────────────────────────────────────────────────────
        lay.addStretch()

        title = QLabel("ArSL  —  Arabic Sign Language Recognition")
        title.setFont(mono(13, bold=True))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"color: {TEXT}; background: transparent;")
        lay.addWidget(title)

        lay.addStretch()

        # ── About (right) ─────────────────────────────────────────────────────
        about_btn = QPushButton("About")
        about_btn.setFixedSize(76, 32)
        about_btn.setCursor(Qt.PointingHandCursor)
        about_btn.setFont(mono(9))
        about_btn.setStyleSheet(f"""
            QPushButton {{
                background: {SURFACE3}; color: {TEXT_DIM};
                border: 1px solid {BORDER}; border-radius: 6px;
            }}
            QPushButton:hover {{
                background: {ACCENT_BG}; color: {ACCENT}; border-color: {ACCENT_DIM};
            }}
        """)
        about_btn.clicked.connect(self.about_clicked)
        lay.addWidget(about_btn)

    def _load_logo(self):
        for fname in ("logo.png", "logo.svg", "logo.ico"):
            px = QPixmap(fname)
            if not px.isNull():
                self._logo_lbl.setPixmap(
                    px.scaled(34, 34, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
                return
        # Fallback: symbol
        self._logo_lbl.setText("◆")
        self._logo_lbl.setFont(mono(20))
        self._logo_lbl.setStyleSheet(f"color: {ACCENT}; background: transparent;")


# ─── Webcam controls bar ──────────────────────────────────────────────────────
class WebcamControls(QWidget):
    play_clicked  = Signal()
    pause_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(50)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setStyleSheet(
            f"background: {SURFACE2}; border-top: 1px solid {BORDER};"
        )

        lay = QHBoxLayout(self)
        lay.setContentsMargins(20, 0, 20, 0)
        lay.setSpacing(10)

        cam_lbl = QLabel("Camera")
        cam_lbl.setFont(mono(9))
        cam_lbl.setStyleSheet(f"color: {TEXT_DIM}; background: transparent;")
        lay.addWidget(cam_lbl)

        self.cam_combo = QComboBox()
        self.cam_combo.setFixedSize(180, 32)
        self.cam_combo.setFont(mono(9))
        for i in range(4):
            self.cam_combo.addItem(f"Camera {i}", i)
        self.cam_combo.setStyleSheet(f"""
            QComboBox {{
                background: {SURFACE3}; color: {TEXT};
                border: 1px solid {BORDER}; border-radius: 6px;
                padding-left: 10px;
            }}
            QComboBox::drop-down {{
                border: none; width: 24px;
            }}
            QComboBox::down-arrow {{
                border-left:  5px solid transparent;
                border-right: 5px solid transparent;
                border-top:   5px solid {TEXT_DIM};
                width: 0; height: 0;
                margin-right: 6px;
            }}
            QComboBox QAbstractItemView {{
                background: {SURFACE3}; color: {TEXT};
                border: 1px solid {BORDER};
                selection-background-color: {ACCENT_DIM};
                selection-color: {BG};
                outline: none;
            }}
        """)
        lay.addWidget(self.cam_combo)

        lay.addStretch()

        self.play_btn  = self._ctrl_btn("▶")
        self.pause_btn = self._ctrl_btn("‖")
        self.play_btn.clicked.connect(self.play_clicked)
        self.pause_btn.clicked.connect(self.pause_clicked)
        lay.addWidget(self.play_btn)
        lay.addWidget(self.pause_btn)

    def _ctrl_btn(self, label: str) -> QPushButton:
        btn = QPushButton(label)
        btn.setFixedSize(38, 32)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFont(mono(12))
        btn.setStyleSheet(f"""
            QPushButton {{
                background: {SURFACE3}; color: {TEXT_DIM};
                border: 1px solid {BORDER}; border-radius: 6px;
            }}
            QPushButton:hover {{
                background: {ACCENT_DIM}; color: {BG}; border-color: {ACCENT};
            }}
        """)
        return btn

    def selected_camera_index(self) -> int:
        return self.cam_combo.currentData() or 0


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
        self._mode = "idle"

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
        self.topbar.about_clicked.connect(self._open_about)
        root.addWidget(self.topbar)

        # Middle
        mid_widget = QWidget()
        mid_widget.setStyleSheet(f"background: {BG};")
        mid_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        mid_layout = QHBoxLayout(mid_widget)
        mid_layout.setContentsMargins(0, 0, 0, 0)
        mid_layout.setSpacing(0)

        # Left: canvas
        left_wrapper = QWidget()
        left_wrapper.setStyleSheet(f"background: {BG};")
        left_col = QVBoxLayout(left_wrapper)
        left_col.setContentsMargins(20, 16, 20, 0)
        left_col.setSpacing(0)

        section_lbl = QLabel("Webcam Window")
        section_lbl.setFont(mono(10, bold=True))
        section_lbl.setStyleSheet(f"color: {TEXT_DIM}; background: transparent;")
        left_col.addWidget(section_lbl)
        left_col.addSpacing(8)

        self.canvas = WebcamCanvas()
        left_col.addWidget(self.canvas)
        left_col.addStretch()

        mid_layout.addWidget(left_wrapper, stretch=1)

        vsep = QFrame()
        vsep.setFrameShape(QFrame.VLine)
        vsep.setFixedWidth(1)
        vsep.setStyleSheet(f"background: {BORDER}; border: none;")
        mid_layout.addWidget(vsep)

        # Right: letter panel
        self.letter_panel = LetterPanel()
        self.letter_panel.back_webcam_btn.clicked.connect(self._back_to_webcam)
        self.letter_panel.folder_btn.clicked.connect(self._open_image)
        self.letter_panel.cam_btn.clicked.connect(self._open_image)
        mid_layout.addWidget(self.letter_panel)

        root.addWidget(mid_widget, stretch=1)

        # Webcam controls — full width
        self.webcam_ctrl = WebcamControls()
        self.webcam_ctrl.play_clicked.connect(self._start_webcam)
        self.webcam_ctrl.pause_clicked.connect(self._pause_webcam)
        root.addWidget(self.webcam_ctrl)

        # Status bar — full width
        self.status_bar = StatusBar()
        root.addWidget(self.status_bar)

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
        for w in (self.webcam_ctrl.play_btn, self.webcam_ctrl.pause_btn,
                  self.letter_panel.cam_btn, self.letter_panel.folder_btn):
            w.setEnabled(enabled)

    # ── Slots ─────────────────────────────────────────────────────────────────
    def _open_about(self):
        AboutDialog(self).exec()

    def _start_webcam(self):
        if self.model is None:
            return
        self._stop_threads()
        cam_idx = self.webcam_ctrl.selected_camera_index()
        self._mode = "webcam"
        self.status_bar.set_webcam_on()

        self._cam_thread = WebcamThread(self.model, cam_idx, DEVICE, CONF, IMG_SIZE)
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

    def _back_to_webcam(self):
        self._stop_threads()
        self.canvas.reset()
        self.status_bar.set_idle()
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
        if key in LETTER_DATA:
            self.letter_panel.show_letter(key)

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
