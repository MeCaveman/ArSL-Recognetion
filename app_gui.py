"""
ArSL21L Recognition — GUI App (YOLO26)
=========================================
To Run: python app_gui.py
"""

import sys
import os
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk

WEIGHTS  = "runs/train/arsl21l/weights/best.pt"
IMG_SIZE = 640
CONF     = 0.45
DEVICE   = 0

BG      = "#0d0d0d"
SURFACE = "#161616"
ACCENT  = "#00e5aa"
ACCENT2 = "#005f46"
TEXT    = "#f0f0f0"
MUTED   = "#666666"


class ArSLApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ArSL Recognition — YOLO26")
        self.configure(bg=BG)
        self.geometry("1060x680")
        self.resizable(False, False)

        self.model   = None
        self.cap     = None
        self.running = False

        self._load_model()
        self._build_ui()

    def _load_model(self):
        if not os.path.exists(WEIGHTS):
            messagebox.showerror("Model not found",
                f"Weights not found:\n{WEIGHTS}\n\nRun train.py first.")
            self.destroy()
            return
        try:
            from ultralytics import YOLO
            self.model = YOLO(WEIGHTS)
        except ImportError:
            messagebox.showerror("Missing package", "Run: uv add ultralytics")
            self.destroy()

    def _build_ui(self):
        sidebar = tk.Frame(self, bg=SURFACE, width=220)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="🤟", font=("Segoe UI Emoji", 36),
                 bg=SURFACE, fg=ACCENT).pack(pady=(30, 0))
        tk.Label(sidebar, text="ArSL21L", font=("Consolas", 14, "bold"),
                 bg=SURFACE, fg=TEXT).pack()
        tk.Label(sidebar, text="YOLO26 Recognition", font=("Consolas", 9),
                 bg=SURFACE, fg=MUTED).pack(pady=(0, 30))

        self._btn(sidebar, "📷  Webcam", self._start_webcam)
        self._btn(sidebar, "🖼  Image",  self._open_image)
        self._btn(sidebar, "🎬  Video",  self._open_video)
        self._btn(sidebar, "⏹  Stop",   self._stop, color="#cc3333")

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(sidebar, textvariable=self.status_var, font=("Consolas", 8),
                 bg=SURFACE, fg=MUTED, wraplength=180).pack(side=tk.BOTTOM, pady=12)

        main = tk.Frame(self, bg=BG)
        main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(main, bg="#0a0a0a", highlightthickness=0,
                                width=760, height=560)
        self.canvas.pack(pady=(20, 0), padx=20)

        info = tk.Frame(main, bg=BG)
        info.pack(fill=tk.X, padx=20, pady=8)

        self.label_var = tk.StringVar(value="—")
        tk.Label(info, text="Detected:", font=("Consolas", 10),
                 bg=BG, fg=MUTED).pack(side=tk.LEFT)
        tk.Label(info, textvariable=self.label_var,
                 font=("Consolas", 13, "bold"),
                 bg=BG, fg=ACCENT).pack(side=tk.LEFT, padx=8)

        self.fps_var = tk.StringVar(value="")
        tk.Label(info, textvariable=self.fps_var, font=("Consolas", 10),
                 bg=BG, fg=MUTED).pack(side=tk.RIGHT)

        self._placeholder()

    def _btn(self, parent, text, cmd, color=ACCENT):
        tk.Button(parent, text=text, command=cmd,
                  bg=ACCENT2, fg=color, activebackground="#003d2c",
                  activeforeground=ACCENT, font=("Consolas", 10, "bold"),
                  bd=0, padx=14, pady=10, anchor="w", cursor="hand2",
                  relief=tk.FLAT, width=18).pack(pady=3, padx=16, fill=tk.X)

    def _placeholder(self):
        self.canvas.delete("all")
        self.canvas.create_text(380, 280,
            text="Select a mode from the left →",
            font=("Consolas", 14), fill=MUTED)

    def _run_inference(self, frame):
        results   = self.model.predict(frame, imgsz=IMG_SIZE, conf=CONF,
                                       device=DEVICE, verbose=False)
        annotated = results[0].plot()
        labels = [
            f"{self.model.names[int(b.cls)]} ({float(b.conf):.0%})"
            for b in results[0].boxes
        ]
        self.label_var.set("  |  ".join(labels) if labels else "—")
        return annotated

    def _frame_to_tk(self, frame):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img.thumbnail((760, 560))
        return ImageTk.PhotoImage(img)

    def _show_frame(self, tk_img):
        self.canvas.delete("all")
        self.canvas.create_image(380, 280, image=tk_img, anchor=tk.CENTER)
        self.canvas._img = tk_img

    def _start_webcam(self):
        self._stop()
        self.running = True
        self.status_var.set("Mode: Webcam")
        threading.Thread(target=self._webcam_loop, daemon=True).start()

    def _webcam_loop(self):
        self.cap = cv2.VideoCapture(0)
        prev = time.time()
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            annotated = self._run_inference(frame)
            now = time.time()
            self.fps_var.set(f"FPS: {1/(now-prev+1e-6):.1f}")
            prev = now
            self.after(0, self._show_frame, self._frame_to_tk(annotated))
        if self.cap:
            self.cap.release()

    def _open_image(self):
        self._stop()
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if not path:
            return
        self.status_var.set(f"Image: {os.path.basename(path)}")
        annotated = self._run_inference(cv2.imread(path))
        self._show_frame(self._frame_to_tk(annotated))
        self.fps_var.set("")

    def _open_video(self):
        self._stop()
        path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if not path:
            return
        self.running = True
        self.status_var.set(f"Video: {os.path.basename(path)}")
        threading.Thread(target=self._video_loop, args=(path,), daemon=True).start()

    def _video_loop(self, path):
        self.cap = cv2.VideoCapture(path)
        prev = time.time()
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.status_var.set("Video finished.")
                break
            annotated = self._run_inference(frame)
            now = time.time()
            self.fps_var.set(f"FPS: {1/(now-prev+1e-6):.1f}")
            prev = now
            self.after(0, self._show_frame, self._frame_to_tk(annotated))
            time.sleep(0.01)
        if self.cap:
            self.cap.release()

    def _stop(self):
        self.running = False
        time.sleep(0.1)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.status_var.set("Stopped")
        self.fps_var.set("")
        self.label_var.set("—")
        self._placeholder()

    def on_close(self):
        self._stop()
        self.destroy()


if __name__ == "__main__":
    app = ArSLApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
