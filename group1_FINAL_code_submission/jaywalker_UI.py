"""
jaywalker_UI.py
===============
ENGG2112 — Real-time jaywalking prediction UI.

Loads the trained Keras model and runs inference on test-set images,
displaying the prediction and confidence live.  
Usage:
    python jaywalker_UI.py

Dependencies:
    pip install customtkinter tensorflow pillow opencv-python
"""

import os
import random
import threading
from pathlib import Path

import cv2
import customtkinter as ctk
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow import keras

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE           = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(_HERE, "jaywalking_cnn_transfer.keras")
TEST_DIR        = r"C:\Users\vkolm\OneDrive\Documents\ENGG2112\data\cityscapes-aug-dataset\test"
HOG_PADDING     = 30     # px added around each detected box
HOG_SCORE_MIN   = 0.5   # minimum SVM score to keep a detection
HOG_IOU_THRESH  = 0.3   # NMS overlap threshold

# ── Model / image settings (must match training) ───────────────────────────────
IMG_HEIGHT  = 224
IMG_WIDTH   = 224
THRESHOLD   = 0.5

# ── UI settings ────────────────────────────────────────────────────────────────
AUTO_DELAY_MS   = 2500    # ms between auto-advance frames
DISPLAY_SIZE    = 420     # px — square image display area

# ── Colour palette ─────────────────────────────────────────────────────────────
C_HEADER   = "#3B5CC5"
C_RED      = "#C0392B"
C_GREEN    = "#27AE60"
C_ORANGE   = "#E67E22"
C_GRAY     = "#7F8C8D"
C_DARK     = "#2C3E50"
C_WHITE    = "#FFFFFF"
C_BG       = "#EAECEE"
C_CARD     = "#FFFFFF"

def collect_test_images(test_dir: str) -> list[tuple[str, str]]:
    """
    Walk test_dir/jaywalk/ and test_dir/no_jaywalk/ and return a shuffled
    list of (image_path, true_label) tuples.
    """
    items = []
    for label in ("jaywalk", "no_jaywalk"):
        folder = os.path.join(test_dir, label)
        if not os.path.isdir(folder):
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            for p in Path(folder).glob(ext):
                items.append((str(p), label))
    random.shuffle(items)
    return items


def preprocess_for_model(image_path: str) -> np.ndarray:
    """Load and preprocess one image for inference (raw pixels, no normalisation —
    the model's own Rescaling / preprocess_input layer handles that)."""
    img = Image.open(image_path).convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT))
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)   # shape: (1, H, W, 3)


def _hog_nms(boxes: list, scores: list, iou_thresh: float = HOG_IOU_THRESH) -> np.ndarray:
    """Non-maximum suppression — keep the highest-scoring non-overlapping boxes."""
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=int)
    boxes  = np.array(boxes,  dtype=float)
    scores = np.array(scores, dtype=float)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas  = (x2 - x1 + 1) * (y2 - y1 + 1)
    order  = scores.argsort()[::-1]
    keep   = []
    while len(order):
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / areas[order[1:]]
        order   = order[np.where(overlap <= iou_thresh)[0] + 1]
    return boxes[keep].astype(int)


class JaywalkUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.title("ENGG2112 — Jaywalking Detector")
        self.geometry("1000x680")
        self.minsize(900, 600)
        self.configure(fg_color=C_BG)

        self._model        = None
        self._images       = []
        self._idx          = -1
        self._auto_running = False
        self._after_id     = None
        self._correct      = 0
        self._total        = 0
        self._inferencing  = False

        self._build_ui()
        self._load_model_async()


    def _build_ui(self):
        # ── Header ────────────────────────────────────────────────────────────
        header = ctk.CTkFrame(self, fg_color=C_HEADER, corner_radius=0, height=56)
        header.pack(fill="x")
        header.pack_propagate(False)
        ctk.CTkLabel(
            header, text="🚶  ENGG2112 — Jaywalking Detector",
            font=ctk.CTkFont(size=20, weight="bold"), text_color=C_WHITE
        ).pack(side="left", padx=20)
        self._model_status = ctk.CTkLabel(
            header, text="⏳  Loading model…",
            font=ctk.CTkFont(size=13), text_color="#BDC3C7"
        )
        self._model_status.pack(side="right", padx=20)

        # ── Body ──────────────────────────────────────────────────────────────
        body = ctk.CTkFrame(self, fg_color=C_BG)
        body.pack(fill="both", expand=True, padx=16, pady=12)

        # Left: image panel
        left = ctk.CTkFrame(body, fg_color=C_CARD, corner_radius=14)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))

        self._img_label = ctk.CTkLabel(left, text="", image=None)
        self._img_label.pack(expand=True)

        self._filename_label = ctk.CTkLabel(
            left, text="—",
            font=ctk.CTkFont(size=11), text_color=C_GRAY
        )
        self._filename_label.pack(pady=(0, 10))

        # Right: results panel
        right = ctk.CTkFrame(body, fg_color=C_CARD, corner_radius=14, width=300)
        right.pack(side="left", fill="y")
        right.pack_propagate(False)

        inner = ctk.CTkFrame(right, fg_color="transparent")
        inner.place(relx=0.5, rely=0.42, anchor="center")

        ctk.CTkLabel(
            inner, text="Prediction",
            font=ctk.CTkFont(size=14, weight="bold"), text_color=C_GRAY
        ).pack(pady=(0, 6))

        # Big prediction badge
        self._pred_badge = ctk.CTkLabel(
            inner, text="—",
            font=ctk.CTkFont(size=22, weight="bold"),
            fg_color=C_GRAY, text_color=C_WHITE,
            corner_radius=10, width=220, height=60
        )
        self._pred_badge.pack(pady=4)

        # Jaywalker count (shown when HOG detects people)
        self._jaywalker_count_label = ctk.CTkLabel(
            inner, text="",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=C_RED
        )
        self._jaywalker_count_label.pack(pady=(2, 0))

        # Confidence bar
        ctk.CTkLabel(
            inner, text="Confidence",
            font=ctk.CTkFont(size=13), text_color=C_GRAY
        ).pack(pady=(14, 2))
        self._conf_bar = ctk.CTkProgressBar(inner, width=220, height=18, corner_radius=8)
        self._conf_bar.set(0)
        self._conf_bar.pack()
        self._conf_label = ctk.CTkLabel(
            inner, text="—",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=C_DARK
        )
        self._conf_label.pack(pady=(2, 12))

        # True label
        ctk.CTkLabel(
            inner, text="True Label",
            font=ctk.CTkFont(size=13), text_color=C_GRAY
        ).pack(pady=(10, 2))
        self._true_badge = ctk.CTkLabel(
            inner, text="—",
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color="#D5D8DC", text_color=C_DARK,
            corner_radius=8, width=220, height=44
        )
        self._true_badge.pack(pady=2)

        # Result tick/cross
        self._result_label = ctk.CTkLabel(
            inner, text="",
            font=ctk.CTkFont(size=28)
        )
        self._result_label.pack(pady=6)

        # Running accuracy
        ctk.CTkLabel(
            inner, text="Session Accuracy",
            font=ctk.CTkFont(size=13), text_color=C_GRAY
        ).pack(pady=(12, 2))
        self._accuracy_label = ctk.CTkLabel(
            inner, text="— / —",
            font=ctk.CTkFont(size=18, weight="bold"), text_color=C_DARK
        )
        self._accuracy_label.pack()
        self._acc_bar = ctk.CTkProgressBar(inner, width=220, height=12, corner_radius=6)
        self._acc_bar.configure(progress_color=C_GREEN)
        self._acc_bar.set(0)
        self._acc_bar.pack(pady=(4, 0))

        # ── Controls ──────────────────────────────────────────────────────────
        ctrl = ctk.CTkFrame(self, fg_color=C_BG)
        ctrl.pack(fill="x", padx=16, pady=(0, 14))

        self._prev_btn = ctk.CTkButton(
            ctrl, text="◀  Prev", width=110, height=42,
            fg_color=C_GRAY, hover_color="#5D6D7E",
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._prev_image
        )
        self._prev_btn.pack(side="left", padx=(0, 8))

        self._next_btn = ctk.CTkButton(
            ctrl, text="Next  ▶", width=110, height=42,
            fg_color=C_HEADER, hover_color="#2A4ABF",
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._next_image
        )
        self._next_btn.pack(side="left", padx=(0, 8))

        self._auto_btn = ctk.CTkButton(
            ctrl, text="▶▶  Auto", width=120, height=42,
            fg_color=C_ORANGE, hover_color="#CA6F1E",
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._toggle_auto
        )
        self._auto_btn.pack(side="left", padx=(0, 16))

        self._counter_label = ctk.CTkLabel(
            ctrl, text="Image — / —",
            font=ctk.CTkFont(size=13), text_color=C_GRAY
        )
        self._counter_label.pack(side="left")

        self._reset_btn = ctk.CTkButton(
            ctrl, text="↺  Reset Stats", width=130, height=42,
            fg_color="#AAB7B8", hover_color="#808B96",
            font=ctk.CTkFont(size=13),
            command=self._reset_stats
        )
        self._reset_btn.pack(side="right")

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model_async(self):
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        try:
            model  = keras.models.load_model(MODEL_PATH)
            images = collect_test_images(TEST_DIR)
            self.after(0, self._on_model_loaded, model, images)
        except Exception as e:
            self.after(0, self._on_model_error, str(e))

    def _on_model_loaded(self, model, images):
        self._model  = model
        self._images = images
        self._model_status.configure(
            text=f"✓  Model ready  ·  {len(images)} test images",
            text_color="#58D68D"
        )
        self._next_image()

    def _on_model_error(self, msg):
        self._model_status.configure(
            text=f"✗  {msg}", text_color="#E74C3C"
        )

    # ── Navigation ────────────────────────────────────────────────────────────

    def _next_image(self):
        if not self._images or self._inferencing:
            return
        self._idx = (self._idx + 1) % len(self._images)
        self._run_inference()

    def _prev_image(self):
        if not self._images or self._inferencing:
            return
        self._idx = (self._idx - 1) % len(self._images)
        self._run_inference()

    def _toggle_auto(self):
        if self._auto_running:
            self._auto_running = False
            if self._after_id:
                self.after_cancel(self._after_id)
            self._auto_btn.configure(text="▶▶  Auto", fg_color=C_ORANGE)
        else:
            self._auto_running = True
            self._auto_btn.configure(text="⏹  Stop", fg_color=C_RED)
            self._auto_advance()

    def _auto_advance(self):
        if not self._auto_running:
            return
        self._next_image()
        self._after_id = self.after(AUTO_DELAY_MS, self._auto_advance)

    def _reset_stats(self):
        self._correct = 0
        self._total   = 0
        self._accuracy_label.configure(text="— / —")
        self._acc_bar.set(0)

    # ── Inference ─────────────────────────────────────────────────────────────

    def _run_inference(self):
        if self._model is None or self._inferencing:
            return
        self._inferencing = True
        path, true_label = self._images[self._idx]

        # Show image immediately
        self._display_image(path)
        self._filename_label.configure(text=os.path.basename(path))
        self._counter_label.configure(
            text=f"Image {self._idx + 1} / {len(self._images)}"
        )

        # Reset badges while inferencing
        self._pred_badge.configure(text="…", fg_color=C_GRAY)
        self._conf_label.configure(text="—")
        self._conf_bar.set(0)
        self._result_label.configure(text="")

        # Run model in background thread
        threading.Thread(
            target=self._infer_worker,
            args=(path, true_label),
            daemon=True
        ).start()

    def _infer_worker(self, path: str, true_label: str):
        try:
            arr  = preprocess_for_model(path)
            prob = float(self._model.predict(arr, verbose=0)[0][0])
        except Exception as e:
            self.after(0, lambda: self._model_status.configure(
                text=f"Inference error: {e}", text_color="#E74C3C"))
            self._inferencing = False
            return

        # If jaywalking is predicted, run HOG to annotate bounding boxes
        annotated_img   = None
        jaywalker_count = 0
        pred_label = "jaywalk" if prob < THRESHOLD else "no_jaywalk"
        if pred_label == "jaywalk":
            annotated_img, jaywalker_count = self._run_hog(path)

        self.after(0, self._display_result, prob, true_label, annotated_img, jaywalker_count)

    def _display_result(self, prob: float, true_label: str,
                        annotated_img=None, jaywalker_count: int = 0):
        # Predicted class
        pred_label = "jaywalk" if prob < THRESHOLD else "no_jaywalk"
        confidence = prob if pred_label == "no_jaywalk" else (1.0 - prob)
        correct    = pred_label == true_label

        # Update stats
        self._total   += 1
        self._correct += int(correct)
        accuracy       = self._correct / self._total

        # Prediction badge colour
        if pred_label == "jaywalk":
            badge_color = C_RED
            badge_text  = "⚠  JAYWALKING"
        else:
            badge_color = C_GREEN
            badge_text  = "✓  SAFE"

        self._pred_badge.configure(text=badge_text, fg_color=badge_color)

        # Jaywalker count + annotated image (only for jaywalk predictions)
        if pred_label == "jaywalk" and annotated_img is not None:
            noun = "jaywalker" if jaywalker_count == 1 else "jaywalkers"
            self._jaywalker_count_label.configure(
                text=f"👥  {jaywalker_count} {noun} detected"
            )
            # Replace the plain image with the HOG-annotated version
            self._display_image(None, pil_img=annotated_img)
        else:
            self._jaywalker_count_label.configure(text="")

        # Confidence
        self._conf_bar.configure(
            progress_color=badge_color
        )
        self._conf_bar.set(confidence)
        self._conf_label.configure(text=f"{confidence:.1%}")

        # True label
        true_color = C_RED if true_label == "jaywalk" else C_GREEN
        self._true_badge.configure(
            text=true_label.replace("_", " ").upper(),
            fg_color=true_color, text_color=C_WHITE
        )

        # Tick / cross
        self._result_label.configure(
            text="✅" if correct else "❌"
        )

        # Accuracy bar
        self._accuracy_label.configure(
            text=f"{self._correct} / {self._total}  ({accuracy:.1%})"
        )
        self._acc_bar.set(accuracy)
        self._acc_bar.configure(
            progress_color=C_GREEN if accuracy >= 0.80 else C_ORANGE
        )

        self._inferencing = False

    # ── Image display ─────────────────────────────────────────────────────────

    def _display_image(self, path: str, pil_img: Image.Image = None):
        """Letterbox-fit an image into DISPLAY_SIZE×DISPLAY_SIZE.

        If *pil_img* is provided it is used directly (e.g. a HOG-annotated
        frame); otherwise the image is loaded from *path*.
        """
        if pil_img is not None:
            img = pil_img.convert("RGB")
        else:
            img = Image.open(path).convert("RGB")

        # Letterbox into DISPLAY_SIZE × DISPLAY_SIZE
        img.thumbnail((DISPLAY_SIZE, DISPLAY_SIZE), Image.LANCZOS)
        bg = Image.new("RGB", (DISPLAY_SIZE, DISPLAY_SIZE), (240, 240, 240))
        offset = (
            (DISPLAY_SIZE - img.width)  // 2,
            (DISPLAY_SIZE - img.height) // 2,
        )
        bg.paste(img, offset)

        ctk_img = ctk.CTkImage(light_image=bg, size=(DISPLAY_SIZE, DISPLAY_SIZE))
        self._img_label.configure(image=ctk_img, text="")
        self._img_label._image = ctk_img   # keep reference

    # ── HOG helpers ───────────────────────────────────────────────────────────

    # One shared HOG descriptor for the whole session (cheap to create).
    _HOG = cv2.HOGDescriptor()
    _HOG.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def _run_hog(self, image_path: str) -> tuple[Image.Image | None, int]:
        """Run HOG person-detection on *image_path*.

        Replicates the detection pipeline from HOG.py:
          • histogram equalisation for contrast
          • detectMultiScale with fine stride/scale
          • aspect-ratio, area, and position filters
          • NMS to remove duplicate boxes
          • green bounding-box overlay with person labels

        Returns (annotated_pil_image, person_count).
        """
        img = cv2.imread(image_path)
        if img is None:
            return None, 0

        H, W = img.shape[:2]

        # Improve contrast before detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        boxes, weights = self._HOG.detectMultiScale(
            gray,
            winStride=(4, 4),
            padding=(8, 8),
            scale=1.02,
        )

        rects, scores = [], []
        for (x, y, bw, bh), score in zip(
            boxes if len(boxes) else [],
            weights if len(weights) else [],
        ):
            score = float(score)
            aspect = bh / float(bw)
            area   = bw * bh

            if score < HOG_SCORE_MIN:               continue
            if aspect < 1.4 or aspect > 3.0:        continue
            if area < 1500 or area > 0.35 * W * H:  continue
            if y + bh < 0.25 * H:                   continue   # skip sky/roof detections

            x1 = max(0, x - HOG_PADDING)
            y1 = max(0, y - HOG_PADDING)
            x2 = min(W, x + bw + HOG_PADDING)
            y2 = min(H, y + bh + HOG_PADDING)
            rects.append([x1, y1, x2, y2])
            scores.append(score)

        kept = _hog_nms(rects, scores)

        for i, (x1, y1, x2, y2) in enumerate(kept):
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img, f"Person {i}",
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
            )

        ann_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(ann_rgb), len(kept)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = JaywalkUI()
    app.mainloop()
