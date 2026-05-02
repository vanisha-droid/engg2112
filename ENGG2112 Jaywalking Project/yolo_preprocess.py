"""
yolo_preprocess.py
==================
ENGG2112 - Jaywalking Detector | Team: Victoria, Shiyao, Vanisha, Rejaksi

WHAT THIS FILE DOES
-------------------
This script processes raw Cityscapes images (which have NO bounding boxes)
and produces annotated versions with bounding boxes drawn around every
detected person. These annotated images can then be fed into the trained CNN.

WHY DO WE NEED THIS?
--------------------
The CNN was trained on Street Scene images that already have bounding boxes
drawn on them. When we test on Cityscapes images, we need to first draw
similar bounding boxes so the CNN sees the same kind of input it was
trained on.

HOW IT WORKS
------------
1. Load a pre-trained YOLOv8 model (already knows how to detect people)
2. For each Cityscapes image:
   a. Run YOLO → get bounding box coordinates for every detected person
   b. Draw those boxes on the image in red rectangles
   c. Save the annotated image to an output folder
3. (Optional) Also run the CNN on the annotated images and compute metrics

YOLO (You Only Look Once):
    A CNN itself, pre-trained on millions of images.
    It divides the image into a grid, and each grid cell predicts
    bounding boxes + confidence scores for objects it can see.
    We use it as a tool — we don't train it; it already works.
    YOLOv8 is the 2023 version and is very accurate and fast.

HOW TO RUN
----------
1. pip install ultralytics opencv-python tensorflow
2. Set CITYSCAPES_RAW_DIR and OUTPUT_DIR below.
3. Run: python yolo_preprocess.py
   - Annotated images will appear in OUTPUT_DIR/annotated/
   - If EVALUATE=True, CNN predictions are also printed.

FOLDER STRUCTURE ASSUMED
------------------------
cityscapes_raw/
    image1.jpg
    image2.jpg
    ...        (raw, unannotated images — any arrangement works)

If you have ground-truth labels for evaluation:
cityscapes_raw/
    jaywalk/
        img1.jpg ...
    no_jaywalk/
        img2.jpg ...

Set HAS_LABELS = True in that case.
"""

import os
import sys
import numpy as np
import cv2    # OpenCV — for reading, drawing on, and saving images

# ── Check ultralytics is installed ──────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed. Run:  pip install ultralytics")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════
# CONFIGURATION — CHANGE THESE
# ══════════════════════════════════════════════════════════════

# Path to raw Cityscapes images (no bounding boxes yet)
CITYSCAPES_RAW_DIR = r"C:\path\to\cityscapes_raw"   # <-- CHANGE THIS

# Where to save the annotated images (created automatically if it doesn't exist)
OUTPUT_DIR = r"C:\path\to\output_annotated"         # <-- CHANGE THIS

# Path to your trained CNN model (output of jaywalking_cnn.py)
CNN_MODEL_PATH = r"jaywalking_cnn.keras"            # <-- CHANGE IF NEEDED

# Set to True if your Cityscapes images are already sorted into jaywalk/ and
# no_jaywalk/ subfolders (so we can compute accuracy/F1 at the end).
# Set to False if they're all in one folder with no labels.
HAS_LABELS = False

# YOLO settings
YOLO_MODEL     = "yolov8n.pt"   # yolov8n = nano (smallest, fastest).
                                  # Options: yolov8s.pt, yolov8m.pt, yolov8l.pt
                                  # Larger = more accurate but slower.
YOLO_CONF      = 0.40            # Minimum confidence to accept a detection.
                                  # 0.40 = YOLO must be ≥40% sure it's a person.
YOLO_PERSON_ID = 0               # In COCO (YOLO's training dataset), class 0 = "person"

# CNN classification threshold
# Probability above this → predict "jaywalk"
CNN_THRESHOLD = 0.5

# Image size for CNN (must match what you used in jaywalking_cnn.py)
CNN_IMG_SIZE = (224, 224)


# ══════════════════════════════════════════════════════════════
# STEP 1 — LOAD MODELS
# ══════════════════════════════════════════════════════════════

def load_models():
    """Loads the YOLO detector and the trained CNN classifier."""

    # Load YOLO — if the .pt file doesn't exist locally, ultralytics downloads it.
    print(f"Loading YOLO model: {YOLO_MODEL} ...")
    yolo = YOLO(YOLO_MODEL)
    print("  YOLO loaded OK.")

    # Load our trained CNN
    print(f"Loading CNN model: {CNN_MODEL_PATH} ...")
    try:
        import tensorflow as tf
        cnn = tf.keras.models.load_model(CNN_MODEL_PATH)
        print("  CNN loaded OK.")
    except Exception as e:
        print(f"  WARNING: Could not load CNN — {e}")
        print("  Bounding-box annotation will still work without CNN.")
        cnn = None

    return yolo, cnn


# ══════════════════════════════════════════════════════════════
# STEP 2 — RUN YOLO ON ONE IMAGE
# ══════════════════════════════════════════════════════════════
#
# Plain English:
#   YOLO takes a raw photo and returns a list of detected objects.
#   Each detection has:
#     - xyxy: the pixel coordinates of the bounding box corners
#             (x_top_left, y_top_left, x_bottom_right, y_bottom_right)
#     - conf: confidence score (0 to 1, how sure YOLO is)
#     - cls:  class index (0 = person, 2 = car, etc. in COCO)
#
#   We only keep detections where cls == 0 (person) and
#   confidence >= YOLO_CONF.

def detect_persons(yolo_model, image_bgr):
    """
    Runs YOLO on a single image and returns bounding boxes for detected persons.

    Args:
        yolo_model: loaded YOLO model
        image_bgr:  image as a NumPy array in BGR colour (how OpenCV loads images)

    Returns:
        boxes: list of [x1, y1, x2, y2] bounding box coordinates (pixel units)
               x1,y1 = top-left corner; x2,y2 = bottom-right corner
    """
    # Run YOLO. verbose=False suppresses the per-frame log spam.
    results = yolo_model(image_bgr, verbose=False)

    boxes = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])        # object class index
            conf   = float(box.conf[0])     # confidence score

            # Only keep "person" detections above our confidence threshold
            if cls_id == YOLO_PERSON_ID and conf >= YOLO_CONF:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes.append([int(x1), int(y1), int(x2), int(y2)])

    return boxes


# ══════════════════════════════════════════════════════════════
# STEP 3 — DRAW BOUNDING BOXES ON IMAGE
# ══════════════════════════════════════════════════════════════
#
# Plain English:
#   Once we know where the people are (from YOLO), we use OpenCV to
#   draw red rectangles on the image at those coordinates.
#   This creates an image that looks like the Street Scene training
#   images, so our CNN can make sense of it.

def draw_boxes(image_bgr, boxes, label="person", colour=(0, 0, 255)):
    """
    Draws bounding boxes on a copy of the image and returns the annotated version.

    Args:
        image_bgr: original image (NumPy array, BGR colour order)
        boxes:     list of [x1, y1, x2, y2] from detect_persons()
        label:     text to show above each box
        colour:    BGR colour tuple. Default = red (0, 0, 255).

    Returns:
        annotated: image with boxes drawn on it (same shape as input)
    """
    annotated = image_bgr.copy()   # don't modify the original

    for (x1, y1, x2, y2) in boxes:
        # Draw rectangle: (image, top-left, bottom-right, colour, thickness)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, thickness=2)

        # Draw a filled label background for readability
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated,
                      (x1, y1 - label_size[1] - 4),
                      (x1 + label_size[0], y1),
                      colour, -1)  # -1 = filled

        # Draw label text in white
        cv2.putText(annotated, label, (x1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return annotated


# ══════════════════════════════════════════════════════════════
# STEP 4 — CNN CLASSIFICATION ON ANNOTATED IMAGE
# ══════════════════════════════════════════════════════════════
#
# Plain English:
#   After drawing the bounding boxes, we feed the annotated image
#   into our trained CNN to get a jaywalking prediction.
#
#   The CNN needs:
#     1. Image resized to 224×224
#     2. Pixel values normalised to [0, 1]
#     3. A "batch" dimension added (shape must be (1, 224, 224, 3))
#        because the CNN always expects batches of images, even if
#        there's just one.

def cnn_predict(cnn_model, image_bgr):
    """
    Runs the trained CNN on an annotated image.

    Returns:
        prob (float): probability of jaywalking [0 to 1]
        label (str):  "jaywalk" or "no_jaywalk"
    """
    # Convert BGR (OpenCV default) → RGB (what the CNN was trained on)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Resize to the CNN's expected input size
    image_resized = cv2.resize(image_rgb, CNN_IMG_SIZE, interpolation=cv2.INTER_AREA)

    # Normalise pixels from [0, 255] → [0, 1]
    image_norm = image_resized.astype(np.float32) / 255.0

    # Add batch dimension: (224, 224, 3) → (1, 224, 224, 3)
    # The CNN always works with batches; even a single image needs this.
    image_batch = np.expand_dims(image_norm, axis=0)

    # Get the CNN's raw output (a sigmoid probability close to 0 or 1)
    prob = float(cnn_model.predict(image_batch, verbose=0)[0][0])

    label = "jaywalk" if prob >= CNN_THRESHOLD else "no_jaywalk"
    return prob, label


# ══════════════════════════════════════════════════════════════
# STEP 5 — PROCESS ALL CITYSCAPES IMAGES
# ══════════════════════════════════════════════════════════════

def find_images(root_dir):
    """
    Recursively finds all .jpg/.png images under root_dir.

    Returns a list of (filepath, true_label_or_None) tuples.
    If HAS_LABELS=True, the parent folder name is used as the label.
    """
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    found = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in valid_ext:
                fpath = os.path.join(dirpath, fname)
                # Use parent folder name as label if HAS_LABELS is True
                parent_name = os.path.basename(dirpath)
                label = parent_name if HAS_LABELS else None
                found.append((fpath, label))

    return found


def process_all_images(yolo_model, cnn_model, image_list):
    """
    Runs the full pipeline on every image:
        raw image → YOLO → annotated image → CNN → prediction

    Saves annotated images to OUTPUT_DIR/annotated/.
    Returns results list for evaluation.
    """
    annotated_dir = os.path.join(OUTPUT_DIR, "annotated")
    os.makedirs(annotated_dir, exist_ok=True)

    results = []   # [(true_label, pred_label, prob, filepath), ...]
    n = len(image_list)

    print(f"\n{'='*55}")
    print(f"PROCESSING {n} CITYSCAPES IMAGES")
    print(f"{'='*55}")
    print(f"Annotated output folder: {annotated_dir}\n")

    for i, (fpath, true_label) in enumerate(image_list, 1):
        # ── Load image ────────────────────────────────────────
        img_bgr = cv2.imread(fpath)
        if img_bgr is None:
            print(f"  [{i}/{n}] SKIP (unreadable): {os.path.basename(fpath)}")
            continue

        # ── YOLO: detect all persons ──────────────────────────
        boxes = detect_persons(yolo_model, img_bgr)

        # ── Draw bounding boxes on the image ──────────────────
        # If no person was detected, the image goes through without boxes.
        # The CNN may still produce a prediction, but it will be less reliable.
        annotated = draw_boxes(img_bgr, boxes)

        # ── Optionally stamp box count on the image ────────────
        n_boxes_text = f"{len(boxes)} person(s) detected"
        cv2.putText(annotated, n_boxes_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # ── Save annotated image ───────────────────────────────
        out_fname = f"annotated_{os.path.basename(fpath)}"
        out_path  = os.path.join(annotated_dir, out_fname)
        cv2.imwrite(out_path, annotated)

        # ── CNN: classify the annotated image ─────────────────
        pred_label = "N/A"
        prob       = None
        if cnn_model is not None:
            prob, pred_label = cnn_predict(cnn_model, annotated)

        results.append((true_label, pred_label, prob, fpath))

        # Print progress
        correct_str = ""
        if HAS_LABELS and true_label is not None:
            correct_str = " ✓" if pred_label == true_label else " ✗"
        prob_str = f"prob={prob:.3f}" if prob is not None else ""
        print(f"  [{i:>4}/{n}] {os.path.basename(fpath):<40} "
              f"boxes={len(boxes)}  pred={pred_label:<12} {prob_str}{correct_str}")

    return results


# ══════════════════════════════════════════════════════════════
# STEP 6 — COMPUTE METRICS (if labels available)
# ══════════════════════════════════════════════════════════════

def compute_metrics(results):
    """
    Computes accuracy, precision, recall, F1 if HAS_LABELS is True.
    """
    if not HAS_LABELS:
        print("\nNo labels available — skipping metric computation.")
        print("Set HAS_LABELS=True if your images are in jaywalk/no_jaywalk subfolders.")
        return

    from sklearn.metrics import classification_report, confusion_matrix

    y_true = [r[0] for r in results if r[0] is not None and r[1] != "N/A"]
    y_pred = [r[1] for r in results if r[0] is not None and r[1] != "N/A"]

    if len(y_true) == 0:
        print("\nNo labelled results to evaluate.")
        return

    print(f"\n{'='*55}")
    print("CITYSCAPES EVALUATION METRICS")
    print(f"{'='*55}")
    print(classification_report(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(set(y_true))
    print("Confusion Matrix:")
    header = " " * 14 + "  ".join(f"{l[:12]:>12}" for l in labels)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {labels[i][:12]:>12}  " + "  ".join(f"{v:>12}" for v in row))


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("ENGG2112 — YOLO Preprocessing + CNN Evaluation")
    print("=" * 55)

    # ── Load models ───────────────────────────────────────────
    yolo_model, cnn_model = load_models()

    # ── Find all images ───────────────────────────────────────
    image_list = find_images(CITYSCAPES_RAW_DIR)
    if len(image_list) == 0:
        print(f"\nERROR: No images found in {CITYSCAPES_RAW_DIR}")
        sys.exit(1)
    print(f"\nFound {len(image_list)} images to process.")

    # ── Process each image ────────────────────────────────────
    results = process_all_images(yolo_model, cnn_model, image_list)

    # ── Compute metrics ───────────────────────────────────────
    compute_metrics(results)

    print(f"\nAll done! Annotated images saved to: {os.path.join(OUTPUT_DIR, 'annotated')}")


if __name__ == "__main__":
    main()