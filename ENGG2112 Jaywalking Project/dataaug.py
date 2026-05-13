"""
dataaug.py
==========
ENGG2112 - Jaywalking Detector

Reads from INPUT_DIR (cityscapes-dataset) and writes a new OUTPUT_DIR
(cityscapes-aug-dataset) with the same train/val/test + class subfolder
structure.  Originals are copied first, then augmented images are added
until each folder hits its target count.

Test folders are copied as-is — no augmentation on held-out data.

USAGE
-----
    python dataaug.py
"""

import os
import shutil
import random
import tensorflow as tf

# ── PATHS ─────────────────────────────────────────────────────────────────────
INPUT_DIR  = r"C:\Users\vkolm\OneDrive\Documents\ENGG2112\data\cityscapes-dataset"
OUTPUT_DIR = r"C:\Users\vkolm\OneDrive\Documents\ENGG2112\data\cityscapes-aug-dataset"

# ── TARGETS ───────────────────────────────────────────────────────────────────
TRAIN_TARGET = 320    # images per class in train/  (73 originals → 320)
VAL_TARGET   = 30     # images per class in val/    (0-9 originals → 30)
# test/ is copied unchanged — do not augment held-out evaluation data.

# ── SETTINGS ──────────────────────────────────────────────────────────────────
JPEG_QUALITY = 92
RANDOM_SEED  = 42
VALID_EXT    = {".jpg", ".jpeg", ".png"}


# ── AUGMENTATION TRANSFORMS ───────────────────────────────────────────────────

def flip_lr(img):
    return tf.image.flip_left_right(img)

def random_brightness(img):
    delta = random.uniform(-0.35, 0.35)
    return tf.clip_by_value(tf.image.adjust_brightness(img, delta), 0.0, 1.0)

def random_contrast(img):
    factor = random.uniform(0.6, 1.6)
    return tf.clip_by_value(tf.image.adjust_contrast(img, factor), 0.0, 1.0)

def random_saturation(img):
    factor = random.uniform(0.4, 2.5)
    return tf.clip_by_value(tf.image.adjust_saturation(img, factor), 0.0, 1.0)

def random_hue(img):
    delta = random.uniform(-0.08, 0.08)
    return tf.clip_by_value(tf.image.adjust_hue(img, delta), 0.0, 1.0)

def add_noise(img):
    factor = random.uniform(0.02, 0.08)
    noise  = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=factor)
    return tf.clip_by_value(img + noise, 0.0, 1.0)

def random_crop_and_resize(img):
    """Crop a random 80-100% region then resize back to original dimensions."""
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    frac   = random.uniform(0.80, 1.0)
    crop_h = tf.cast(tf.cast(h, tf.float32) * frac, tf.int32)
    crop_w = tf.cast(tf.cast(w, tf.float32) * frac, tf.int32)
    cropped = tf.image.random_crop(img, size=[crop_h, crop_w, 3])
    return tf.image.resize(cropped, [h, w])

def random_augment(img):
    if random.random() < 0.60:
        img = flip_lr(img)
    if random.random() < 0.80:
        img = random_brightness(img)
    if random.random() < 0.60:
        img = random_contrast(img)
    if random.random() < 0.40:
        img = random_saturation(img)
    if random.random() < 0.25:
        img = random_hue(img)
    if random.random() < 0.30:
        img = add_noise(img)
    if random.random() < 0.50:
        img = random_crop_and_resize(img)
    return img


# ── I/O HELPERS ───────────────────────────────────────────────────────────────

def load_image(path):
    raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(raw, channels=3)
    return tf.cast(img, tf.float32) / 255.0

def save_image(img_tensor, path):
    uint8   = tf.cast(tf.clip_by_value(img_tensor * 255.0, 0, 255), tf.uint8)
    encoded = tf.image.encode_jpeg(uint8, quality=JPEG_QUALITY)
    tf.io.write_file(path, encoded)

def image_files(folder):
    return [
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in VALID_EXT
    ]


# ── CORE FUNCTIONS ────────────────────────────────────────────────────────────

def copy_folder(src, dst):
    """Copy all images from src to dst (creates dst if needed)."""
    os.makedirs(dst, exist_ok=True)
    files = image_files(src)
    for f in files:
        shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
    return len(files)


def augment_to_target(src_folder, dst_folder, target):
    """
    Copy originals from src_folder to dst_folder, then generate augmented
    images (sourced only from originals in src_folder) until dst_folder
    contains `target` images.

    Args:
        src_folder : source class folder (read-only — never modified)
        dst_folder : destination class folder (already populated with originals)
        target     : desired total count in dst_folder after augmentation
    """
    os.makedirs(dst_folder, exist_ok=True)

    originals = image_files(src_folder)   # source of truth for augmentation
    current   = len(image_files(dst_folder))

    if not originals:
        print(f"    ERROR: no source images in {src_folder}")
        return

    needed = max(0, target - current)
    print(f"    originals: {len(originals)}  copied: {current}  "
          f"to generate: {needed}  target: {target}")

    for i in range(needed):
        src_name  = random.choice(originals)
        stem, _   = os.path.splitext(src_name)
        out_name  = f"aug_{i:04d}_{stem}.jpg"
        out_path  = os.path.join(dst_folder, out_name)

        try:
            img = load_image(os.path.join(src_folder, src_name))
            img = random_augment(img)
            save_image(img, out_path)
        except Exception as e:
            print(f"    WARNING: failed on {src_name}: {e}")

    final = len(image_files(dst_folder))
    print(f"    done — {dst_folder.split(os.sep)[-3:][-1]}/{dst_folder.split(os.sep)[-2]}/{dst_folder.split(os.sep)[-1]} now has {final} images")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    if not os.path.isdir(INPUT_DIR):
        print(f"ERROR: INPUT_DIR not found:\n  {INPUT_DIR}")
        return

    classes = ["jaywalk", "no_jaywalk"]

    print("=" * 60)
    print("DATA AUGMENTATION — copy + augment to new folder")
    print("=" * 60)
    print(f"Input  : {INPUT_DIR}")
    print(f"Output : {OUTPUT_DIR}")
    print(f"Targets: train={TRAIN_TARGET}/class  val={VAL_TARGET}/class")
    print("Test   : copied unchanged\n")

    # ── Train: copy then augment to target ───────────────────
    print("── TRAIN ─────────────────────────────────────────────")
    for cls in classes:
        src = os.path.join(INPUT_DIR,  "train", cls)
        dst = os.path.join(OUTPUT_DIR, "train", cls)
        if not os.path.isdir(src):
            print(f"  SKIP (not found): {src}")
            continue
        n_copied = copy_folder(src, dst)
        print(f"\n  {cls}/ — {n_copied} originals copied")
        augment_to_target(src, dst, TRAIN_TARGET)

    # ── Val: copy then augment to target ─────────────────────
    print("\n── VAL ───────────────────────────────────────────────")
    for cls in classes:
        src = os.path.join(INPUT_DIR,  "val", cls)
        dst = os.path.join(OUTPUT_DIR, "val", cls)
        if not os.path.isdir(src):
            print(f"  SKIP (not found): {src}")
            continue
        n_copied = copy_folder(src, dst)
        print(f"\n  {cls}/ — {n_copied} originals copied")
        augment_to_target(src, dst, VAL_TARGET)

    # ── Test: copy only, no augmentation ─────────────────────
    print("\n── TEST (copy only, no augmentation) ─────────────────")
    for cls in classes:
        src = os.path.join(INPUT_DIR,  "test", cls)
        dst = os.path.join(OUTPUT_DIR, "test", cls)
        if not os.path.isdir(src):
            print(f"  SKIP (not found): {src}")
            continue
        n = copy_folder(src, dst)
        print(f"  {cls}/ — {n} images copied")

    print("\n" + "=" * 60)
    print("DONE")
    print(f"Update STREET_SCENE_DIR in jaywalking_cnn.py to:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
