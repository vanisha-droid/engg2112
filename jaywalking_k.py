"""
jaywalking_kfold.py
===================
ENGG2112 - Jaywalking Detector | K-Fold Cross Validation

WHAT THIS FILE DOES
-------------------
Instead of a fixed train/val split, K-Fold CV trains the model K times,
each time using a different 1/K slice of the training data as the
validation fold. The final metrics are averaged across all K folds,
giving a much more reliable estimate of true model performance.

WHY K-FOLD?
-----------
With a small dataset (e.g. 73 images per class), a single 80/20 split
can be "lucky" or "unlucky" depending on which images end up in val.
K-Fold removes this variance by systematically rotating the val set.

    ┌───┬───┬───┬───┬───┐
    │ 1 │ 2 │ 3 │ 4 │ 5 │  ← 5 equal folds
    ├───┴───┴───┴───┴───┤
    │●  │   │   │   │   │  Fold 1: fold 1 = val, rest = train
    │   │●  │   │   │   │  Fold 2: fold 2 = val, rest = train
    │   │   │●  │   │   │  Fold 3: fold 3 = val, rest = train
    │   │   │   │●  │   │  Fold 4: fold 4 = val, rest = train
    │   │   │   │   │●  │  Fold 5: fold 5 = val, rest = train
    └───────────────────┘
    Final score = average of all 5 val results.

NOTE ON DATA AUGMENTATION
--------------------------
K-Fold is run on the ORIGINAL (non-augmented) data to get a clean
generalisation estimate. If you pass in the augmented directory, the
overlap between augmented images derived from the same original can
leak information across folds. Use cityscapes-dataset, not
cityscapes-aug-dataset, for the most honest evaluation.

USAGE
-----
    python jaywalking_kfold.py

After running, the best model (by val F1) is saved as:
    best_kfold_model.keras

HOW IT DIFFERS FROM jaywalking_cnn.py
--------------------------------------
- Uses StratifiedKFold so each fold has the same class ratio.
- Trains from a combined train+val pool (test/ is kept held-out).
- Reports per-fold and aggregate metrics.
- Saves the best model across all folds.
- Supports the same transfer learning flag as jaywalking_cnn.py.
"""

# ══════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════
import os
import random
import shutil
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight


# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

#no aug
STREET_SCENE_DIR = os.path.dirname(os.path.abspath(__file__))

CLASS_NAMES = ["jaywalk", "no_jaywalk"]
NUM_CLASSES  = len(CLASS_NAMES)

IMG_HEIGHT   = 224
IMG_WIDTH    = 224
IMG_CHANNELS = 3

# K-Fold settings
K_FOLDS       = 5      # Number of folds. 5 is the standard choice.
                       # With small datasets you can try K=10 for more folds.

# Training settings (per fold)
BATCH_SIZE    = 16
EPOCHS        = 30     # Max per fold. EarlyStopping will usually stop sooner.
LEARNING_RATE = 0.001
RANDOM_SEED   = 42

# Transfer learning: True = MobileNetV2 backbone (recommended)
USE_TRANSFER_LEARNING = True

# Fine-tune Phase 2 within each fold?
# Adds accuracy but is MUCH slower (roughly 2× training time per fold).
FINE_TUNE = False

# Classification threshold
THRESHOLD = 0.5

# Where to save outputs (plots, model, fold summaries)
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Augmentation inside each fold's training split (on-the-fly, no disk writes)
AUGMENT_TRAIN = True


# ══════════════════════════════════════════════════════════════
# STEP 1 — COLLECT ALL IMAGE PATHS FROM TRAIN/ AND VAL/
# ══════════════════════════════════════════════════════════════
#
# We merge the original train/ and val/ into one pool, then let
# StratifiedKFold re-split it. The test/ folder stays untouched.

VALID_EXT = {".jpg", ".jpeg", ".png"}

def collect_images(data_dir):
    """
    Walks train/ and val/ subdirectories and returns parallel lists:
        paths  : list of absolute file paths
        labels : integer class index for each path (0 or 1)

    Keras assigns class indices alphabetically, so:
        jaywalk    → 0
        no_jaywalk → 1
    """
    sorted_classes = sorted(CLASS_NAMES)   # alphabetical = Keras convention
    class_to_idx   = {cls: i for i, cls in enumerate(sorted_classes)}

    paths, labels = [], []

    for split in ("train", "val"):
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            print(f"  WARNING: {split_dir} not found — skipping.")
            continue

        for cls in sorted_classes:
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if os.path.splitext(fname)[1].lower() in VALID_EXT:
                    paths.append(os.path.join(cls_dir, fname))
                    labels.append(class_to_idx[cls])

    return paths, labels


def collect_test_images(data_dir):
    """Returns (paths, labels) for the held-out test/ set."""
    sorted_classes = sorted(CLASS_NAMES)
    class_to_idx   = {cls: i for i, cls in enumerate(sorted_classes)}

    paths, labels = [], []
    test_dir = os.path.join(data_dir, "test")
    if not os.path.isdir(test_dir):
        return paths, labels

    for cls in sorted_classes:
        cls_dir = os.path.join(test_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if os.path.splitext(fname)[1].lower() in VALID_EXT:
                paths.append(os.path.join(cls_dir, fname))
                labels.append(class_to_idx[cls])

    return paths, labels


# ══════════════════════════════════════════════════════════════
# STEP 2 — KERAS DATASET FROM PATH LISTS
# ══════════════════════════════════════════════════════════════
#
# We build a tf.data.Dataset directly from file paths.
# This avoids writing temporary folders to disk for each fold.

def path_label_to_dataset(paths, labels, augment=False, batch_size=BATCH_SIZE):
    """
    Creates a tf.data.Dataset from lists of file paths and integer labels.

    Args:
        paths   : list of image file paths
        labels  : list of integer class indices (0 or 1)
        augment : if True, apply random flips/colour jitter during map()
        batch_size: batch size for training

    Returns:
        A batched, prefetched tf.data.Dataset ready to pass to model.fit()
    """
    path_ds  = tf.data.Dataset.from_tensor_slices(paths)
    label_ds = tf.data.Dataset.from_tensor_slices(
        tf.cast(labels, tf.float32)
    )
    ds = tf.data.Dataset.zip((path_ds, label_ds))

    def load_and_preprocess(path, label):
        # Read and decode the image
        raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])

        if USE_TRANSFER_LEARNING:
            # MobileNetV2 expects pixels in [-1, 1]
            img = keras.applications.mobilenet_v2.preprocess_input(img)
        else:
            # Scratch CNN normalises inside the model (Rescaling layer),
            # but we still need float here. Keep [0, 255].
            img = tf.cast(img, tf.float32)

        return img, label

    def augment_fn(img, label):
        # Random horizontal flip
        img = tf.image.random_flip_left_right(img)
        # Random brightness (small delta relative to input range)
        img = tf.image.random_brightness(img, max_delta=0.2)
        # Random contrast
        img = tf.image.random_contrast(img, lower=0.7, upper=1.3)
        # Clip to valid range for MobileNetV2 [-1,1] or scratch [0,255]
        if USE_TRANSFER_LEARNING:
            img = tf.clip_by_value(img, -1.0, 1.0)
        else:
            img = tf.clip_by_value(img, 0.0, 255.0)
        return img, label

    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ══════════════════════════════════════════════════════════════
# STEP 3 — MODEL BUILDERS
# ══════════════════════════════════════════════════════════════
# (Same architecture as jaywalking_cnn.py )

def build_scratch_cnn():
    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    x = layers.Rescaling(1.0 / 255)(inputs)

    x = layers.Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = layers.Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(64, (3,3), activation="relu", padding="same")(x)
    x = layers.Conv2D(64, (3,3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = layers.Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    return keras.Model(inputs, outputs, name="Scratch_CNN")


def build_transfer_model():
    inputs    = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    x         = inputs   # preprocess_input applied in dataset pipeline

    base_model = keras.applications.MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="MobileNetV2_Transfer")
    model._base = base_model   # stash for fine-tuning
    return model


def compile_model(model, lr=LEARNING_RATE):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ]
    )


def apply_fine_tune(model, train_ds, val_ds, class_weights, fold_idx):
    """Unfreeze top 30 layers and fine-tune at 100× smaller LR."""
    if not hasattr(model, "_base"):
        return None

    base = model._base
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    compile_model(model, lr=LEARNING_RATE / 100)

    ft_ckpt = os.path.join(OUTPUT_DIR, f"_tmp_ft_fold{fold_idx}.keras")
    history_ft = model.fit(
        train_ds,
        epochs=15,
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5,
                restore_best_weights=True, verbose=0
            ),
            keras.callbacks.ModelCheckpoint(
                ft_ckpt, monitor="val_loss",
                save_best_only=True, verbose=0
            ),
        ],
        verbose=0
    )
    # Clean up temporary checkpoint
    if os.path.exists(ft_ckpt):
        os.remove(ft_ckpt)

    return history_ft


# ══════════════════════════════════════════════════════════════
# STEP 4 — K-FOLD TRAINING LOOP
# ══════════════════════════════════════════════════════════════

def run_kfold(all_paths, all_labels):
    """
    Main k-fold loop.

    Returns:
        fold_results : list of dicts, one per fold, with keys:
                       fold, accuracy, precision, recall, f1,
                       val_loss_history, val_acc_history
        best_model   : the Keras model with the highest val F1
        best_fold    : index (1-based) of the best fold
    """
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    all_paths  = np.array(all_paths)
    all_labels = np.array(all_labels)

    fold_results = []
    best_f1      = -1.0
    best_model   = None
    best_fold    = -1

    sorted_classes = sorted(CLASS_NAMES)

    print(f"\n{'='*60}")
    print(f"STARTING {K_FOLDS}-FOLD CROSS VALIDATION")
    print(f"  Total images (train+val pool): {len(all_paths)}")
    print(f"  Transfer learning: {USE_TRANSFER_LEARNING}")
    print(f"  Fine-tuning: {FINE_TUNE}")
    print(f"{'='*60}\n")

    for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(all_paths, all_labels), start=1):

        print(f"{'─'*60}")
        print(f"FOLD {fold_idx}/{K_FOLDS}")
        print(f"  Train: {len(train_idx)} images  |  Val: {len(val_idx)} images")
        print(f"{'─'*60}")

        # Split paths and labels
        train_paths  = all_paths[train_idx].tolist()
        train_labels = all_labels[train_idx].tolist()
        val_paths    = all_labels[val_idx]          # labels for class weight
        val_paths_s  = all_paths[val_idx].tolist()
        val_labels   = all_labels[val_idx].tolist()

        # Class weights from this fold's training split
        unique_cls = np.unique(train_labels)
        cw_values  = compute_class_weight(
            class_weight="balanced",
            classes=unique_cls,
            y=train_labels
        )
        class_weights = dict(enumerate(cw_values))
        print(f"  Class weights: {class_weights}")

        # Print class distribution for this fold
        unique, counts = np.unique(train_labels, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"    {sorted_classes[u]}: {c} train images")

        # Build tf.data pipelines
        train_ds = path_label_to_dataset(
            train_paths, train_labels,
            augment=AUGMENT_TRAIN, batch_size=BATCH_SIZE
        )
        val_ds = path_label_to_dataset(
            val_paths_s, val_labels,
            augment=False, batch_size=BATCH_SIZE
        )

        # Build a fresh model for this fold
        # (IMPORTANT: must rebuild weights each fold, not reuse)
        tf.keras.backend.clear_session()
        if USE_TRANSFER_LEARNING:
            model = build_transfer_model()
        else:
            model = build_scratch_cnn()

        compile_model(model)

        # Phase 1 training
        ckpt_path = os.path.join(OUTPUT_DIR, f"_tmp_fold{fold_idx}.keras")
        history = model.fit(
            train_ds,
            epochs=EPOCHS,
            validation_data=val_ds,
            class_weight=class_weights,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=8,
                    restore_best_weights=True, verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5,
                    patience=3, min_lr=1e-7, verbose=1
                ),
                keras.callbacks.ModelCheckpoint(
                    ckpt_path, monitor="val_loss",
                    save_best_only=True, verbose=0
                ),
            ],
            verbose=1
        )

        # Phase 2: optional fine-tuning
        if FINE_TUNE and USE_TRANSFER_LEARNING:
            print(f"\n  Fine-tuning fold {fold_idx}...")
            apply_fine_tune(model, train_ds, val_ds, class_weights, fold_idx)

        # ── Evaluate on the validation fold ──────────────────
        # Predict probabilities on the val set
        y_prob = []
        y_true = []

        # Build a simple unbatched prediction dataset
        pred_ds = path_label_to_dataset(
            val_paths_s, val_labels,
            augment=False, batch_size=1
        )
        for img_batch, lbl_batch in pred_ds:
            prob = float(model.predict(img_batch, verbose=0)[0][0])
            y_prob.append(prob)
            y_true.append(int(lbl_batch.numpy()[0]))

        y_pred = [1 if p >= THRESHOLD else 0 for p in y_prob]

        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)

        print(f"\n  Fold {fold_idx} results:")
        print(f"    Accuracy : {acc:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall   : {rec:.4f}")
        print(f"    F1-score : {f1:.4f}")

        fold_results.append({
            "fold"           : fold_idx,
            "accuracy"       : acc,
            "precision"      : prec,
            "recall"         : rec,
            "f1"             : f1,
            "y_true"         : y_true,
            "y_pred"         : y_pred,
            "val_loss"       : history.history.get("val_loss", []),
            "val_accuracy"   : history.history.get("val_accuracy", []),
            "train_loss"     : history.history.get("loss", []),
            "train_accuracy" : history.history.get("accuracy", []),
        })

        # Track best model
        if f1 > best_f1:
            best_f1    = f1
            best_model = model
            best_fold  = fold_idx
            print(f"  ★ New best model (F1={best_f1:.4f})")

        # Clean up temporary checkpoint
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

        print()

    return fold_results, best_model, best_fold


# ══════════════════════════════════════════════════════════════
# STEP 5 — AGGREGATE RESULTS
# ══════════════════════════════════════════════════════════════

def print_summary(fold_results, best_fold):
    """Prints a formatted summary table of all fold results."""
    metrics = ["accuracy", "precision", "recall", "f1"]

    print(f"\n{'='*60}")
    print(f"K-FOLD CROSS VALIDATION SUMMARY  ({K_FOLDS} folds)")
    print(f"{'='*60}")
    print(f"{'Fold':<6} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"{'─'*48}")

    for r in fold_results:
        marker = " ★" if r["fold"] == best_fold else ""
        print(f"  {r['fold']:<4} "
              f"{r['accuracy']:>10.4f} "
              f"{r['precision']:>10.4f} "
              f"{r['recall']:>10.4f} "
              f"{r['f1']:>10.4f}{marker}")

    print(f"{'─'*48}")

    for m in metrics:
        vals = [r[m] for r in fold_results]
        print(f"  {'Mean ' + m:<10} "
              f"{'':>10}" * (metrics.index(m))
              + f"{np.mean(vals):>10.4f}  "
              + f"(± {np.std(vals):.4f})")

    print(f"\n  ★ Best fold by F1: Fold {best_fold}")
    print(f"{'='*60}\n")


# ══════════════════════════════════════════════════════════════
# STEP 6 — PLOT RESULTS
# ══════════════════════════════════════════════════════════════

def plot_fold_curves(fold_results):
    """
    Plots training and validation loss/accuracy for all folds,
    overlaid on the same axes with fold number in the legend.
    Saves to OUTPUT_DIR.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"ENGG2112 — {K_FOLDS}-Fold CV Training Curves", fontsize=14)

    cmap = plt.cm.get_cmap("tab10", K_FOLDS)

    ax_tl, ax_ta = axes[0]  # top row: loss, accuracy
    ax_vl, ax_va = axes[1]  # bottom row: same for validation

    for r in fold_results:
        c = cmap(r["fold"] - 1)
        label = f"Fold {r['fold']}"
        epochs = range(1, len(r["train_loss"]) + 1)

        ax_tl.plot(epochs, r["train_loss"],     color=c, label=label)
        ax_ta.plot(epochs, r["train_accuracy"],  color=c, label=label)
        ax_vl.plot(epochs, r["val_loss"],        color=c, label=label, linestyle="--")
        ax_va.plot(epochs, r["val_accuracy"],    color=c, label=label, linestyle="--")

    for ax, title, ylabel in [
        (ax_tl, "Training Loss",      "Loss"),
        (ax_ta, "Training Accuracy",  "Accuracy"),
        (ax_vl, "Validation Loss",    "Loss"),
        (ax_va, "Validation Accuracy","Accuracy"),
    ]:
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    ax_va.axhline(y=0.80, color="black", linestyle=":", alpha=0.7, label="Target 0.80")

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "kfold_training_curves.png")
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Training curves plot saved → {save_path}")


def plot_fold_metrics(fold_results):
    """
    Bar chart comparing Accuracy / Precision / Recall / F1 across folds,
    with mean ± std shown as a dashed line.
    """
    metrics = ["accuracy", "precision", "recall", "f1"]
    folds   = [r["fold"] for r in fold_results]
    x       = np.arange(K_FOLDS)
    width   = 0.20

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(f"ENGG2112 — Per-Fold Metrics ({K_FOLDS}-Fold CV)", fontsize=13)

    colours = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for i, (metric, colour) in enumerate(zip(metrics, colours)):
        vals = [r[metric] for r in fold_results]
        bars = ax.bar(x + i * width, vals, width, label=metric.capitalize(),
                      color=colour, alpha=0.85)
        # Mean line
        ax.axhline(np.mean(vals), color=colour, linestyle="--",
                   linewidth=1.0, alpha=0.6)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "kfold_metrics.png")
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Metrics bar chart saved → {save_path}")


# ══════════════════════════════════════════════════════════════
# STEP 7 — EVALUATE BEST MODEL ON HELD-OUT TEST SET
# ══════════════════════════════════════════════════════════════

def evaluate_on_test(model, test_paths, test_labels):
    """Evaluates the best fold model on the held-out test set."""
    if not test_paths:
        print("\nNo test images found — skipping held-out evaluation.")
        return

    sorted_classes = sorted(CLASS_NAMES)
    idx_to_name    = {i: n for i, n in enumerate(sorted_classes)}

    print(f"\n{'='*60}")
    print("HELD-OUT TEST SET EVALUATION (best fold model)")
    print(f"{'='*60}")
    print(f"  Test images: {len(test_paths)}")

    test_ds = path_label_to_dataset(
        test_paths, test_labels,
        augment=False, batch_size=1
    )

    y_prob, y_true = [], []
    for img_batch, lbl_batch in test_ds:
        prob = float(model.predict(img_batch, verbose=0)[0][0])
        y_prob.append(prob)
        y_true.append(int(lbl_batch.numpy()[0]))

    y_pred       = [1 if p >= THRESHOLD else 0 for p in y_prob]
    target_names = [idx_to_name[i] for i in range(NUM_CLASSES)]

    print(f"\nThreshold: {THRESHOLD}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=target_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix (rows=actual, cols=predicted):")
    header = " " * 16 + "  ".join(f"{n[:10]:>10}" for n in target_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>10}" for v in row)
        print(f"  {target_names[i][:14]:>14}  {row_str}")

    # Per-image breakdown
    print(f"\n{'─'*70}")
    print(f"{'IMAGE':<40} {'TRUE':<12} {'PRED':<12} {'CONF':>8}")
    print(f"{'─'*70}")
    for fpath, tl, pl, prob in zip(test_paths, y_true, y_pred, y_prob):
        conf     = prob if pl == 1 else 1 - prob
        correct  = "✓" if tl == pl else "✗"
        print(f"{os.path.basename(fpath):<40} "
              f"{idx_to_name[tl]:<12} "
              f"{idx_to_name[pl]:<12} "
              f"{conf:>7.1%}  {correct}")
    print(f"{'─'*70}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    print("=" * 60)
    print("ENGG2112 — Jaywalking Detector | K-Fold Cross Validation")
    print("=" * 60)
    print(f"  Dataset : {STREET_SCENE_DIR}")
    print(f"  K folds : {K_FOLDS}")
    print(f"  Transfer: {USE_TRANSFER_LEARNING}  |  Fine-tune: {FINE_TUNE}")
    print(f"  Augment : {AUGMENT_TRAIN}")
    print(f"  Epochs  : up to {EPOCHS} (EarlyStopping patience=8)")

    if not os.path.isdir(STREET_SCENE_DIR):
        print(f"\nERROR: STREET_SCENE_DIR not found:\n  {STREET_SCENE_DIR}")
        return

    # ── 1. Collect images ─────────────────────────────────────
    print("\nCollecting train+val images...")
    all_paths, all_labels = collect_images(STREET_SCENE_DIR)

    if len(all_paths) == 0:
        print("ERROR: No images found in train/ or val/. "
              "Check STREET_SCENE_DIR and folder structure.")
        return

    print(f"  Found {len(all_paths)} total images in pool.")
    unique, counts = np.unique(all_labels, return_counts=True)
    sorted_classes = sorted(CLASS_NAMES)
    for u, c in zip(unique, counts):
        print(f"    {sorted_classes[u]}: {c} images")

    # Collect test set separately — never touched during CV
    test_paths, test_labels = collect_test_images(STREET_SCENE_DIR)
    print(f"  Test set (held out): {len(test_paths)} images")

    # ── 2. Run K-Fold ─────────────────────────────────────────
    fold_results, best_model, best_fold = run_kfold(all_paths, all_labels)

    # ── 3. Print summary ──────────────────────────────────────
    print_summary(fold_results, best_fold)

    # ── 4. Plot results ───────────────────────────────────────
    plot_fold_curves(fold_results)
    plot_fold_metrics(fold_results)

    # ── 5. Evaluate best model on test set ────────────────────
    evaluate_on_test(best_model, test_paths, test_labels)

    # ── 6. Save best model ────────────────────────────────────
    model_path = os.path.join(OUTPUT_DIR, "best_kfold_model.keras")
    best_model.save(model_path)
    print(f"\nBest model (Fold {best_fold}) saved → {model_path}")

    # Aggregate metric summary for quick reference
    metrics = ["accuracy", "precision", "recall", "f1"]
    print("\nFinal aggregate metrics (mean ± std across folds):")
    for m in metrics:
        vals = [r[m] for r in fold_results]
        print(f"  {m.capitalize():<12}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()