"""
jaywalking_cnn.py
=================
ENGG2112 - Jaywalking Detector CNN
Team: Victoria Kolmac, Shiyao Lin, Vanisha Goyal, Rejaksi G

What this file does:
    Defines, trains, and evaluates a Convolutional Neural Network (CNN)
    that classifies images into four jaywalking risk levels:
        0 = no_jaywalk
        1 = low_risk
        2 = medium_risk
        3 = high_risk

How to use:
    1. Make sure your data is in the folder structure described in README.md
    2. Set DATA_DIR (below) to your dataset folder path
    3. Run:  python jaywalking_cnn.py

Dependencies:
    pip install tensorflow scikit-learn matplotlib numpy opencv-python
"""

# ─────────────────────────────────────────────────────────────
# IMPORTS
# Standard Python libraries and third-party packages we need.
# ─────────────────────────────────────────────────────────────

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Saves plots as files instead of opening a window.
                        # Works on all systems including headless servers.
import matplotlib.pyplot as plt

# TensorFlow / Keras — our deep learning framework.
# Keras is the user-friendly "front-end" built into TensorFlow.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Scikit-learn — for train/test splitting and evaluation metrics.
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# All the settings for the project in one place.
# Change these as needed — no need to hunt through the code.
# ─────────────────────────────────────────────────────────────

# --- Data ---
# Path to your dataset folder (see README.md for expected structure).
# Use a raw string (r"...") on Windows to avoid backslash issues.
DATA_DIR = r"C:\path\to\your\dataset"   # <-- CHANGE THIS

# The four class folders that must exist inside DATA_DIR.
CLASS_NAMES = ["no_jaywalk", "low_risk", "medium_risk", "high_risk"]

# --- Image settings ---
# All images will be resized to this. Must be consistent across all images.
# 224x224 is required if using transfer learning (MobileNetV2).
# 128x128 is fine if training from scratch and your PC is slow.
IMG_HEIGHT   = 224
IMG_WIDTH    = 224
IMG_CHANNELS = 3   # 3 = RGB colour. Change to 1 only if images are grayscale.

# --- Training settings ---
BATCH_SIZE    = 16      # How many images to process per weight update.
                        # Smaller = more updates, uses less memory.
EPOCHS        = 30      # Maximum training passes through the full dataset.
                        # EarlyStopping (below) will stop sooner if needed.
LEARNING_RATE = 0.001   # How big a step the optimizer takes each update.
RANDOM_SEED   = 42      # Makes results reproducible.

# --- Transfer learning switch ---
# True  = use MobileNetV2 (pre-trained, better for small datasets)
# False = train CNN from scratch (simpler, no pre-trained weights needed)
USE_TRANSFER_LEARNING = True

# --- Output ---
# Saves plots and model next to this script file.
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────
# STEP 1: LOAD DATA
#
# Reads images from disk and returns two arrays:
#   X (images): shape (N, height, width, channels) — pixel values 0-255
#   y (labels): shape (N,)                         — integer class 0-3
# ─────────────────────────────────────────────────────────────

def load_dataset(data_dir):
    """
    Reads all images from the class subfolders inside data_dir.

    Expected folder layout (see README.md):
        data_dir/
            no_jaywalk/    <- images of normal pedestrian behaviour
            low_risk/      <- images of low-risk jaywalking
            medium_risk/   <- images of medium-risk jaywalking
            high_risk/     <- images of high-risk jaywalking

    Returns:
        X -- numpy array, shape (N, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), uint8
        y -- numpy array, shape (N,), int32, values 0-3
    """
    import cv2   # OpenCV for reading and resizing images.

    images = []
    labels = []
    skipped = 0

    print(f"\nLoading images from: {data_dir}")

    for class_index, class_name in enumerate(CLASS_NAMES):
        class_folder = os.path.join(data_dir, class_name)

        if not os.path.isdir(class_folder):
            print(f"  WARNING: folder not found — {class_folder}")
            continue

        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        all_files = [
            f for f in os.listdir(class_folder)
            if os.path.splitext(f)[1].lower() in valid_extensions
        ]

        if len(all_files) == 0:
            print(f"  WARNING: no images found in {class_folder}")
            continue

        print(f"  {class_name}: loading {len(all_files)} images ...", end="", flush=True)

        for filename in all_files:
            filepath = os.path.join(class_folder, filename)

            # Read image — OpenCV returns BGR colour order by default.
            img = cv2.imread(filepath)
            if img is None:
                skipped += 1
                continue

            # Convert BGR -> RGB (OpenCV loads BGR, everything else uses RGB).
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize to consistent dimensions.
            # INTER_AREA is best when shrinking (less blurring than other methods).
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT),
                             interpolation=cv2.INTER_AREA)

            images.append(img)
            labels.append(class_index)

        print(f" done.")

    if len(images) == 0:
        print("\nERROR: No images loaded. Check DATA_DIR and folder structure.")
        sys.exit(1)

    X = np.array(images, dtype=np.uint8)   # pixel values 0-255
    y = np.array(labels, dtype=np.int32)   # class indices 0-3

    print(f"\nTotal loaded: {len(X)} images (skipped {skipped} unreadable files)")
    print(f"Image array shape: {X.shape}")   # e.g. (1200, 224, 224, 3)

    # Show class distribution.
    print("\nClass distribution:")
    for i, name in enumerate(CLASS_NAMES):
        count = int(np.sum(y == i))
        pct   = 100 * count / len(y)
        bar   = "█" * (count // max(1, len(y) // 40))
        print(f"  {name:<14} {count:>5} images ({pct:5.1f}%)  {bar}")

    return X, y


# ─────────────────────────────────────────────────────────────
# STEP 2: SPLIT DATA
#
# Divides dataset into:
#   Train (80%)      — model learns from these
#   Validation (10%) — checked during training (not used for learning)
#   Test (10%)       — final honest evaluation after all training
# ─────────────────────────────────────────────────────────────

def split_dataset(X, y):
    """Splits X and y into train / validation / test sets (80/10/10)."""

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y   # keeps class proportions equal in each split
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=y_temp
    )

    print(f"\nDataset split:")
    print(f"  Train:      {len(X_train)} images (80%)")
    print(f"  Validation: {len(X_val)} images (10%)")
    print(f"  Test:       {len(X_test)} images (10%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────────────────────
# STEP 3: COMPUTE CLASS WEIGHTS
#
# Addresses class imbalance: if high_risk frames are rare, the model
# might just predict no_jaywalk for everything and still get decent
# accuracy. Class weights fix this by penalising errors on rare
# classes more heavily during training.
# ─────────────────────────────────────────────────────────────

def get_class_weights(y_train):
    """Computes per-class weights inversely proportional to class frequency."""
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weight_dict = dict(zip(classes.tolist(), weights.tolist()))

    print("\nClass weights (higher = rarer class, penalised more for errors):")
    for idx, w in weight_dict.items():
        print(f"  {CLASS_NAMES[idx]:<14} weight = {w:.3f}")

    return weight_dict


# ─────────────────────────────────────────────────────────────
# STEP 4: BUILD THE CNN MODEL
# ─────────────────────────────────────────────────────────────

def build_model():
    """Returns the chosen model (transfer learning or scratch CNN)."""
    input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    if USE_TRANSFER_LEARNING:
        return _build_transfer_model(input_shape)
    else:
        return _build_scratch_cnn(input_shape)


def _build_scratch_cnn(input_shape):
    """
    CNN built entirely from scratch. No pre-trained weights.

    Architecture (top to bottom):
    ┌─────────────────────────────────────────────────────────────┐
    │ Input (e.g. 224×224×3)                                      │
    ├─────────────────────────────────────────────────────────────┤
    │ Data Augmentation (flip, brightness, zoom)                  │
    │   → Only active during training. Increases data variety     │
    │     without saving extra files.                             │
    ├─────────────────────────────────────────────────────────────┤
    │ Rescaling: pixels 0-255 → 0-1                               │
    ├─────────────────────────────────────────────────────────────┤
    │ Conv Block 1: 32 filters, 3×3, ReLU ×2 → MaxPool → BN      │
    │ Conv Block 2: 64 filters, 3×3, ReLU ×2 → MaxPool → BN      │
    │ Conv Block 3: 128 filters, 3×3, ReLU ×2 → MaxPool → BN     │
    ├─────────────────────────────────────────────────────────────┤
    │ GlobalAveragePooling (summarises each feature map to 1 num) │
    ├─────────────────────────────────────────────────────────────┤
    │ Dense 256, ReLU → Dropout 0.5 (prevents overfitting)        │
    │ Dense 4, Softmax → [P(no_jaywalk), P(low), P(med), P(high)] │
    └─────────────────────────────────────────────────────────────┘
    """
    print("\nBuilding scratch CNN ...")

    inputs = keras.Input(shape=input_shape, name="input_image")

    # Augmentation — only active during model.fit(), not during predict().
    x = layers.RandomFlip("horizontal", name="aug_flip")(inputs)
    x = layers.RandomBrightness(factor=0.2, name="aug_brightness")(x)
    x = layers.RandomZoom(height_factor=0.1, name="aug_zoom")(x)

    # Rescaling — pixels 0-255 → 0-1. Neural networks train faster with small values.
    x = layers.Rescaling(1.0 / 255.0, name="rescaling")(x)

    # Convolutional Block 1
    # Conv2D: a 3×3 filter slides across the image detecting local patterns.
    # 32 filters = 32 different detectors in parallel, each learning something different.
    # ReLU: max(0, x) — zeroes out negatives, keeps positives. Standard hidden activation.
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv1a")(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv1b")(x)
    # MaxPooling: keeps the max value in each 2×2 block. Halves spatial size.
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    # BatchNormalization: stabilises values between layers so training doesn't go unstable.
    x = layers.BatchNormalization(name="bn1")(x)

    # Convolutional Block 2 — more filters (64) because deeper = more complex patterns.
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2a")(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2b")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = layers.BatchNormalization(name="bn2")(x)

    # Convolutional Block 3
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv3a")(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv3b")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)
    x = layers.BatchNormalization(name="bn3")(x)

    # GlobalAveragePooling: averages each feature map to a single number.
    # Produces 128 values total — a compact summary of what was detected.
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # Dense layer: fully connected, same as MLP from Week 8.
    x = layers.Dense(256, activation="relu", name="dense1")(x)

    # Dropout: randomly sets 50% of neurons to 0 per training step.
    # Forces model not to rely on any single neuron → reduces overfitting.
    # Automatically disabled during evaluation/prediction.
    x = layers.Dropout(0.5, name="dropout")(x)

    # Output: 4 neurons (one per class), softmax activation.
    # Softmax ensures all four probabilities sum to exactly 1.
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="JaywalkingCNN_Scratch")
    return model


def _build_transfer_model(input_shape):
    """
    Transfer learning using MobileNetV2 pre-trained on ImageNet.

    MobileNetV2 was trained on 1.2 million images across 1000 categories.
    It already knows how to detect edges, textures, shapes, and object parts.
    We transfer that knowledge to our task by:
        1. Keeping all those learned weights (frozen in Phase 1)
        2. Replacing the final layer with our own 4-class output head
        3. Optionally unfreezing the top layers later (Phase 2 fine-tuning)

    Not cheating — standard industry practice. If rubric says scratch only,
    set USE_TRANSFER_LEARNING = False above.
    """
    print("\nBuilding transfer learning model (MobileNetV2 base) ...")

    inputs = keras.Input(shape=input_shape, name="input_image")

    # Augmentation (same as scratch CNN).
    x = layers.RandomFlip("horizontal", name="aug_flip")(inputs)
    x = layers.RandomBrightness(factor=0.2, name="aug_brightness")(x)
    x = layers.RandomZoom(height_factor=0.1, name="aug_zoom")(x)

    # MobileNetV2 expects pixels in [-1, 1] (different from our 0-1 rescaling).
    x = keras.applications.mobilenet_v2.preprocess_input(x)

    # Load base model without top (classification) layer.
    # weights="imagenet" downloads pre-trained weights once (~14MB).
    base = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False   # Lock base weights for Phase 1.

    # Pass images through the frozen base to extract features.
    x = base(x, training=False)

    # Our new classification head.
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.5, name="dropout1")(x)
    x = layers.Dense(128, activation="relu", name="dense2")(x)
    x = layers.Dropout(0.3, name="dropout2")(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs,
                        name="JaywalkingCNN_MobileNetV2")
    model._base_model = base   # Store reference so fine_tune_model can access it.
    return model


def fine_tune_model(model, X_train, y_train, X_val, y_val, class_weights):
    """
    Phase 2: unfreeze top layers of MobileNetV2 and continue training at
    a very low learning rate. Only call this after Phase 1 is complete.

    Why low LR? Pre-trained weights are already good. We only want to
    nudge them slightly towards our specific task, not overwrite them.
    """
    if not USE_TRANSFER_LEARNING or not hasattr(model, "_base_model"):
        return None

    base = model._base_model
    base.trainable = True

    # Keep early layers frozen (they detect basic edges — universally useful).
    # Only unfreeze the top 30 layers (more task-specific features).
    for layer in base.layers[:-30]:
        layer.trainable = False

    n_trainable = sum(1 for l in model.layers if l.trainable)
    print(f"\nFine-tuning: {n_trainable} trainable layers")

    # Must recompile after changing trainability.
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE / 100),  # 100x smaller LR
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=_get_callbacks(prefix="finetune"),
        verbose=1
    )
    return history


# ─────────────────────────────────────────────────────────────
# STEP 5: TRAIN
# ─────────────────────────────────────────────────────────────

def _get_callbacks(prefix="train"):
    """
    Callbacks = functions Keras calls automatically at the end of each epoch.

    EarlyStopping: stops training if val_loss hasn't improved in 7 epochs.
                   Restores the best weights automatically.
    ReduceLROnPlateau: halves learning rate if val_loss stalls for 3 epochs.
    ModelCheckpoint: saves the best model to disk as training progresses.
    """
    checkpoint_path = os.path.join(OUTPUT_DIR, f"best_model_{prefix}.keras")

    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=0
        ),
    ]


def train_model(model, X_train, y_train, X_val, y_val, class_weights):
    """
    Compiles and trains the model (Phase 1).

    Adam optimizer: smart gradient descent that adjusts step size
    automatically per parameter. Standard for image classification.

    sparse_categorical_crossentropy: cross-entropy loss for integer labels
    (0, 1, 2, 3). "Sparse" = don't need to one-hot encode labels first.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    print(f"\nTraining for up to {EPOCHS} epochs ...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=_get_callbacks(prefix="phase1"),
        verbose=1
    )
    return history


# ─────────────────────────────────────────────────────────────
# STEP 6: EVALUATE
# ─────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on test images (never seen during training).

    Precision: of images predicted as class X, what fraction actually were X?
    Recall:    of images that are actually class X, what fraction did we find?
    F1-score:  harmonic mean of precision and recall.
    Accuracy:  fraction of all predictions that were correct.

    Targets from proposal: F1 >= 0.80, Recall >= 0.85
    """
    print("\n" + "=" * 55)
    print("EVALUATION ON TEST SET")
    print("=" * 55)

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred       = np.argmax(y_pred_probs, axis=1)

    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=CLASS_NAMES,
        zero_division=0   # silences the UndefinedMetricWarning from earlier
    ))

    print("Confusion Matrix (row=actual, col=predicted):")
    cm = confusion_matrix(y_test, y_pred)
    header = "        " + "  ".join(f"{n[:8]:>8}" for n in CLASS_NAMES)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>8}" for v in row)
        print(f"  {CLASS_NAMES[i][:8]:>8}  {row_str}")

    return y_pred, y_pred_probs


# ─────────────────────────────────────────────────────────────
# STEP 7: RISK SCORE
# ─────────────────────────────────────────────────────────────

def compute_risk_score(prob_vector):
    """
    Converts 4-class probabilities → single risk score in [0, 1].

    Example:
        probs = [0.05, 0.15, 0.55, 0.25]
        risk  = 0.00×0.05 + 0.33×0.15 + 0.67×0.55 + 1.00×0.25 = 0.669
    """
    weights = np.array([0.0, 0.33, 0.67, 1.0])
    return float(np.dot(prob_vector, weights))


def show_risk_score_demo(model, X_test, y_test, n=5):
    """Shows risk scores for the first n test images."""
    print("\n" + "=" * 55)
    print("RISK SCORE DEMO")
    print("=" * 55)

    probs = model.predict(X_test[:n], verbose=0)

    for i, p in enumerate(probs):
        score      = compute_risk_score(p)
        pred_class = CLASS_NAMES[np.argmax(p)]
        true_class = CLASS_NAMES[y_test[i]]
        correct    = "CORRECT" if pred_class == true_class else "WRONG"

        print(f"  Image {i+1}: actual={true_class:<14} "
              f"predicted={pred_class:<14} risk={score:.3f}  [{correct}]")


# ─────────────────────────────────────────────────────────────
# STEP 8: PLOT TRAINING HISTORY
# ─────────────────────────────────────────────────────────────

def plot_training_history(history, filename="training_curves.png"):
    """
    Saves a loss + accuracy plot next to this script.

    What to look for:
        Good:         train and val loss both decrease and converge.
        Overfitting:  train keeps falling, val starts rising.
        Underfitting: both stay high — model too simple or needs more epochs.
    """
    save_path = os.path.join(OUTPUT_DIR, filename)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Training History", fontsize=14)

    ax1.plot(history.history["loss"],     label="Training loss",   color="royalblue")
    ax1.plot(history.history["val_loss"], label="Validation loss", color="tomato", linestyle="--")
    ax1.set_title("Loss per Epoch\n(overfitting = val rises while train falls)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (cross-entropy)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history["accuracy"],     label="Training accuracy",   color="royalblue")
    ax2.plot(history.history["val_accuracy"], label="Validation accuracy", color="tomato", linestyle="--")
    ax2.axhline(y=0.80, color="green", linestyle=":", alpha=0.7, label="F1 target (0.80)")
    ax2.set_title("Accuracy per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"\nTraining curves saved → {save_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 55)
    print("ENGG2112 — Jaywalking Detector CNN")
    print("=" * 55)

    X, y = load_dataset(DATA_DIR)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    class_weights = get_class_weights(y_train)

    model   = build_model()
    history = train_model(model, X_train, y_train, X_val, y_val, class_weights)

    if USE_TRANSFER_LEARNING:
        history_ft = fine_tune_model(
            model, X_train, y_train, X_val, y_val, class_weights
        )
        if history_ft:
            plot_training_history(history_ft, filename="finetune_curves.png")

    evaluate_model(model, X_test, y_test)
    show_risk_score_demo(model, X_test, y_test)
    plot_training_history(history, filename="training_curves.png")

    model_path = os.path.join(OUTPUT_DIR, "jaywalking_cnn_final.keras")
    model.save(model_path)
    print(f"\nModel saved → {model_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
