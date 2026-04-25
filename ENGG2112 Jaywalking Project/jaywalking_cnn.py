"""
ENGG2112 - Jaywalking Detector CNN
===================================
Author: Shiyao Lin (Project Lead / ML Engineer)
Week 8 starting point - CNN for 4-class risk classification

Risk Classes:
    0 = No jaywalking
    1 = Low risk
    2 = Medium risk
    3 = High risk

This script uses SYNTHETIC (fake) data so you can run and understand
the architecture immediately, before connecting your real datasets.

To adapt for real data, see the section marked:
    # ====== SWAP THIS OUT FOR REAL DATA ======
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # suppress TF info/warnings

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")       # non-interactive backend (safe for all environments)
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────
# 0.  GLOBAL SETTINGS  (change these as needed)
# ──────────────────────────────────────────────
IMG_HEIGHT   = 128          # resize all images to this height
IMG_WIDTH    = 128          # resize all images to this width
NUM_CHANNELS = 3            # 3 = RGB colour, 1 = grayscale
NUM_CLASSES  = 4            # none / low / medium / high risk
BATCH_SIZE   = 32           # how many images per gradient-update step
EPOCHS       = 20           # how many full passes through training data
LEARNING_RATE = 0.001       # step size for the Adam optimiser
RANDOM_SEED  = 42           # for reproducibility

CLASS_NAMES = ["no_jaywalk", "low_risk", "medium_risk", "high_risk"]

# ──────────────────────────────────────────────
# 1.  DATA — synthetic placeholder
#     Replace this section with real image loading
# ──────────────────────────────────────────────

def make_synthetic_data(n_samples=1200):
    """
    Creates random pixel arrays and random labels.
    Every image is (IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS) with pixel
    values in [0, 255] — exactly what real photos look like before
    normalisation.

    Real-data replacement: load your UCSD / Cityscapes images here
    and return (images_array, labels_array) in the same format.
    """
    print("Generating synthetic data …")
    X = np.random.randint(0, 256,
                          size=(n_samples, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS),
                          dtype=np.uint8)

    # Simulate class imbalance: 60% no-jaywalk, 20% low, 12% med, 8% high
    # (mirrors what you'd expect in real street footage)
    probs  = [0.60, 0.20, 0.12, 0.08]
    labels = np.random.choice(NUM_CLASSES, size=n_samples, p=probs)
    return X, labels


def load_real_data_from_folder(root_dir):
    """
    ====== SWAP THIS OUT FOR REAL DATA ======

    Expected folder structure:
        root_dir/
            no_jaywalk/      ← class 0 images (jpg / png)
            low_risk/        ← class 1 images
            medium_risk/     ← class 2 images
            high_risk/       ← class 3 images

    Returns numpy arrays (X, y) identical in shape to make_synthetic_data().

    Usage (once your data is ready):
        X, y = load_real_data_from_folder("/path/to/dataset")
    """
    import pathlib, cv2

    images, labels = [], []
    root = pathlib.Path(root_dir)

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = root / class_name
        if not class_dir.exists():
            print(f"  WARNING: folder not found: {class_dir}")
            continue
        for img_path in class_dir.glob("*"):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            images.append(img)
            labels.append(class_idx)

    X = np.array(images, dtype=np.uint8)
    y = np.array(labels, dtype=np.int32)
    print(f"Loaded {len(X)} images from {root_dir}")
    return X, y


# ──────────────────────────────────────────────
# 2.  PRE-PROCESSING PIPELINE
# ──────────────────────────────────────────────

def build_preprocessing_layers():
    """
    These run INSIDE the model graph — no extra code needed at inference.

    Rescaling: converts uint8 [0,255] → float32 [0,1]
               (like StandardScaler but for images)

    Data augmentation (training only):
        - Random horizontal flip  (a car-camera would see both sides)
        - Random brightness shift (deals with dawn/dusk/night lighting)
        - Random zoom/crop        (deals with different camera distances)

    These augmentations directly address the limitations you listed
    in your proposal (variable lighting, camera position).
    """
    augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomBrightness(factor=0.2),        # ±20% brightness
        layers.RandomZoom(height_factor=0.1),        # up to 10% zoom
    ], name="augmentation")

    rescaling = layers.Rescaling(1.0 / 255.0, name="rescaling")

    return augmentation, rescaling


# ──────────────────────────────────────────────
# 3.  CNN ARCHITECTURE
# ──────────────────────────────────────────────

def build_cnn(input_shape, num_classes):
    """
    Builds a 3-block CNN for 4-class jaywalking risk classification.

    Architecture overview:
    ┌─────────────────────────────────────────────────────────┐
    │ Input  (128 × 128 × 3)                                  │
    ├─────────────────────────────────────────────────────────┤
    │ Rescaling  → pixels from [0,255] to [0,1]               │
    ├─────────────────────────────────────────────────────────┤
    │ Conv Block 1:  32 filters, 3×3, ReLU  → MaxPool(2×2)   │
    │ Conv Block 2:  64 filters, 3×3, ReLU  → MaxPool(2×2)   │
    │ Conv Block 3: 128 filters, 3×3, ReLU  → MaxPool(2×2)   │
    ├─────────────────────────────────────────────────────────┤
    │ GlobalAveragePooling (replaces Flatten — less overfit)   │
    ├─────────────────────────────────────────────────────────┤
    │ Dense(256, ReLU)  → Dropout(0.5)  ← prevents overfit   │
    │ Dense(4, Softmax) → class probabilities                  │
    └─────────────────────────────────────────────────────────┘

    Why these choices?
    • 3 conv blocks: enough depth for street-scene feature extraction
      without needing a GPU or huge dataset.
    • Filters double each block (32→64→128): early layers learn simple
      edges; deeper layers learn complex shapes (people, road markings).
    • MaxPooling halves spatial dimensions: reduces computation and
      makes the model translation-invariant (pedestrian at left or right
      of frame still detected).
    • Dropout(0.5): randomly zeroes half the neurons during training.
      Forces the network NOT to rely on any single neuron → reduces
      overfitting (a key risk with UCSD's small dataset).
    • GlobalAveragePooling: averages each feature map to a single number.
      Less parameters than Flatten → less overfitting.
    • Softmax output: converts raw scores to probabilities that sum to 1.
      Consistent with Week 8 slides on multi-class classification.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    # --- Normalisation (always applied, train and test) ---
    x = layers.Rescaling(1.0 / 255.0, name="rescaling")(inputs)

    # --- Convolutional Block 1 ---
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                      name="conv1_a")(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                      name="conv1_b")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.BatchNormalization(name="bn1")(x)     # stabilises training

    # --- Convolutional Block 2 ---
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same",
                      name="conv2_a")(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same",
                      name="conv2_b")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = layers.BatchNormalization(name="bn2")(x)

    # --- Convolutional Block 3 ---
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same",
                      name="conv3_a")(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same",
                      name="conv3_b")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)
    x = layers.BatchNormalization(name="bn3")(x)

    # --- Classification Head ---
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.5, name="dropout")(x)          # 50% dropout
    outputs = layers.Dense(num_classes, activation="softmax",
                           name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="JaywalkingCNN")
    return model


# ──────────────────────────────────────────────
# 4.  CLASS WEIGHTS (for class imbalance)
# ──────────────────────────────────────────────

def compute_class_weights(y_train):
    """
    Computes per-class weights so the loss function penalises mistakes
    on rare classes (high-risk jaywalking) more than common ones.

    This directly addresses the class imbalance limitation you noted
    in your proposal.

    Formula: weight_c = n_total / (n_classes × n_samples_in_class_c)
    """
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weight_dict = dict(zip(classes, weights))
    print("\nClass weights (higher = rarer class, penalised more):")
    for cls_idx, w in weight_dict.items():
        print(f"  Class {cls_idx} ({CLASS_NAMES[cls_idx]}): {w:.3f}")
    return weight_dict


# ──────────────────────────────────────────────
# 5.  TRAINING
# ──────────────────────────────────────────────

def train_model(model, X_train, y_train, X_val, y_val, class_weights):
    """
    Compiles and trains the model.

    Adam optimiser: the standard choice for CNNs. It automatically
    adjusts the learning rate per weight (adaptive), which is why it
    works better than plain SGD for most image tasks.

    Callbacks:
    • EarlyStopping: stops training if validation loss hasn't improved
      for 5 epochs. Prevents overfitting (Week 8 slide concept).
    • ReduceLROnPlateau: halves learning rate when training stalls.
      Implements the "just right" learning rate concept from Week 8.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",   # expects integer labels
        metrics=["accuracy"]
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
    ]

    print(f"\nTraining for up to {EPOCHS} epochs …")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    return history


# ──────────────────────────────────────────────
# 6.  EVALUATION
# ──────────────────────────────────────────────

def evaluate_model(model, X_test, y_test):
    """
    Reports the metrics specified in your proposal:
    precision, recall, F1-score, and accuracy.

    Target from proposal: F1 ≥ 0.80, Recall ≥ 0.85
    """
    print("\n" + "="*50)
    print("EVALUATION ON TEST SET")
    print("="*50)

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred       = np.argmax(y_pred_probs, axis=1)  # pick class with highest prob

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    print("Confusion Matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    return y_pred, y_pred_probs


# ──────────────────────────────────────────────
# 7.  RISK SCORE  (your proposal's risk output)
# ──────────────────────────────────────────────

def compute_risk_score(prob_vector):
    """
    Converts the CNN's 4-class probability output into a single
    continuous risk score in [0, 1].

    Formula: weighted sum of class probabilities
        risk = 0×P(no_jaywalk) + 0.33×P(low) + 0.67×P(medium) + 1.0×P(high)

    A score of 0.0 = definitely safe; 1.0 = definitely high-risk.
    This is the output your evaluator (Rejaksi) needs for benchmarking.

    Example:
        probs = [0.1, 0.2, 0.5, 0.2]
        risk  = 0×0.1 + 0.33×0.2 + 0.67×0.5 + 1.0×0.2 = 0.666
    """
    weights = np.array([0.0, 0.33, 0.67, 1.0])
    return float(np.dot(prob_vector, weights))


def demo_risk_scores(model, X_test, n=5):
    """Shows risk scores for the first n test images."""
    print("\n" + "="*50)
    print("RISK SCORE DEMO (first 5 test images)")
    print("="*50)
    probs = model.predict(X_test[:n], verbose=0)
    for i, p in enumerate(probs):
        score = compute_risk_score(p)
        pred_class = CLASS_NAMES[np.argmax(p)]
        print(f"  Image {i+1}: predicted='{pred_class}' | "
              f"risk_score={score:.3f} | probs={np.round(p, 3)}")


# ──────────────────────────────────────────────
# 8.  PLOT TRAINING CURVES
# ──────────────────────────────────────────────

def plot_history(history, save_path="training_curves.png"):
    """Saves a loss/accuracy plot — the 'loss curves per epoch'
    your proposal mentions for overfitting detection."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["loss"],     label="train loss")
    ax1.plot(history.history["val_loss"], label="val loss")
    ax1.set_title("Loss per Epoch\n(if val loss rises while train falls → overfitting)")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(True)

    ax2.plot(history.history["accuracy"],     label="train acc")
    ax2.plot(history.history["val_accuracy"], label="val acc")
    ax2.set_title("Accuracy per Epoch")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    print(f"\nTraining curves saved → {save_path}")


# ──────────────────────────────────────────────
# 9.  MAIN
# ──────────────────────────────────────────────

def main():
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # ── Load data ──────────────────────────────
    # Using synthetic data for now.
    # When real data is ready, replace with:
    #   X, y = load_real_data_from_folder("/path/to/your/dataset")
    X, y = make_synthetic_data(n_samples=1200)

    print(f"\nDataset shape: {X.shape}")   # (1200, 128, 128, 3)
    print(f"Labels shape:  {y.shape}")     # (1200,)
    unique, counts = np.unique(y, return_counts=True)
    print("Class distribution:")
    for u, c in zip(unique, counts):
        print(f"  {CLASS_NAMES[u]}: {c} images ({100*c/len(y):.1f}%)")

    # ── Split: 80% train, 10% val, 10% test (matches your proposal) ──
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp)

    print(f"\nSplit sizes — train:{len(X_train)}, "
          f"val:{len(X_val)}, test:{len(X_test)}")

    # ── Build model ────────────────────────────
    input_shape = (IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS)
    model = build_cnn(input_shape, NUM_CLASSES)
    model.summary()

    # ── Class weights ──────────────────────────
    class_weights = compute_class_weights(y_train)

    # ── Train ──────────────────────────────────
    history = train_model(model, X_train, y_train, X_val, y_val, class_weights)

    # ── Evaluate ───────────────────────────────
    evaluate_model(model, X_test, y_test)

    # ── Risk scores demo ───────────────────────
    demo_risk_scores(model, X_test)

    # ── Plot ───────────────────────────────────
    plot_history(history, save_path="/mnt/user-data/outputs/training_curves.png")

    # ── Save model ─────────────────────────────
    model.save("/mnt/user-data/outputs/jaywalking_cnn.keras")
    print("\nModel saved → jaywalking_cnn.keras")
    print("\nDone! Next steps:")
    print("  1. Replace make_synthetic_data() with load_real_data_from_folder()")
    print("  2. Tune EPOCHS, BATCH_SIZE, LEARNING_RATE")
    print("  3. Try transfer learning (see jaywalking_transfer_learning.py)")


if __name__ == "__main__":
    main()