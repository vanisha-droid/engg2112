"""
jaywalking_cnn.py
=================
ENGG2112 - Jaywalking Detector | Team: Victoria, Shiyao, Vanisha, Rejaksi

WHAT THIS FILE DOES
-------------------
This script trains a Convolutional Neural Network (CNN) to classify images
into two categories:
    - "jaywalk"    → an image showing jaywalking behaviour
    - "no_jaywalk" → an image showing safe/normal pedestrian behaviour

The images it trains on come from the Street Scene dataset.
These images already have bounding boxes drawn around the jaywalkers, so the
CNN sees the box markings as part of the visual input and learns to associate
them with the jaywalking label.

HOW TO RUN
----------
1. Set STREET_SCENE_DIR below to where your Street Scene dataset lives.
2. Make sure the folder structure matches what's described in the config section.
3. Run:  python jaywalking_cnn.py

EXPECTED FOLDER STRUCTURE (Street Scene dataset)
-------------------------------------------------
street_scene/
    train/
        jaywalk/        ← images of jaywalking scenes (with bounding boxes)
        no_jaywalk/     ← images of normal crossing scenes
    val/
        jaywalk/
        no_jaywalk/
    test/
        jaywalk/
        no_jaywalk/

DEPENDENCIES
------------
pip install tensorflow scikit-learn matplotlib
"""

# ══════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════
# These are libraries (pre-written code) that we borrow from so we
# don't have to implement everything ourselves.

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Saves figures to files instead of opening a popup window.
                        # This works even on servers or machines without a display.
import matplotlib.pyplot as plt

# TensorFlow is Google's deep learning library.
# Keras is the user-friendly "layer" on top of TensorFlow that makes
# building neural networks much simpler.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# scikit-learn gives us evaluation tools (F1, precision, recall, etc.)
from sklearn.metrics import classification_report, confusion_matrix


# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════
# All settings live here so you only need to change one section,
# not hunt through hundreds of lines of code.

# --- CHANGE THIS: path to your Street Scene dataset folder ---
STREET_SCENE_DIR = r"C:\path\to\street_scene"   # e.g. r"C:\Users\Shiyao\street_scene"

# Class names — must exactly match the subfolder names in train/val/test.
CLASS_NAMES = ["jaywalk", "no_jaywalk"]
NUM_CLASSES = len(CLASS_NAMES)   # 2

# Image dimensions — all images get resized to this before entering the CNN.
# Why? The CNN requires every image to be the same size.
# 224×224 is the standard size for transfer learning (MobileNetV2).
# Use 128×128 if you want faster training and your computer is slow.
IMG_HEIGHT   = 224
IMG_WIDTH    = 224
IMG_CHANNELS = 3    # 3 channels = RGB (red, green, blue). Always 3 for colour images.

# Training settings
BATCH_SIZE    = 16    # How many images to feed in per weight update.
                      # Think of it as: study 16 flashcards, then update your notes.
EPOCHS        = 30    # Maximum number of full passes through the training set.
                      # Early stopping (below) will halt training before this if needed.
LEARNING_RATE = 0.001 # How big each weight-update step is.
                      # Too big → overshoots the answer. Too small → takes forever.
RANDOM_SEED   = 42    # Makes results reproducible. Same seed = same results each run.

# Transfer learning switch
# True  = use MobileNetV2 (pre-trained on 1.2M images — recommended for small datasets)
# False = train CNN from scratch (simpler but needs more data to work well)
USE_TRANSFER_LEARNING = True

# Where to save model files, plots, etc. (same folder as this script by default)
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ══════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════
#
# Plain English:
#   We tell Keras to look inside the train/, val/, and test/ folders,
#   read every image, resize it to 224×224, and remember which subfolder
#   (class) each image came from.
#
#   Keras's ImageDataGenerator handles all of this automatically — it
#   reads images in batches so we don't have to load everything into
#   RAM at once. This is especially important when you have thousands
#   of large images.
#
#   Data augmentation (only on training data):
#   Instead of saving extra files, we "virtually" create more training
#   examples by randomly flipping, zooming, and brightening existing
#   images on-the-fly. The model never sees the exact same image twice
#   this way, which reduces overfitting.

def load_data(data_dir):
    """
    Creates Keras data generators for train, val, and test sets.

    Returns:
        train_gen  -- generator that yields batches of training images
        val_gen    -- generator that yields batches of validation images
        test_gen   -- generator that yields batches of test images
                      (shuffle=False so we can match predictions to filenames later)
    """

    # Training generator: includes augmentation (artificial data variety)
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,       # Normalise pixel values: 0-255 → 0-1
                                  # Neural networks train much faster when
                                  # inputs are small numbers near 0.

        horizontal_flip=True,     # Randomly mirror image left-right.
                                  # A person jaywalking from left is the same
                                  # as one jaywalking from the right.

        brightness_range=[0.8, 1.2],  # Randomly make image 20% darker or brighter.
                                      # Makes the model robust to different lighting.

        zoom_range=0.1,           # Randomly zoom in/out up to 10%.

        rotation_range=5,         # Tiny random rotation (±5°). Street cameras
                                  # can be slightly tilted.

        fill_mode="nearest"       # When rotating, fill empty corners by
                                  # repeating nearby pixel values.
    )

    # Validation and test generators: NO augmentation — we need a fair evaluation.
    # Only rescale (normalise) so the numbers match what the model was trained on.
    eval_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255
    )

    print(f"\n{'='*55}")
    print("LOADING DATA")
    print(f"{'='*55}")

    # flow_from_directory: reads images from disk folder-by-folder.
    # Each subfolder name becomes a class label automatically.
    train_gen = train_datagen.flow_from_directory(
        directory=os.path.join(data_dir, "train"),  # path to train/ folder
        target_size=(IMG_HEIGHT, IMG_WIDTH),          # resize all images to this
        color_mode="rgb",                             # load as 3-channel colour
        class_mode="binary",                          # 2 classes → single 0/1 output
        batch_size=BATCH_SIZE,
        shuffle=True,                                 # shuffle training data each epoch
        seed=RANDOM_SEED
    )

    val_gen = eval_datagen.flow_from_directory(
        directory=os.path.join(data_dir, "val"),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode="rgb",
        class_mode="binary",
        batch_size=BATCH_SIZE,
        shuffle=False    # don't shuffle validation — we want consistent comparison
    )

    test_gen = eval_datagen.flow_from_directory(
        directory=os.path.join(data_dir, "test"),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode="rgb",
        class_mode="binary",
        batch_size=1,    # one at a time for evaluation so we get per-image results
        shuffle=False
    )

    # Print a summary of what was found
    print(f"\nClass mapping: {train_gen.class_indices}")
    print(f"  Train:      {train_gen.n} images across {NUM_CLASSES} classes")
    print(f"  Validation: {val_gen.n} images")
    print(f"  Test:       {test_gen.n} images")

    # Count class balance — important to detect if one class dominates
    unique, counts = np.unique(train_gen.classes, return_counts=True)
    print("\nTraining class distribution:")
    for cls_idx, cnt in zip(unique, counts):
        name = [k for k, v in train_gen.class_indices.items() if v == cls_idx][0]
        pct  = 100 * cnt / train_gen.n
        bar  = "█" * (cnt // max(1, train_gen.n // 40))
        print(f"  {name:<14} {cnt:>5} ({pct:.1f}%)  {bar}")

    return train_gen, val_gen, test_gen


# ══════════════════════════════════════════════════════════════
# STEP 2 — BUILD THE CNN MODEL
# ══════════════════════════════════════════════════════════════
#
# Plain English — What is a CNN?
#   A CNN is a stack of layers, each transforming the image into a
#   more and more abstract representation until the final layer just
#   says "jaywalk" or "not jaywalk".
#
#   Layer-by-layer breakdown:
#
#   ┌─────────────────────────────────────────────────────────┐
#   │ INPUT IMAGE (224 × 224 × 3)                             │
#   │   - 224 pixels wide, 224 pixels tall, 3 colour channels │
#   ├─────────────────────────────────────────────────────────┤
#   │ CONV + RELU BLOCK (repeated 3 times)                    │
#   │   Conv2D: A small 3×3 filter slides across the image.   │
#   │   At each position it multiplies the 9 pixels under it  │
#   │   by 9 learned weights and sums them up → one number.   │
#   │   Doing this for every position produces a "feature map"│
#   │   — a new image where bright spots = "pattern found".   │
#   │   ReLU: max(0, x). Zeroes out negatives, keeps positives│
#   │   MaxPooling: Shrinks the feature map by 2×. Takes the  │
#   │   maximum value in each 2×2 block. Keeps the strongest  │
#   │   signals. Makes the model size-robust.                 │
#   ├─────────────────────────────────────────────────────────┤
#   │ GLOBALAVERAGEPOOLING                                    │
#   │   Collapses each feature map to a single average number.│
#   │   Turns a 7×7×512 tensor into a vector of 512 numbers.  │
#   ├─────────────────────────────────────────────────────────┤
#   │ DENSE LAYERS (like an MLP from Week 8)                  │
#   │   Takes that 512-number summary and learns to combine   │
#   │   them into a classification decision.                  │
#   │   Dropout randomly turns off neurons during training    │
#   │   to prevent over-memorising training examples.         │
#   ├─────────────────────────────────────────────────────────┤
#   │ OUTPUT: 1 neuron, sigmoid activation                    │
#   │   Outputs a number between 0 and 1:                     │
#   │   Close to 0 → no_jaywalk                               │
#   │   Close to 1 → jaywalk                                  │
#   └─────────────────────────────────────────────────────────┘

def build_scratch_cnn(input_shape):
    """
    Builds a CNN from scratch (no pre-trained weights).
    Good as a simple baseline or if the rubric requires it.
    """
    print("\nBuilding scratch CNN model ...")

    # keras.Input defines the shape of data coming in.
    # input_shape = (224, 224, 3) — height, width, channels.
    inputs = keras.Input(shape=input_shape, name="input_image")

    # ── Conv Block 1 ──────────────────────────────────────────
    # Conv2D(32, (3,3)):
    #   32 = number of filters (detectors). Each learns a different pattern.
    #        First few might learn "horizontal edge", "diagonal line", etc.
    #   (3,3) = filter size. A tiny 3×3 window slides across the image.
    #   activation="relu" = apply ReLU immediately after convolution.
    #   padding="same" = add zeros around the border so the output stays
    #                    the same spatial size as the input (no shrinking yet).
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv1a")(inputs)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv1b")(x)
    # MaxPooling2D((2,2)):
    #   Shrinks from 224×224 → 112×112 by taking the max in each 2×2 block.
    #   This reduces computation and makes the model less sensitive to exact
    #   pixel positions (a box shifted 1 pixel still gets detected).
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    # BatchNormalization: rescales values between layers so training stays stable.
    x = layers.BatchNormalization(name="bn1")(x)

    # ── Conv Block 2 ──────────────────────────────────────────
    # Now 64 filters — more patterns because we've already narrowed the image.
    # Each filter now "sees" a larger effective region because of the pooling above.
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2a")(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2b")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    # 112×112 → 56×56
    x = layers.BatchNormalization(name="bn2")(x)

    # ── Conv Block 3 ──────────────────────────────────────────
    # 128 filters — even more complex patterns at this level.
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv3a")(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv3b")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)
    # 56×56 → 28×28
    x = layers.BatchNormalization(name="bn3")(x)

    # ── GlobalAveragePooling ───────────────────────────────────
    # After Conv Block 3 we have 128 feature maps of size 28×28.
    # GlobalAveragePooling takes the average of all 28×28 = 784 values in each
    # feature map, collapsing it to a single number.
    # Result: a flat vector of 128 numbers summarising what patterns were found.
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # ── Dense (MLP) Head ──────────────────────────────────────
    # This is just like the MLP from Week 8 — fully connected neurons that
    # learn to combine the 128 extracted features into a classification.
    x = layers.Dense(256, activation="relu", name="dense1")(x)

    # Dropout(0.5): during training, randomly set 50% of neurons to 0.
    # This stops the model from becoming too reliant on any single neuron
    # and forces it to learn more robust features → less overfitting.
    # During testing/prediction, dropout is automatically disabled.
    x = layers.Dropout(0.5, name="dropout")(x)

    # ── Output Layer ──────────────────────────────────────────
    # 1 neuron with sigmoid activation.
    # Sigmoid squashes the output to [0, 1]:
    #   Close to 0 → model thinks "no_jaywalk"
    #   Close to 1 → model thinks "jaywalk"
    # This is exactly logistic regression from Week 6, just at the end of a CNN!
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="JaywalkCNN_Scratch")
    return model


def build_transfer_model(input_shape):
    """
    Builds a model using MobileNetV2 as a pre-trained feature extractor.

    Why use transfer learning?
        MobileNetV2 was already trained on 1.2 million images across 1000 categories.
        It already knows how to detect edges, textures, shapes, faces, car parts, etc.
        We "transfer" that knowledge to our task by:
            1. Keeping all those learned filters (frozen — don't touch them yet)
            2. Adding our own small classification layers on top
            3. Training ONLY our new layers first (Phase 1)
            4. Later unfreezing the top of the base and fine-tuning (Phase 2)

        This is like hiring someone who already knows how to read and write,
        and teaching them specifically about jaywalking law, rather than starting
        from scratch with someone who has never seen a letter.
    """
    print("\nBuilding transfer learning model (MobileNetV2 backbone) ...")

    inputs = keras.Input(shape=input_shape, name="input_image")

    # MobileNetV2 expects pixels rescaled to [-1, 1], not [0, 1].
    # preprocess_input handles that conversion.
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)

    # Load MobileNetV2 without its final classification layer (include_top=False).
    # weights="imagenet" downloads and uses the pre-trained ImageNet weights.
    # These weights encode knowledge about 1000 object categories.
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,          # cut off the original final layers
        weights="imagenet"          # use pre-trained knowledge
    )

    # Freeze the base model: its weights will NOT change during Phase 1 training.
    # We're only training our new head layers.
    base_model.trainable = False

    # Pass images through the frozen base to get rich feature maps.
    # training=False keeps BatchNorm in inference mode (important for frozen layers).
    x = base_model(x, training=False)

    # ── Our Custom Classification Head ────────────────────────
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.4, name="dropout1")(x)
    x = layers.Dense(64, activation="relu", name="dense2")(x)
    x = layers.Dropout(0.2, name="dropout2")(x)

    # Binary output: 1 neuron, sigmoid. Same logic as scratch CNN above.
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="JaywalkCNN_MobileNetV2")

    # Store reference to base model so fine_tune() can access it later
    model._base = base_model
    return model


# ══════════════════════════════════════════════════════════════
# STEP 3 — CALLBACKS (helper functions that run during training)
# ══════════════════════════════════════════════════════════════
#
# Callbacks are things Keras automatically does at the end of each epoch.
# Think of them as automatic helpers watching your training.

def get_callbacks(run_label="run"):
    """
    Returns a list of callbacks:

    EarlyStopping:
        Watches validation loss each epoch. If it hasn't improved in
        8 epochs straight, stop training and restore the best weights.
        Why? Training longer doesn't always help — after a certain point
        the model just memorises training noise (overfitting).

    ReduceLROnPlateau:
        If validation loss plateaus for 3 epochs, halve the learning rate.
        Smaller steps = more careful adjustments. Often unsticks training.

    ModelCheckpoint:
        Every time a new best validation loss is achieved, save the model.
        This way if training crashes we don't lose the best version.
    """
    best_path = os.path.join(OUTPUT_DIR, f"best_model_{run_label}.keras")

    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",        # watch validation loss
            patience=8,                # stop if no improvement for 8 epochs
            restore_best_weights=True, # revert to best weights when stopped
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,                # multiply LR by 0.5 when stuck
            patience=3,                # wait 3 stagnant epochs before reducing
            min_lr=1e-7,               # never go below this tiny LR
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=best_path,
            monitor="val_loss",
            save_best_only=True,       # only save when it improves
            verbose=0
        )
    ]


# ══════════════════════════════════════════════════════════════
# STEP 4 — TRAIN (Phase 1)
# ══════════════════════════════════════════════════════════════
#
# Plain English:
#   model.compile() sets up HOW the model will learn:
#     - optimizer: Adam — a smart version of gradient descent.
#       It automatically adjusts the learning rate for each weight
#       individually, so it converges faster and more reliably.
#     - loss: binary_crossentropy — the loss function for binary (2-class)
#       classification. This is exactly the cross-entropy from Week 8.
#       It penalises wrong predictions, especially when the model is
#       very confident but wrong.
#     - metrics: accuracy — the fraction of correct predictions.
#       We also want F1 later (done separately after training).
#
#   model.fit() runs training:
#     Each epoch:
#       1. Forward pass: feed a batch of images through the network.
#       2. Compute loss: how wrong was the output?
#       3. Backward pass: backpropagation — calculate gradient of loss
#          with respect to every weight.
#       4. Update weights: Adam steps each weight in the direction that
#          reduces loss.
#     Repeat until EarlyStopping says "enough".

def train_phase1(model, train_gen, val_gen):
    """Compiles and trains the model (Phase 1 — head only for transfer learning)."""

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",   # for binary (2-class) output with sigmoid
        metrics=["accuracy"]
    )

    model.summary()   # prints a table of all layers and parameter counts

    print(f"\nPhase 1 training (up to {EPOCHS} epochs) ...")
    history = model.fit(
        train_gen,                          # training data generator
        epochs=EPOCHS,
        validation_data=val_gen,            # validation data generator
        callbacks=get_callbacks("phase1"),  # early stopping, LR reduction, checkpoint
        verbose=1                           # print progress each epoch
    )
    return history


def fine_tune(model, train_gen, val_gen):
    """
    Phase 2 (transfer learning only): Unfreeze top layers and train further.

    After Phase 1, our new classification head is trained. Now we can
    carefully unfreeze the top layers of MobileNetV2 and nudge them
    towards our specific jaywalking task.

    Key: use a MUCH smaller learning rate (100× smaller). The pre-trained
    weights are already good — we only want tiny adjustments, not to
    overwrite what they learned from 1.2M images.
    """
    if not USE_TRANSFER_LEARNING or not hasattr(model, "_base"):
        print("\nSkipping fine-tuning (transfer learning not in use).")
        return None

    base = model._base
    base.trainable = True   # unfreeze the entire base

    # BUT keep the early layers frozen — they detect basic edges and textures
    # which are universal and don't need to change.
    # Only unfreeze the last 30 layers (higher-level, task-specific features).
    for layer in base.layers[:-30]:
        layer.trainable = False

    n_trainable = sum(1 for l in model.layers if l.trainable)
    print(f"\nPhase 2 fine-tuning: {n_trainable} trainable layers")

    # Recompile with a much smaller learning rate.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE / 100),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    history_ft = model.fit(
        train_gen,
        epochs=20,                              # fewer epochs — already close to good
        validation_data=val_gen,
        callbacks=get_callbacks("finetune"),
        verbose=1
    )
    return history_ft


# ══════════════════════════════════════════════════════════════
# STEP 5 — EVALUATE
# ══════════════════════════════════════════════════════════════
#
# We evaluate on the TEST set — images the model has NEVER seen during
# training or hyperparameter tuning. This gives an honest estimate of
# how the model will perform in the real world.
#
# Metrics we report (from proposal):
#   Accuracy  = (correct predictions) / (total predictions)
#   Precision = of images predicted "jaywalk", what fraction truly were?
#   Recall    = of all true jaywalking images, what fraction did we catch?
#               (Most important for a safety system — missing a real
#                jaywalker is worse than a false alarm.)
#   F1-score  = harmonic mean of precision and recall.
#               Target from proposal: F1 ≥ 0.80, Recall ≥ 0.85.

def evaluate(model, test_gen):
    """Runs the model on the test set and prints all evaluation metrics."""

    print(f"\n{'='*55}")
    print("EVALUATION ON TEST SET")
    print(f"{'='*55}")

    # Get raw sigmoid outputs (probabilities between 0 and 1)
    test_gen.reset()   # always reset before predicting on a generator
    y_prob = model.predict(test_gen, verbose=1).flatten()

    # Convert probabilities to hard class labels.
    # threshold 0.5: above 0.5 → "jaywalk" (1), below → "no_jaywalk" (0).
    # Lower the threshold (e.g. 0.4) to catch more jaywalkers at the cost
    # of more false alarms (better recall, lower precision).
    THRESHOLD = 0.5
    y_pred = (y_prob >= THRESHOLD).astype(int)
    y_true = test_gen.classes   # ground-truth labels from folder structure

    # Map class indices back to names for readable output.
    # test_gen.class_indices = {"jaywalk": 0, "no_jaywalk": 1} or similar.
    idx_to_name = {v: k for k, v in test_gen.class_indices.items()}
    target_names = [idx_to_name[i] for i in range(NUM_CLASSES)]

    print(f"\nThreshold used: {THRESHOLD}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

    # Confusion Matrix:
    #              Predicted No | Predicted Yes
    #   Actual No:     TN       |     FP
    #   Actual Yes:    FN       |     TP
    # We want FN to be low (don't miss real jaywalkers).
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix (rows=actual, cols=predicted):")
    header = " " * 16 + "  ".join(f"{n[:10]:>10}" for n in target_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>10}" for v in row)
        print(f"  {target_names[i][:14]:>14}  {row_str}")

    return y_true, y_pred, y_prob


# ══════════════════════════════════════════════════════════════
# STEP 6 — PLOT TRAINING HISTORY
# ══════════════════════════════════════════════════════════════
#
# What to look for in the plots:
#   GOOD:         Training and validation loss both decrease and converge.
#   OVERFITTING:  Training loss keeps falling, validation loss rises.
#                 Fix: more dropout, data augmentation, or less complex model.
#   UNDERFITTING: Both losses stay high. Model isn't learning.
#                 Fix: more epochs, larger model, lower learning rate.

def plot_history(history, filename="training_curves.png"):
    """Saves a 2-panel training history plot to disk."""
    save_path = os.path.join(OUTPUT_DIR, filename)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("ENGG2112 — Jaywalking CNN Training", fontsize=14)

    # Loss plot
    ax1.plot(history.history["loss"],     label="Training loss",   color="royalblue")
    ax1.plot(history.history["val_loss"], label="Validation loss", color="tomato",
             linestyle="--")
    ax1.set_title("Loss per Epoch\n(overfitting = val rises while train falls)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Binary Cross-Entropy Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(history.history["accuracy"],     label="Training accuracy",   color="royalblue")
    ax2.plot(history.history["val_accuracy"], label="Validation accuracy", color="tomato",
             linestyle="--")
    ax2.axhline(y=0.80, color="green", linestyle=":", alpha=0.7, label="Target (0.80)")
    ax2.set_title("Accuracy per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"\nPlot saved → {save_path}")


# ══════════════════════════════════════════════════════════════
# MAIN — runs everything in order
# ══════════════════════════════════════════════════════════════

def main():
    # Set random seeds so results are reproducible across runs
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 55)
    print("ENGG2112 — Jaywalking Detector CNN Training")
    print("=" * 55)
    print(f"Transfer learning: {USE_TRANSFER_LEARNING}")
    print(f"Input size: {IMG_HEIGHT}×{IMG_WIDTH}×{IMG_CHANNELS}")

    # ── 1. Load data ──────────────────────────────────────────
    train_gen, val_gen, test_gen = load_data(STREET_SCENE_DIR)

    # ── 2. Build model ────────────────────────────────────────
    input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    if USE_TRANSFER_LEARNING:
        model = build_transfer_model(input_shape)
    else:
        model = build_scratch_cnn(input_shape)

    # ── 3. Train Phase 1 ──────────────────────────────────────
    history = train_phase1(model, train_gen, val_gen)
    plot_history(history, "training_phase1.png")

    # ── 4. Fine-tune Phase 2 (transfer learning only) ─────────
    if USE_TRANSFER_LEARNING:
        history_ft = fine_tune(model, train_gen, val_gen)
        if history_ft:
            plot_history(history_ft, "training_finetune.png")

    # ── 5. Evaluate on test set ───────────────────────────────
    evaluate(model, test_gen)

    # ── 6. Save the trained model ─────────────────────────────
    model_path = os.path.join(OUTPUT_DIR, "jaywalking_cnn.keras")
    model.save(model_path)
    print(f"\nModel saved → {model_path}")
    print("\nDone! Now run yolo_preprocess.py to annotate Cityscapes images.")


if __name__ == "__main__":
    main()