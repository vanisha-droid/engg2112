# ENGG2112 — Jaywalking Detector CNN

**Team:** Victoria Kolmac · Shiyao Lin · Vanisha Goyal · Rejaksi G  
**Dataset:** Street Scene Dataset (Ramachandra et al., WACV 2020)  
**Task:** Classify video frames into four jaywalking risk levels using a CNN.

---

## Quick Start

```bash
# 1. Activate your virtual environment
source .venv/bin/activate          # Mac/Linux
.venv\Scripts\Activate.ps1         # Windows PowerShell

# 2. Install dependencies (once only)
pip install tensorflow scikit-learn matplotlib numpy opencv-python

# 3. Set your data path inside jaywalking_cnn.py
#    Change this line near the top of the file:
DATA_DIR = r"C:\path\to\your\dataset"

# 4. Run
python jaywalking_cnn.py
```

---

## Required Folder Structure

The script expects images to be pre-sorted into four folders named exactly:

```
your_dataset/
├── no_jaywalk/          ← normal behaviour, no crossing outside marked zones
│   ├── frame_00001.png
│   ├── frame_00002.png
│   └── ...
├── low_risk/            ← pedestrian near road edge, no immediate danger
│   └── ...
├── medium_risk/         ← pedestrian on road, no vehicle close by
│   └── ...
└── high_risk/           ← pedestrian on road with vehicle in frame
    └── ...
```

**Image format:** JPG or PNG, any resolution (script resizes automatically to 224×224).  
**Colour:** RGB preferred. Grayscale also works — set `IMG_CHANNELS = 1` in the config.

> **For the preprocessing team:** See the [Data Preparation](#data-preparation) section below.

---

## Configuration

All settings live at the top of `jaywalking_cnn.py`. The key ones to change:

| Setting | Default | What it does |
|---|---|---|
| `DATA_DIR` | `"C:\path\..."` | **Change this** — path to your sorted dataset |
| `USE_TRANSFER_LEARNING` | `True` | `True` = MobileNetV2, `False` = scratch CNN |
| `IMG_HEIGHT / IMG_WIDTH` | `224` | Resize target. Use `128` if PC is slow |
| `BATCH_SIZE` | `16` | Images per training step. Lower if out of memory |
| `EPOCHS` | `30` | Max training passes. EarlyStopping will stop sooner |

---

## What the Script Does (Step by Step)

### Step 1 — Load Data
Reads images from the four class folders. Resizes to `IMG_HEIGHT × IMG_WIDTH`. Returns a NumPy array of pixel values (0–255) and an array of integer labels (0–3).

### Step 2 — Split Data
Divides into **80% train / 10% validation / 10% test**. Stratified split — each subset has the same class proportions as the original dataset.

### Step 3 — Class Weights
Calculates per-class weights to handle imbalance. If `high_risk` frames are rare, their errors are penalised more during training. This means the model can't "cheat" by always predicting `no_jaywalk`.

### Step 4 — Build Model
Two options:

**Transfer Learning (recommended):**  
MobileNetV2 base (pre-trained on ImageNet) + new 4-class classification head.  
Phase 1: train only the new head. Phase 2 (fine-tuning): unfreeze and refine top layers.

**Scratch CNN:**  
Three convolutional blocks (32 → 64 → 128 filters) + GlobalAveragePooling + Dense layers.  
Simpler, no internet needed, but needs more data to reach good accuracy.

### Step 5 — Train
Uses Adam optimizer + sparse categorical cross-entropy loss. Callbacks:
- **EarlyStopping** — stops if validation loss stalls (prevents overfitting)
- **ReduceLROnPlateau** — halves learning rate when stuck
- **ModelCheckpoint** — saves best model automatically

### Step 6 — Evaluate
Runs on the test set (never seen during training). Reports:
- Precision, Recall, F1-score per class
- Overall accuracy
- Confusion matrix

**Project targets:** F1 ≥ 0.80, Recall ≥ 0.85

### Step 7 — Risk Score
Converts the four class probabilities into a single number:

```
risk = 0.00 × P(no_jaywalk) + 0.33 × P(low) + 0.67 × P(medium) + 1.00 × P(high)
```

Output range: 0.0 (definitely safe) to 1.0 (definitely high risk).

### Step 8 — Plots
Saves `training_curves.png` next to the script.  
Look for: both curves decreasing → good. Val loss rising while train falls → overfitting.

---

## Output Files

After running, you will find these in the same folder as the script:

| File | Description |
|---|---|
| `training_curves.png` | Loss and accuracy plots per epoch |
| `finetune_curves.png` | Same but for fine-tuning phase (if transfer learning) |
| `best_model_phase1.keras` | Best model saved during Phase 1 |
| `best_model_finetune.keras` | Best model saved during fine-tuning |
| `jaywalking_cnn_final.keras` | Final model after all training |

---

## Data Preparation

> **This section is for Victoria, Vanisha, and Rejaksi.**

### What we need from the Street Scene Dataset

**Download:** https://zenodo.org/records/10870472 (~46GB unzipped)

The dataset contains:
- `Train/` — 46 video sequences of **normal** activity (no jaywalking)
- `Test/` — 35 video sequences with annotated anomalous events (including jaywalking)
- Ground truth annotations: bounding boxes around anomalous events per frame

### How to label images

**Training frames (from `Train/` sequences):**  
All training frames show normal behaviour → label everything as `no_jaywalk`.  
Sample every 5th frame to avoid near-duplicate frames (15fps → every 5th = 3fps equivalent).

**Test frames (from `Test/` sequences):**  
Use the ground truth annotation files to check each frame:
- Frame has a jaywalking bounding box annotation → it's a risk frame
- Frame has no annotation → `no_jaywalk`

For risk level, use the **bounding box area** as a proxy for severity:
- No annotation → `no_jaywalk`
- Small bounding box (pedestrian far from vehicles) → `low_risk`
- Medium bounding box → `medium_risk`
- Large bounding box, or annotation type explicitly says "jaywalker" → `high_risk`

*(Check the dataset README.md for the exact annotation format and the list of 17 anomaly types.)*

### Target folder structure

Sort frames into these four folders (Shiyao's script reads from here):

```
street_scene_sorted/
├── no_jaywalk/
├── low_risk/
├── medium_risk/
└── high_risk/
```

### Handling class imbalance

Most frames will be `no_jaywalk`. **Do not** pre-generate augmented copies to balance classes — this can teach the model to recognise augmentation artifacts rather than actual jaywalking.

Instead:
- **Class weights** (already implemented in the CNN) — penalises rare class errors more
- **Cap no_jaywalk** — you don't need 100k normal frames. Keep ~3-5× the number of your rarest class.

### Expected image counts (rough target)

| Class | Minimum | Ideal |
|---|---|---|
| no_jaywalk | 500 | 2000 |
| low_risk | 200 | 500 |
| medium_risk | 200 | 500 |
| high_risk | 100 | 300 |

---

## What Each Person Needs to Do

### Shiyao (CNN — this file)
- [x] CNN architecture written
- [x] Training pipeline complete
- [x] Risk score output complete
- [ ] Set `DATA_DIR` once data is ready
- [ ] Run training and report metrics
- [ ] Tune hyperparameters if F1 < 0.80

### Victoria (Data labelling)
- [ ] Download Street Scene dataset
- [ ] Read the dataset README for annotation format
- [ ] Extract frames from Test sequences (every 5th frame)
- [ ] Use annotation files to sort frames into the four class folders
- [ ] Share the sorted folder with the team on Google Drive

### Vanisha (Data pipeline)
- [ ] Verify images load correctly (run `check_data.py` once written)
- [ ] Confirm class distribution looks reasonable
- [ ] Do NOT pre-generate augmented copies — augmentation is handled inside the CNN

### Rejaksi (Evaluation / QA)
- [ ] Once Victoria has sorted frames, spot-check 100 random frames
- [ ] Verify labels match what's visible in the image
- [ ] After training, review the confusion matrix and F1 scores
- [ ] Flag if any class has recall < 0.85

---

## Learning Resources

### Concepts covered in this code
- **CNN basics:** https://www.youtube.com/watch?v=YRhxdVk_sIs (deeplizard, 15 min)
- **Why convolutions work:** https://www.youtube.com/watch?v=KuXjwB4LzSA (3Blue1Brown)
- **Transfer learning explained:** https://www.youtube.com/watch?v=yofjFQddwHE (Sentdex, 10 min)
- **TensorFlow image classification tutorial:** https://www.tensorflow.org/tutorials/images/classification

### Dataset paper
Ramachandra, B. & Jones, M. (2020). *Street Scene: A new dataset and evaluation protocol for video anomaly detection.* WACV 2020.  
https://openaccess.thecvf.com/content_WACV_2020/papers/Ramachandra_Street_Scene_A_new_dataset_and_evaluation_protocol_for_video_WACV_2020_paper.pdf

### Reference paper (doing the same task)
*Jaywalking detection and localization in street scene videos using fine-tuned CNNs.*  
https://www.researchgate.net/publication/369233966

---

## Glossary

| Term | Plain English |
|---|---|
| **Epoch** | One full pass through all training images |
| **Batch** | A small group of images processed together before updating weights |
| **Loss** | A number measuring how wrong the model's predictions are (lower = better) |
| **Overfitting** | Model memorises training data but fails on new images |
| **Dropout** | Randomly ignore 50% of neurons per step to prevent overfitting |
| **ReLU** | Activation function: max(0, x). Zeroes out negatives |
| **Softmax** | Turns raw scores into probabilities that sum to 1 |
| **F1-score** | Balanced measure of precision and recall (target: ≥ 0.80) |
| **Transfer learning** | Reusing a model trained on a big dataset for a different (smaller) task |
| **MobileNetV2** | A lightweight CNN pre-trained on 1.2M images — our transfer learning base |
