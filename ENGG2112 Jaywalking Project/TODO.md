# ENGG2112 — Task Lists & Learning Plan

## Your Personal Todo List (Shiyao — CNN)

### Right now (no data needed)

- [ ] **Commit the new code to git**
      Files to commit: `jaywalking_cnn.py`, `README.md`
      Commit message suggestion: "Add CNN skeleton with full documentation"

- [ ] **Confirm the script runs without crashing**
      It will print an error like "ERROR: No images loaded" — that is expected
      and correct. It means the data loading works, it just has nothing to load yet.
      That is all you need to verify for now.

- [ ] **Watch these two videos (30 min total, high priority)**
      1. CNN explained visually (deeplizard, 15 min):
         https://www.youtube.com/watch?v=YRhxdVk_sIs
      2. Why convolutional filters work (3Blue1Brown, 15 min):
         https://www.youtube.com/watch?v=KuXjwB4LzSA
      These directly explain what Conv2D and MaxPooling do in the code.
      After watching, re-read the comments in _build_scratch_cnn() — it will click.

- [ ] **Read the reference paper abstract + methodology section (20 min)**
      https://www.researchgate.net/publication/369233966
      Note: what CNN architecture did they use? What dataset? What metrics?
      You will likely be asked about this in the presentation.

- [ ] **Read the Street Scene dataset README**
      Download the dataset (or ask Victoria to share it) and read the README.md
      inside. You need to understand the annotation format before you can write
      the frame labelling script (next week's task).

### Once you have some data (Week 9)

- [ ] **Update DATA_DIR in jaywalking_cnn.py** to point to your actual data folder
- [ ] **Run the script and check the "Class distribution" printout**
      If one class is 10x larger than another, flag it to Vanisha and Rejaksi
- [ ] **Check the training curves plot**
      Look at training_curves.png after training finishes.
      If val_loss rises while train_loss falls → overfitting → increase Dropout or
      reduce EPOCHS
- [ ] **Report your F1 scores** to the group

### Week 10 (refinement)

- [ ] **If F1 < 0.80:** try switching USE_TRANSFER_LEARNING = True (if not already)
- [ ] **If still < 0.80:** reduce BATCH_SIZE to 8, increase EPOCHS to 50
- [ ] **Write the risk score export** (save a CSV of [filename, risk_score] for Rejaksi)

---

## Todo List for Victoria (Data Labelling)

**Your job:** Sort Street Scene frames into the four class folders.

### Steps

1. [ ] Download Street Scene dataset from https://zenodo.org/records/10870472
       Warning: ~46GB. Check disk space first.

2. [ ] Read the dataset README.md (inside the zip) to understand:
       - How annotation files are structured
       - What the 17 anomaly types are
       - Which anomaly type corresponds to jaywalking

3. [ ] Extract frames from the TEST sequences (these have annotations)
       - Take every 5th frame (15fps / 5 = 3fps equivalent — avoids duplicates)
       - Saves ~29,000 frames from 35 test sequences

4. [ ] For each extracted test frame, check the annotation file:
       - If the frame has a "jaywalking" bounding box → it's a risk frame
       - Classify by bounding box area (small = low_risk, large = high_risk)
       - If no annotation → no_jaywalk

5. [ ] Extract frames from a FEW TRAINING sequences for no_jaywalk class
       - Only need ~2000-3000 no_jaywalk frames total (not all 56,000)
       - Take from Train001 through Train010 (every 10th frame)

6. [ ] Create this folder structure and sort frames into it:
       ```
       street_scene_sorted/
       ├── no_jaywalk/
       ├── low_risk/
       ├── medium_risk/
       └── high_risk/
       ```

7. [ ] Share the sorted folder on Google Drive with the whole team
       Tell Shiyao the full file path or Drive link so DATA_DIR can be set

8. [ ] Report counts per class to Rejaksi for QA

---

## Todo List for Vanisha (Data Pipeline)

**Your job:** Make sure the image data is clean and in the right format.

### Steps

1. [ ] **Stop the pre-generating of augmented copies**
       The tutor's concern was correct. Generating brightness/rotation copies
       before training can bias the model. Augmentation is already handled
       *inside* the CNN (RandomFlip, RandomBrightness, RandomZoom layers).
       These apply randomly during training and are turned off at test time.
       Nothing for you to do here — it's done.

2. [ ] **Once Victoria shares the sorted folder, run a data check:**
       Write a short script (or ask Shiyao) that:
       - Counts images per class
       - Checks all images can be opened
       - Checks image sizes are reasonable (not 1×1 or corrupted)

3. [ ] **Check for near-duplicate frames**
       Street Scene is extracted at 3fps but consecutive frames are very similar.
       If two consecutive frames both show a jaywalker, both are valid.
       If a class has > 5000 images, sample down to 3000 to save training time.

4. [ ] **Report to Shiyao:** the final image counts per class and the
       exact file path to the sorted folder.

---

## Todo List for Rejaksi (Evaluation / QA)

**Your job:** Verify data quality and evaluate model outputs.

### Steps

1. [ ] **Once Victoria shares frames, spot-check 100 random images**
       Open them and verify the label matches what you see.
       A frame labelled "high_risk" should visibly show a person on the road
       with a vehicle nearby. Flag errors to Victoria.

2. [ ] **After Shiyao trains the model, review:**
       - The confusion matrix (which classes are being confused with each other?)
       - Per-class recall (must be ≥ 0.85 for all classes per proposal)
       - F1-score per class (target ≥ 0.80)

3. [ ] **Compare to baseline**
       Your proposal mentions benchmarking against baseline models.
       A simple baseline: always predict the most common class (no_jaywalk).
       Note that baseline's accuracy and F1 — the CNN should beat it clearly.

4. [ ] **Document findings for the report**
       Which class is hardest to detect? Why might that be?
       (Hint: medium_risk is probably hardest — it's between two other classes)

---

## Learning Resources (in order)

### Watch first (total ~1 hour)
These are the most important ones. Do these before anything else.

1. **CNN visually explained** — deeplizard (15 min)
   https://www.youtube.com/watch?v=YRhxdVk_sIs
   *Covers: what a filter does, what MaxPooling does, why CNNs work on images*

2. **Visualising what CNNs learn** — 3Blue1Brown (15 min)
   https://www.youtube.com/watch?v=KuXjwB4LzSA
   *Covers: how filters detect edges → shapes → objects across layers*

3. **Transfer learning in 10 minutes** — Sentdex (10 min)
   https://www.youtube.com/watch?v=yofjFQddwHE
   *Covers: why pre-trained models work, when to use them*

4. **Overfitting and dropout explained** — StatQuest (15 min)
   https://www.youtube.com/watch?v=DEMmkFC6IGM
   *Covers: what overfitting looks like, how dropout prevents it*

### Read after watching (pick one)
- TensorFlow image classification tutorial:
  https://www.tensorflow.org/tutorials/images/classification
  *This is almost exactly your pipeline. Read steps 1-4.*

- TensorFlow transfer learning tutorial:
  https://www.tensorflow.org/tutorials/images/transfer_learning
  *Read this when transfer learning is confirmed allowed by rubric.*

### Your lecture notes (most relevant sections)
- **Week 8 slides** — read fully (MLP, activation functions, backpropagation,
  loss functions, epochs, batch training). This is the theoretical basis for
  everything in the CNN. You should be able to explain each concept.
- **Week 6 slides** — regularisation (L1/L2, dropout). Explains why Dropout
  is in the code and what it does mathematically.
- **Week 7 slides** — SVM. Less directly relevant to CNN but good background
  on classification boundaries and why model choice matters.

### For the report and presentation
Read the abstract + method sections of:
- The reference paper: https://www.researchgate.net/publication/369233966
- The Street Scene paper: https://openaccess.thecvf.com/content_WACV_2020/papers/Ramachandra_Street_Scene_A_new_dataset_and_evaluation_protocol_for_video_WACV_2020_paper.pdf

These will tell you exactly how others approached the same problem and give you
vocabulary and context for explaining your own methodology.
