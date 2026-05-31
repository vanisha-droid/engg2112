# ENGG2112 — Jaywalking Detector
**Group 1 | Victoria, Shiyao, Vanisha, Rejaksi**

This project trains and evaluates a Convolutional Neural Network (CNN) to classify street-scene images as either jaywalking or safe, and provides a live demonstration UI with person detection overlays.


## AI Assistance Acknowledgement

Claude (Anthropic) was used to assist with the development of code in this submission - primarily for implementing structure, debugging, and integrating components. All code was manually reviewed, tested, and evaluated by the team. Final design decisions, model choices, hyperparameter tuning, and results interpretation were carried out by the team members.


### `dataaug.py` — Data Augmentation
Reads the original Cityscapes dataset and writes an augmented copy with a balanced number of images per class (320 per class for training, 30 for validation). Augmentations — flips, brightness changes, zoom, rotation, and shifts. Test data is copied unchanged to preserve an unbiased evaluation set.


### `jaywalking_cnn.py` — CNN Training (Main)
Trains the binary image classifier. Supports two modes controlled by a single flag:
- **Scratch CNN** — a four-block convolutional network built from scratch with batch normalisation, dropout, and L2 regularisation.
- **Transfer learning** — a MobileNetV2 backbone pre-trained on ImageNet, with a custom classification head trained in two phases (head-only first, then selective fine-tuning of the top layers).


### `jaywalking_k.py` — K-Fold Cross-Validation
Runs K-Fold cross-validation on the original (non-augmented) dataset. The model is trained and evaluated K times, each time using a different fold as the validation set. Final metrics are averaged across all folds.

Note: we used k folds for more experimentation of model reliability, and talk about it in the appendix rather than the report's formal results.

### `yolo_preprocess.py` — YOLO Bounding Box Annotation
Note: we used this as an exploratory tool aligning with our future directions - incorporating individualised jaywalker detection so that the number of jaywalkers in an image could be detected. 

The script uses a pre-trained YOLOv8 model to detect all people in raw Cityscapes images and draw bounding boxes around them. The annotated images are saved to an output folder and used as training data for the CNN.

`jaywalker_UI.py` — Live Demonstration UI
An interactive desktop application (built with CustomTkinter) that loads the trained Keras model and steps through test-set images one at a time. For each image it displays:
- The CNN's prediction (JAYWALKING / SAFE) and confidence score
- The true label and whether the prediction was correct
- A running session accuracy bar
- When jaywalking is predicted: bounding boxes drawn by a HOG (Histogram of Oriented Gradients) person detector, with a count of detected jaywalkers

