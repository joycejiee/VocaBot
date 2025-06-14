# VöcaBot - Sign Language Classifier & Translator

VöcaBot is an intelligent system for detecting and classifying hand gestures into letters (A-Z) using a webcam, with optional UI for real-time interaction.

---

## 🎓 Training Pipeline Guide

This project uses [MediaPipe](https://google.github.io/mediapipe/) for hand detection and an MLP classifier for gesture recognition.

### Step 1: Collect Image Frames

```bash
python collect_imgs.py
```

* Saves processed hand landmarks for each image into memory.
* Assumes a folder structure like:

```
data/
├── 0/
│   └── 0.jpg, 1.jpg, ...
├── 1/
│   └── 0.jpg, 1.jpg, ...
...
```

* Output: `data.pickle`

### Step 2: Create Dataset

> Done as part of `collect_imgs.py`:

* Extracts hand landmarks from all valid images.
* Stores them in `data.pickle` using `pickle.dump({'data': ..., 'labels': ...})`

### Step 3: Train the Classifier

```bash
python train_classifier.py
```

* Loads `data.pickle`
* Adds Gaussian noise for robustness
* Trains an MLP model using `scikit-learn`
* Evaluates via:

  * Test set accuracy
  * 5-fold cross-validation
  * Confusion matrix (visualized)
* Output: `model.p`

---

## 🎥 Inference (No UI)

```bash
python inference_classifier.py
```

* Uses the webcam to detect hand
* Predicts a letter in real-time
* Draws a box and label on the hand gesture

---

##  Full UI Version (Dual Interfaces)

```bash
python duals.py
```

* Contains 2 interfaces:

  * **Crew**: detects and displays predicted gestures
  * **Customer**: accepts speech/text input and sends it back to Deaf Crew
* Shows camera input, live transcription, and supports speech-to-text via Whisper or local STT tools

---

##  Dependencies

```bash
pip install -r requirements.txt
```

> For Whisper integration (optional):

```bash
pip install openai
```

---

##  Notes

* Ensure your `data/` folder is well-structured with quality images.
* Always run `collect_imgs.py` and regenerate `data.pickle` when adding new gestures.
* Model performance improves with balanced and varied samples per letter.

---

##  User Guide

## Crew Interface:

Click Enable to start gesture recognition

Camera feed shows hand detection

Detected characters appear below the feed

Use Backspace to delete a character

Use Speak to convert detected text to voice

Click Disable to stop the camera

## Customer Interface:

Input text in the box, then click Send to transmit to Deaf Crew screen

Click Reset to clear the input

Click Enable/Disable to start/stop interaction

Detected or sent text will appear below the canvas with animated text display

Both interfaces sync using the shared word_buffer


## 🚀 Coming Soon

* Word-level recognition
* Gesture segmentation
* Model export to ONNX/TF Lite for embedded deployment
