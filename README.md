---
title: Unified ML Internship App
sdk: docker
app_port: 7860
---
# Unified ML Internship App

One Streamlit app that combines the base gender/age model and six task modules. Each task loads its model only when its page is opened, and an error in one task does not stop the others.

| Page | What it does |
|---|---|
| Base | Gender and age from a face photo or webcam snapshot |
| Task 1 Long Hair | Ages 20–30: long hair → Female, short hair → Male. Other ages use the base model's gender |
| Task 2 Senior Citizen | Live camera, video file or webcam snapshot, several people, age > 60 = Senior Citizen, CSV/Excel visit log with timestamps, no duplicate rows from repeated frames |
| Task 3 Voice | Male voices only ("Upload male voice." otherwise). Age > 60: Senior Citizen plus emotion. Age ≤ 60: age only |
| Task 4 Sign Language | 8 signs (Hungry, I need help, Thirsty, Wait, Careful, Understand, Book, Write) from image, snapshot or video, only inside the 18:00–22:00 IST window. Shows word and confidence. Live camera, snapshot and video upload are available |
| Task 5 Car Colour | Detects cars and people, reports car colour, counts cars and people. Blue cars get a red rectangle, other cars a blue one |
| Task 6 Nationality | Predicts a dataset category (not real nationality), plus emotion, age and dress colour according to the category rules |

## Setup

Two Python environments are needed because the models were trained on different TensorFlow versions.

1. **Main environment** (Python 3.10+):
   ```
   pip install -r requirements.txt
   ```
2. **TensorFlow 2.10 environment** (for Tasks 4, 5, 6):
   ```
   python -m venv tf210
   tf210\Scripts\pip install -r requirements-tf210.txt
   ```
3. Tell the app where that environment's Python is, either in the sidebar Settings box or before launch.
   PowerShell: `$env:TF210_PYTHON = "D:\path\to\tf210\Scripts\python.exe"`
4. Optional: put `face_detection_yunet_2023mar.onnx` (OpenCV Zoo) in `assets/models/` for a better face detector. Without it the app uses OpenCV Haar cascades.
5. Run:
   ```
   streamlit run app.py
   ```

The app sets `KERAS_BACKEND=torch` itself. The TensorFlow 2.10 worker runs on the CPU by default to avoid GPU memory clashes with the main app (set `WORKER_USE_GPU=1` to use the GPU). Tasks 4–6 run in a separate worker process that the app starts automatically.

## Model files
Weights are in the folders listed in `model_registry.py`. The voice model (about 1.3 GB) and the car detector (about 120 MB) exceed GitHub's 100 MB file limit, so use Git LFS or a release download for them.

## Known limitations (measured on our own checks)
- Base model gender accuracy about 0.91 and age error about 9 years on UTKFace faces; it underestimates old ages and finds roughly half of true seniors.
- Sign language validation accuracy is 100% but the split has near-duplicate frames, so real-world accuracy is likely lower.
- Task 5 uses YOLOv8n for detecting cars and people by default (car F1 about 0.70, person about 0.56 in our check). The custom KITTI detector is available but weaker.
- Task 6 categories are dataset labels. Top-1 accuracy about 62% on a 50-face check; Indian and Others recall is weak. Low-confidence results show "Uncertain". Emotion accuracy about 56%.
- Task 1 depends on a face detector and the hair model's direction has been verified only on limited samples.
- Task 3 accuracy has not been measured.
- Live camera (Tasks 2 and 4) needs a browser WebRTC connection. It works on a local run; on a hosted site it may not connect, so snapshot and video upload are provided as alternatives.

## Tests
```
pip install pytest
pytest
```
The suite runs offline in about a second and needs no model weights (they are stored in a private Hugging Face repository). By default `tests/conftest.py` swaps in a small stand-in for `model_registry`, so only the app's own logic is tested.

| Test file | What it checks |
|---|---|
| `test_tracker.py` | Task 2: one person = one track, a visit closes after a gap, senior rule is strictly age > 60, median age |
| `test_recorder.py` | Task 2: visit-log CSV header, timestamps, duration, Yes/No flag, ids continue after restart, no duplicate header |
| `test_sign_language.py` | Task 4: 6:00 PM to 10:00 PM IST window (edges, UTC server clock, overnight wrap), model never runs when closed, word + confidence, "uncertain" result, video frame averaging and sampling |
| `test_car_colour.py` | Task 5: box cropping, colour name, reliability rules for low confidence and tiny crops, bad boxes |
| `test_worker_protocol_and_throttle.py` | TensorFlow worker data format round trip and the camera frame throttle |
| `test_model_files.py` | Optional: every registered model file exists |

To check the real model files, set a Hugging Face token and run:
```
$env:HF_TOKEN = "<your token>"; $env:USE_REAL_MODELS = "1"; pytest tests/test_model_files.py
```
The model-backed behaviour of Tasks 1, 3 and 6 (and the accuracy numbers above) is not covered by these automated tests. Check it by running the app on sample images and audio.
