# ISL/ASL Sign Language Translator

Real-time sign language recognition for letters and words using MediaPipe landmarks + MLP classifier.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare Data
```bash
# Option A: Generate synthetic data (for testing)
python run.py prepare

# Option B: Process Kaggle ASL Alphabet dataset
# Download from: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
python run.py prepare path/to/asl_alphabet_train
```

### 2. Train Model
```bash
python run.py train
```
Model saved to `models/isl_asl_landmark_mlp.h5`

### 3. Run App
```bash
python run.py app
```
Opens Streamlit UI with webcam feed, recognized text box, confidence bar, and TTS.

## Classes
A-Z, space, delete, nothing (29 classes)

## Architecture
- MediaPipe Hands → 84-d landmark vector (21 pts × 2 hands × 2 coords)
- MLP: 84 → 128 → 64 → 29 (softmax)
- Real-time inference ~30 FPS

## Extending for Sentences (Phase 2)
- Collect sequence data (T frames × 84)
- Replace MLP with LSTM/Transformer
- Add CTC loss or sequence-to-sequence decoder