import cv2
import numpy as np
from pathlib import Path
from src.utils.config import get
import mediapipe as mp
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python import BaseOptions

LANDMARK_DIM = get('data.landmark_dim', 84)
MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "hand_landmarker.task"

_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)) if MODEL_PATH.exists() else BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

_landmarker = HandLandmarker.create_from_options(_options)

def extract_landmarks(frame):
    """Extract normalized landmarks from both hands. Returns (84,) vector or None."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
    result = _landmarker.detect_for_video(mp_image, timestamp)
    
    landmarks = np.zeros(LANDMARK_DIM, dtype=np.float32)
    detected = False
    
    if result.hand_landmarks:
        for idx, hand_landmarks in enumerate(result.hand_landmarks):
            if idx >= 2:
                break
            base = idx * 42
            for i, lm in enumerate(hand_landmarks):
                landmarks[base + i*2] = lm.x
                landmarks[base + i*2 + 1] = lm.y
            detected = True
    
    return landmarks if detected else None

def draw_landmarks(frame, landmarks):
    if landmarks is None:
        return frame
    h, w = frame.shape[:2]
    for hand_idx in range(2):
        base = hand_idx * 42
        hand_lms = landmarks[base:base+42]
        if np.all(hand_lms == 0):
            continue
        for i in range(0, 42, 2):
            x = int(hand_lms[i] * w)
            y = int(hand_lms[i+1] * h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
    return frame

def normalize_landmarks(landmarks):
    if landmarks is None:
        return None
    normalized = landmarks.copy()
    for hand_idx in range(2):
        base = hand_idx * 42
        hand_lms = landmarks[base:base+42]
        if np.all(hand_lms == 0):
            continue
        wrist_x, wrist_y = hand_lms[0], hand_lms[1]
        for i in range(0, 42, 2):
            normalized[base + i] -= wrist_x
            normalized[base + i + 1] -= wrist_y
    return normalized