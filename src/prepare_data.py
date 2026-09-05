import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.utils.config import get
from src.utils.landmarks import extract_landmarks, normalize_landmarks

RAW_DIR = Path(get('data.raw_dir'))
PROCESSED_DIR = Path(get('data.processed_dir'))
CLASSES = get('data.classes')
LANDMARK_DIM = get('data.landmark_dim')

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def process_kaggle_asl_dataset(dataset_path):
    """Process Kaggle ASL Alphabet dataset (folder per class with images)."""
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}")
        return None
    
    rows = []
    for class_name in CLASSES:
        class_dir = dataset_path / class_name
        if not class_dir.exists():
            class_dir = dataset_path / class_name.upper()
        if not class_dir.exists():
            continue
        
        print(f"Processing {class_name}...")
        img_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpeg"))
        
        for img_file in tqdm(img_files, desc=class_name):
            frame = cv2.imread(str(img_file))
            if frame is None:
                continue
            
            landmarks = extract_landmarks(frame)
            if landmarks is not None:
                landmarks = normalize_landmarks(landmarks)
                row = {'label': class_name}
                for i, val in enumerate(landmarks):
                    row[f'lm_{i}'] = val
                rows.append(row)
    
    if rows:
        df = pd.DataFrame(rows)
        csv_path = PROCESSED_DIR / "landmarks_asl.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(df)} samples to {csv_path}")
        return df
    return None

def generate_synthetic_landmarks(samples_per_class=500):
    """Generate synthetic landmark data for testing when no dataset available."""
    print("Generating synthetic landmark data...")
    np.random.seed(42)
    rows = []
    
    for class_name in CLASSES:
        for _ in range(samples_per_class):
            landmarks = np.random.randn(LANDMARK_DIM).astype(np.float32) * 0.1
            landmarks[0::2] += 0.5  # x around center
            landmarks[1::2] += 0.5  # y around center
            
            # Add class-specific patterns (simple separation)
            class_idx = CLASSES.index(class_name)
            landmarks[0] += class_idx * 0.02
            landmarks[1] += class_idx * 0.01
            
            row = {'label': class_name}
            for i, val in enumerate(landmarks):
                row[f'lm_{i}'] = val
            rows.append(row)
    
    df = pd.DataFrame(rows)
    csv_path = PROCESSED_DIR / "landmarks_synthetic.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} synthetic samples to {csv_path}")
    return df

def load_processed_data():
    """Load processed landmark CSV."""
    csv_files = list(PROCESSED_DIR.glob("landmarks_*.csv"))
    if not csv_files:
        return None
    # Prefer non-synthetic
    for f in csv_files:
        if 'synthetic' not in f.name:
            return pd.read_csv(f)
    return pd.read_csv(csv_files[0])

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        process_kaggle_asl_dataset(sys.argv[1])
    else:
        generate_synthetic_landmarks()