import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.config import get

MODEL_PATH = get('model.model_path')
INPUT_DIM = get('model.input_dim')
HIDDEN_DIMS = get('model.hidden_dims')
NUM_CLASSES = get('model.num_classes')
DROPOUT = get('model.dropout')
LR = get('model.learning_rate')
EPOCHS = get('model.epochs')
BATCH_SIZE = get('model.batch_size')
CLASSES = get('data.classes')

def build_model():
    inp = layers.Input(shape=(INPUT_DIM,))
    x = inp
    for units in HIDDEN_DIMS:
        x = layers.Dense(units, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(DROPOUT)(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def load_data():
    processed_dir = Path(get('data.processed_dir'))
    csv_files = list(processed_dir.glob("landmarks_*.csv"))
    if not csv_files:
        raise FileNotFoundError("No processed data found. Run prepare_data.py first.")
    
    for f in csv_files:
        if 'synthetic' not in f.name:
            df = pd.read_csv(f)
            break
    else:
        df = pd.read_csv(csv_files[0])
    
    feature_cols = [c for c in df.columns if c.startswith('lm_')]
    X = df[feature_cols].values.astype(np.float32)
    y = df['label'].values
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    return train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc), le

def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history.history['accuracy'], label='train')
    ax1.plot(history.history['val_accuracy'], label='val')
    ax1.set_title('Accuracy')
    ax1.legend()
    ax2.plot(history.history['loss'], label='train')
    ax2.plot(history.history['val_loss'], label='val')
    ax2.set_title('Loss')
    ax2.legend()
    plt.tight_layout()
    plt.savefig('models/training_history.png')
    plt.close()

def plot_confusion(y_true, y_pred, le):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png')
    plt.close()

if __name__ == "__main__":
    from pathlib import Path
    Path("models").mkdir(exist_ok=True)
    
    (X_train, X_val, y_train, y_val), le = load_data()
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"Classes: {le.classes_}")
    
    model = build_model()
    model.summary()
    
    cb = [
        callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
        callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb,
        verbose=1
    )
    
    plot_history(history)
    
    y_pred = model.predict(X_val, verbose=0).argmax(axis=1)
    print(classification_report(y_val, y_pred, target_names=le.classes_))
    plot_confusion(y_val, y_pred, le)
    
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")