"""
Module D: LSTM Model
- Builds, trains, and evaluates an LSTM network
- Saves model + metrics for comparison
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Embedding, LSTM, Dense,
                                     Dropout, Bidirectional, SpatialDropout1D)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             classification_report)
import json
# ── Constants ─────────────────────────────────────────────────────────────────
SAVE_DIR   = "saved_models"
MAX_VOCAB  = 20000
MAX_LEN    = 200
EMBED_DIM  = 128
LSTM_UNITS = 128
DROPOUT    = 0.3
EPOCHS     = 10
BATCH_SIZE = 64


def build_lstm_model():
    model = Sequential([
        Embedding(MAX_VOCAB, EMBED_DIM, input_length=MAX_LEN),
        SpatialDropout1D(0.2),
        Bidirectional(LSTM(LSTM_UNITS, return_sequences=True, dropout=0.2, recurrent_dropout=0.1)),
        Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.1)),
        Dense(64, activation='relu'),
        Dropout(DROPOUT),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


def plot_history(history, save_path="plots/lstm_training.png"):
    os.makedirs("plots", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history['accuracy'],     label='Train Accuracy',      color='steelblue')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    axes[0].set_title('LSTM – Accuracy over Epochs', fontsize=13)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'],     label='Train Loss',      color='steelblue')
    axes[1].plot(history.history['val_loss'], label='Validation Loss', color='orange')
    axes[1].set_title('LSTM – Loss over Epochs', fontsize=13)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"   Training plot saved → {save_path}")


def plot_confusion(y_true, y_pred, save_path="plots/lstm_confusion.png"):
    os.makedirs("plots", exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative','Positive'],
                yticklabels=['Negative','Positive'])
    plt.title('LSTM – Confusion Matrix', fontsize=13)
    plt.ylabel('Actual'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"   Confusion matrix saved → {save_path}")


def train_lstm():
    # ── Load pre-processed data ────────────────────────────────────────────────
    X_train = np.load(os.path.join(SAVE_DIR, 'X_train_seq.npy'))
    X_test  = np.load(os.path.join(SAVE_DIR, 'X_test_seq.npy'))
    y_train = np.load(os.path.join(SAVE_DIR, 'y_train.npy'))
    y_test  = np.load(os.path.join(SAVE_DIR, 'y_test.npy'))

    print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")

    # ── Build & train ──────────────────────────────────────────────────────────
    model = build_lstm_model()

    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True, verbose=1),
        ModelCheckpoint(os.path.join(SAVE_DIR, 'lstm_model.h5'),
                        save_best_only=True, verbose=1)
    ]

    print("\n🚀 Training LSTM …")
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    plot_history(history)

    # ── Evaluate ───────────────────────────────────────────────────────────────
    print("\n📊 Evaluating LSTM …")
    y_pred_prob = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    metrics = {
        "accuracy" : round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall"   : round(recall_score(y_test, y_pred), 4),
        "f1_score" : round(f1_score(y_test, y_pred), 4),
    }

    print("\n── LSTM Results ──────────────────────────────────")
    for k, v in metrics.items():
        print(f"   {k:<12}: {v}")
    print(classification_report(y_test, y_pred, target_names=['Negative','Positive']))

    plot_confusion(y_test, y_pred)

    # Save metrics
    with open(os.path.join(SAVE_DIR, 'lstm_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ LSTM model saved → {SAVE_DIR}/lstm_model.h5")
    return metrics


if __name__ == '__main__':
    train_lstm()
