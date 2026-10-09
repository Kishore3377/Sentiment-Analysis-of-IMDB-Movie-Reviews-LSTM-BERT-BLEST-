"""
Module F: BLEST Ensemble Model
BLEST — BERT-LSTM Ensemble Sentiment Technology

Strategy:
  1. Load saved LSTM + BERT models
  2. Extract probability outputs from both on the full test set
  3. Train a Logistic Regression meta-learner on [lstm_prob, bert_prob]
  4. Save the stacked ensemble (meta-classifier + calibration)
  5. Evaluate — expected accuracy ~0.94–0.96

Run after train_lstm.py and train_bert.py:
    python train_ensemble.py
"""

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)
from tqdm import tqdm

SAVE_DIR   = "saved_models"
BATCH_SIZE = 64


# ─────────────────────────────────────────────────────────────────────────────
#  Extract LSTM probabilities
# ─────────────────────────────────────────────────────────────────────────────
def get_lstm_probs(X_seq, batch_size=BATCH_SIZE):
    """Return raw sigmoid probabilities from the LSTM for all samples."""
    from tensorflow.keras.models import load_model

    model_path = os.path.join(SAVE_DIR, "lstm_model.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError("LSTM model not found. Run train_lstm.py first.")

    model = load_model(model_path)
    probs = model.predict(X_seq, batch_size=batch_size, verbose=1)
    return probs.flatten()          # shape (N,)  — P(Positive)


# ─────────────────────────────────────────────────────────────────────────────
#  Extract BERT probabilities
# ─────────────────────────────────────────────────────────────────────────────
def get_bert_probs(X_raw, batch_size=32):
    """Return softmax P(Positive) from fine-tuned BERT for all samples."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import BertTokenizer, BertForSequenceClassification

    bert_dir = os.path.join(SAVE_DIR, "bert_model")
    if not os.path.exists(bert_dir):
        raise FileNotFoundError("BERT model not found. Run train_bert.py first.")

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(bert_dir)
    model     = BertForSequenceClassification.from_pretrained(bert_dir)
    model.to(device)
    model.eval()

    class TextDataset(Dataset):
        def __init__(self, texts, tok, max_len=128):
            self.texts = texts
            self.tok   = tok
            self.max_len = max_len
        def __len__(self):  return len(self.texts)
        def __getitem__(self, idx):
            enc = self.tok(
                self.texts[idx],
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            return {
                "input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
            }

    loader = DataLoader(TextDataset(list(X_raw), tokenizer),
                        batch_size=batch_size, shuffle=False, num_workers=0)
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  BERT inference"):
            out    = model(
                input_ids      = batch["input_ids"].to(device),
                attention_mask = batch["attention_mask"].to(device),
            )
            probs  = torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs)

    return np.array(all_probs)      # shape (N,)  — P(Positive)


# ─────────────────────────────────────────────────────────────────────────────
#  Build feature matrix
# ─────────────────────────────────────────────────────────────────────────────
def build_features(lstm_probs, bert_probs):
    """
    Stacked feature vector for the meta-learner:
      [lstm_p, bert_p, mean_p, abs_diff, product, weighted_p]
    Richer features give the LR meta-learner more signal.
    """
    mean_p      = (lstm_probs + bert_probs) / 2
    diff        = np.abs(lstm_probs - bert_probs)
    product     = lstm_probs * bert_probs
    weighted    = 0.4 * lstm_probs + 0.6 * bert_probs  # BERT = higher weight
    return np.column_stack([lstm_probs, bert_probs,
                             mean_p, diff, product, weighted])


# ─────────────────────────────────────────────────────────────────────────────
#  Plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion(y_true, y_pred, save_path="plots/blest_confusion.png"):
    os.makedirs("plots", exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.title("BLEST Ensemble – Confusion Matrix", fontsize=13)
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"   Confusion matrix saved → {save_path}")


def plot_roc(y_true, y_prob, save_path="plots/blest_roc.png"):
    os.makedirs("plots", exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc         = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="purple", lw=2,
             label=f"BLEST ROC (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("BLEST – ROC Curve", fontsize=13)
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"   ROC curve saved        → {save_path}")


def plot_prob_distribution(y_true, y_prob, save_path="plots/blest_prob_dist.png"):
    os.makedirs("plots", exist_ok=True)
    pos_probs = y_prob[y_true == 1]
    neg_probs = y_prob[y_true == 0]
    plt.figure(figsize=(8, 4))
    plt.hist(pos_probs, bins=40, alpha=0.6, color="green",  label="Positive reviews")
    plt.hist(neg_probs, bins=40, alpha=0.6, color="red",    label="Negative reviews")
    plt.axvline(0.5, color="black", linestyle="--", label="Decision boundary (0.5)")
    plt.xlabel("P(Positive)"); plt.ylabel("Count")
    plt.title("BLEST – Predicted Probability Distribution", fontsize=13)
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"   Prob distribution saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def train_ensemble():
    print("=" * 60)
    print("  BLEST — BERT-LSTM Ensemble Sentiment Technology")
    print("=" * 60)

    # ── Load pre-processed data ────────────────────────────────────────────────
    print("\n📂 Loading pre-processed data …")
    X_train_seq = np.load(os.path.join(SAVE_DIR, "X_train_seq.npy"))
    X_test_seq  = np.load(os.path.join(SAVE_DIR, "X_test_seq.npy"))
    X_train_raw = np.load(os.path.join(SAVE_DIR, "X_train_raw.npy"), allow_pickle=True)
    X_test_raw  = np.load(os.path.join(SAVE_DIR, "X_test_raw.npy"),  allow_pickle=True)
    y_train     = np.load(os.path.join(SAVE_DIR, "y_train.npy"))
    y_test      = np.load(os.path.join(SAVE_DIR, "y_test.npy"))
    print(f"   Train: {len(y_train)} | Test: {len(y_test)}")

    # ── Get base-model probabilities on TRAINING set (for meta-learner fit) ───
    print("\n🔵 Extracting LSTM probabilities on TRAIN set …")
    lstm_train_probs = get_lstm_probs(X_train_seq)

    print("\n🟠 Extracting BERT probabilities on TRAIN set …")
    bert_train_probs = get_bert_probs(X_train_raw)

    X_train_meta = build_features(lstm_train_probs, bert_train_probs)

    # ── Train meta-learner ─────────────────────────────────────────────────────
    print("\n🟣 Training BLEST meta-learner (Logistic Regression + Calibration) …")
    base_lr  = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver="lbfgs")
    meta_clf = CalibratedClassifierCV(base_lr, cv=5, method="isotonic")
    meta_clf.fit(X_train_meta, y_train)
    print("   Meta-learner fitted ✅")

    # ── Evaluate on TEST set ───────────────────────────────────────────────────
    print("\n🔵 Extracting LSTM probabilities on TEST set …")
    lstm_test_probs = get_lstm_probs(X_test_seq)

    print("\n🟠 Extracting BERT probabilities on TEST set …")
    bert_test_probs = get_bert_probs(X_test_raw)

    X_test_meta = build_features(lstm_test_probs, bert_test_probs)

    y_pred      = meta_clf.predict(X_test_meta)
    y_prob      = meta_clf.predict_proba(X_test_meta)[:, 1]

    metrics = {
        "accuracy"  : round(accuracy_score(y_test, y_pred),              4),
        "precision" : round(precision_score(y_test, y_pred),             4),
        "recall"    : round(recall_score(y_test, y_pred),                4),
        "f1_score"  : round(f1_score(y_test, y_pred),                    4),
        "roc_auc"   : round(roc_auc_score(y_test, y_prob),               4),
    }

    print("\n── BLEST Ensemble Results ────────────────────────────────────")
    for k, v in metrics.items():
        print(f"   {k:<12}: {v}")
    print()
    print(classification_report(y_test, y_pred,
                                target_names=["Negative", "Positive"]))

    # ── Load individual model metrics for comparison ───────────────────────────
    lstm_m_path = os.path.join(SAVE_DIR, "lstm_metrics.json")
    bert_m_path = os.path.join(SAVE_DIR, "bert_metrics.json")
    lstm_m = json.load(open(lstm_m_path)) if os.path.exists(lstm_m_path) else {}
    bert_m = json.load(open(bert_m_path)) if os.path.exists(bert_m_path) else {}

    if lstm_m and bert_m:
        print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"{'Model':<10} {'Accuracy':>10} {'Precision':>10} {'Recall':>8} {'F1':>8}")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        for name, m in [("LSTM", lstm_m), ("BERT", bert_m), ("BLEST ★", metrics)]:
            print(f"{name:<10} {m['accuracy']:>10.4f} {m['precision']:>10.4f}"
                  f" {m['recall']:>8.4f} {m['f1_score']:>8.4f}")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # ── Plots ──────────────────────────────────────────────────────────────────
    plot_confusion(y_test, y_pred)
    plot_roc(y_test, y_prob)
    plot_prob_distribution(y_test, y_prob)

    # ── Save ensemble artifacts ────────────────────────────────────────────────
    meta_path = os.path.join(SAVE_DIR, "blest_meta_clf.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(meta_clf, f)

    # Save the test-set base probabilities so we can skip re-inference in API
    np.save(os.path.join(SAVE_DIR, "blest_lstm_test_probs.npy"), lstm_test_probs)
    np.save(os.path.join(SAVE_DIR, "blest_bert_test_probs.npy"), bert_test_probs)

    with open(os.path.join(SAVE_DIR, "blest_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ BLEST model saved → {meta_path}")
    print(f"✅ BLEST metrics    → {SAVE_DIR}/blest_metrics.json")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
#  Inference helper  (used by api_service.py and app.py)
# ─────────────────────────────────────────────────────────────────────────────
def predict_blest_single(text: str,
                          lstm_model, lstm_tokenizer,
                          bert_model, bert_tokenizer,
                          meta_clf,
                          lstm_max_len=200, bert_max_len=128) -> dict:
    """
    End-to-end BLEST prediction for a single review string.
    Returns {"label": "Positive"|"Negative", "confidence": float, "model": "BLEST"}
    """
    import re, pickle
    import torch
    import numpy as np
    from nltk.corpus import stopwords
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    STOP_WORDS = set(stopwords.words("english"))

    def _clean(t):
        t = re.sub(r"<.*?>", " ", t)
        t = re.sub(r"[^a-zA-Z\s]", " ", t)
        t = t.lower().strip()
        return " ".join([w for w in t.split() if w not in STOP_WORDS and len(w) > 1])

    # LSTM prob
    seq    = lstm_tokenizer.texts_to_sequences([_clean(text)])
    padded = pad_sequences(seq, maxlen=lstm_max_len, truncating="post")
    lstm_p = float(lstm_model.predict(padded, verbose=0)[0][0])

    # BERT prob
    enc = bert_tokenizer(text, max_length=bert_max_len, padding="max_length",
                         truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = bert_model(**enc).logits
    bert_p = float(torch.softmax(logits, dim=-1)[0][1].numpy())

    # Meta-learner
    features = build_features(
        np.array([lstm_p]), np.array([bert_p])
    )
    proba      = meta_clf.predict_proba(features)[0]
    pred_idx   = int(np.argmax(proba))
    label      = "Positive" if pred_idx == 1 else "Negative"
    confidence = round(float(proba[pred_idx]) * 100, 2)

    return {"label": label, "confidence": confidence, "model": "BLEST"}


if __name__ == "__main__":
    train_ensemble()
