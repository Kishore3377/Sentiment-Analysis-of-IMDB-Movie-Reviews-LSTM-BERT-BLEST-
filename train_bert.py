"""
Module E: BERT Model
- Fine-tunes bert-base-uncased on IMDB reviews
- Saves model + metrics for comparison
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (BertTokenizer, BertForSequenceClassification,
                          get_linear_schedule_with_warmup)
from torch.optim import AdamW
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             classification_report)
from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────────────
SAVE_DIR    = "saved_models"
BERT_MODEL  = "bert-base-uncased"
MAX_LEN     = 128       # shorter than LSTM to stay within BERT memory budget
BATCH_SIZE  = 16
EPOCHS      = 3
LR          = 2e-5
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Dataset wrapper ───────────────────────────────────────────────────────────
class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids'     : enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels'        : torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ── Training helpers ──────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss, correct = 0, 0

    for batch in tqdm(loader, desc="  Train", leave=False):
        optimizer.zero_grad()
        outputs = model(
            input_ids      = batch['input_ids'].to(DEVICE),
            attention_mask = batch['attention_mask'].to(DEVICE),
            labels         = batch['labels'].to(DEVICE)
        )
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds  = outputs.logits.argmax(dim=-1)
        correct += (preds == batch['labels'].to(DEVICE)).sum().item()

    return total_loss / len(loader), correct / (len(loader) * loader.batch_size)


def eval_epoch(model, loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Eval", leave=False):
            outputs = model(
                input_ids      = batch['input_ids'].to(DEVICE),
                attention_mask = batch['attention_mask'].to(DEVICE),
            )
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch['labels'].numpy())

    return np.array(all_preds), np.array(all_labels)


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_confusion(y_true, y_pred, save_path="plots/bert_confusion.png"):
    os.makedirs("plots", exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=['Negative','Positive'],
                yticklabels=['Negative','Positive'])
    plt.title('BERT – Confusion Matrix', fontsize=13)
    plt.ylabel('Actual'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"   Confusion matrix saved → {save_path}")


# ── Main training function ────────────────────────────────────────────────────
def train_bert():
    print(f"🖥  Using device: {DEVICE}")

    # ── Load pre-processed data ────────────────────────────────────────────────
    X_train = np.load(os.path.join(SAVE_DIR, 'X_train_raw.npy'), allow_pickle=True)
    X_test  = np.load(os.path.join(SAVE_DIR, 'X_test_raw.npy'),  allow_pickle=True)
    y_train = np.load(os.path.join(SAVE_DIR, 'y_train.npy'))
    y_test  = np.load(os.path.join(SAVE_DIR, 'y_test.npy'))

    # Subsample for speed if no GPU available (optional override)
    if DEVICE.type == 'cpu':
        print("⚠️  CPU detected – using 5 000 samples for demo. Set QUICK=False for full run.")
        idx = np.random.default_rng(42).choice(len(X_train), 5000, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
        idx_t = np.random.default_rng(42).choice(len(X_test), 1000, replace=False)
        X_test, y_test = X_test[idx_t], y_test[idx_t]

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # ── Tokenizer & Datasets ───────────────────────────────────────────────────
    print(f"\n📥 Loading tokenizer: {BERT_MODEL} …")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    train_ds = IMDBDataset(list(X_train), list(y_train), tokenizer, MAX_LEN)
    test_ds  = IMDBDataset(list(X_test),  list(y_test),  tokenizer, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Model ──────────────────────────────────────────────────────────────────
    print(f"📦 Loading BERT model: {BERT_MODEL} …")
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=2)
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    # ── Training loop ──────────────────────────────────────────────────────────
    print(f"\n🚀 Fine-tuning BERT for {EPOCHS} epochs …")
    for epoch in range(1, EPOCHS + 1):
        loss, acc = train_epoch(model, train_loader, optimizer, scheduler)
        print(f"  Epoch {epoch}/{EPOCHS}  loss={loss:.4f}  train_acc={acc:.4f}")

    # ── Evaluate ───────────────────────────────────────────────────────────────
    print("\n📊 Evaluating BERT …")
    y_pred, y_true = eval_epoch(model, test_loader)

    metrics = {
        "accuracy" : round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall"   : round(recall_score(y_true, y_pred), 4),
        "f1_score" : round(f1_score(y_true, y_pred), 4),
    }

    print("\n── BERT Results ──────────────────────────────────")
    for k, v in metrics.items():
        print(f"   {k:<12}: {v}")
    print(classification_report(y_true, y_pred, target_names=['Negative','Positive']))

    plot_confusion(y_true, y_pred)

    # ── Save model ─────────────────────────────────────────────────────────────
    bert_save = os.path.join(SAVE_DIR, 'bert_model')
    model.save_pretrained(bert_save)
    tokenizer.save_pretrained(bert_save)

    with open(os.path.join(SAVE_DIR, 'bert_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ BERT model saved → {bert_save}/")
    return metrics


if __name__ == '__main__':
    train_bert()
