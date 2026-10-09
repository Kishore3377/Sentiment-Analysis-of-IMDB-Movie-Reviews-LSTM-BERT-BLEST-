# 🎬 Sentiment Analysis of IMDB Movie Reviews
## LSTM vs BERT | Nandha College of Technology

---

## 📁 Project Structure

```
sentiment_analysis/
│
├── preprocess.py        # Module A & B — Data download + cleaning
├── train_lstm.py        # Module D   — LSTM training & evaluation
├── train_bert.py        # Module E   — BERT fine-tuning & evaluation
├── compare_models.py    # Module G   — Comparison chart (Figure 3)
├── app.py               # Module H/I — Streamlit UI
├── requirements.txt     # All dependencies
│
├── saved_models/        # Created automatically during training
│   ├── tokenizer.pkl
│   ├── lstm_model.h5
│   ├── bert_model/
│   ├── lstm_metrics.json
│   └── bert_metrics.json
│
└── plots/               # Created automatically
    ├── lstm_training.png
    ├── lstm_confusion.png
    ├── bert_confusion.png
    └── comparison_graph.png
```

---

## ⚙️ Step-by-Step Setup & Run Guide

### Step 1 — Install Python
Make sure Python **3.9 or 3.10** is installed.
```bash
python --version
```

### Step 2 — Create a Virtual Environment (recommended)
```bash
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on Mac/Linux:
source venv/bin/activate
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```
> ⏳ This will take a few minutes (TensorFlow + PyTorch + Transformers).

---

## 🚀 Running the Project

### Step 4 — Preprocess the IMDB Dataset
```bash
python preprocess.py
```
**What it does:**
- Downloads the IMDB dataset (50 000 reviews) from HuggingFace
- Cleans text (removes HTML, punctuation, stop-words)
- Tokenises and pads sequences for LSTM
- Saves arrays to `saved_models/`

**Expected output:**
```
📥 Loading IMDB dataset …
   Total reviews : 50000
🧹 Cleaning text …
   Train size : 40000 | Test size : 10000
🔢 Tokenising for LSTM …
✅ Preprocessing complete. Artifacts saved to saved_models/
```

---

### Step 5 — Train the LSTM Model
```bash
python train_lstm.py
```
**What it does:**
- Builds a Bidirectional LSTM with Embedding + Dropout layers
- Trains for up to 10 epochs with Early Stopping
- Saves the best model to `saved_models/lstm_model.h5`
- Prints Accuracy, Precision, Recall, F1 Score
- Saves plots to `plots/`

**Expected results (matches Table 1 in the paper):**
```
── LSTM Results ──────────────────────────────────
   accuracy    : 0.86
   precision   : 0.84
   recall      : 0.82
   f1_score    : 0.83
```
> ⏱ Training time: ~15–30 minutes on CPU, ~5 minutes on GPU.

---

### Step 6 — Fine-tune the BERT Model
```bash
python train_bert.py
```
**What it does:**
- Downloads `bert-base-uncased` from HuggingFace (~440 MB, one-time)
- Fine-tunes for 3 epochs on IMDB reviews
- Saves model to `saved_models/bert_model/`
- Prints metrics + confusion matrix

**Expected results:**
```
── BERT Results ──────────────────────────────────
   accuracy    : 0.93
   precision   : 0.92
   recall      : 0.91
   f1_score    : 0.91
```
> ⏱ Training time: ~20 min on GPU, ~2–3 hours on CPU (auto-subsamples to 5 000 reviews on CPU).  
> 💡 For full CPU run, open `train_bert.py` and change the subsample block.

---

### Step 7 — Generate Comparison Chart (Figure 3)
```bash
python compare_models.py
```
**Output:** `plots/comparison_graph.png` — bar chart comparing LSTM vs BERT.

---

### Step 8 — Launch the Streamlit App
```bash
streamlit run app.py
```
Then open your browser at: **http://localhost:8501**

**App features:**
- 📝 Type any movie review and click **Analyse Sentiment**
- 🔵 LSTM result shown on the left with confidence score
- 🟠 BERT result shown on the right with confidence score
- 📊 Figure 3 comparison chart always visible
- 📋 Table 1 metrics displayed below the chart

---

## 📊 Expected Results (from the paper)

| Algorithm | Accuracy | Precision | Recall | F1 Score |
|-----------|----------|-----------|--------|----------|
| LSTM      | 0.86     | 0.84      | 0.82   | 0.83     |
| BERT      | 0.93     | 0.92      | 0.91   | 0.91     |

---

## 🖥️ Hardware Recommendations

| Component | Minimum       | Recommended      |
|-----------|---------------|------------------|
| RAM       | 8 GB          | 16 GB            |
| GPU       | Optional      | NVIDIA (4 GB+)   |
| Disk      | 5 GB free     | 10 GB free       |
| Python    | 3.9           | 3.10             |

---

## ❓ Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| BERT download fails | Check internet connection; HuggingFace may be slow |
| Out of memory (BERT) | Reduce `BATCH_SIZE` in `train_bert.py` to 8 |
| Streamlit not found | Run `pip install streamlit` |
| Models missing in app | Run Steps 4–6 first before launching the app |

---

## 🔑 Key Technologies

- **Dataset:** [IMDB Large Movie Review Dataset](https://huggingface.co/datasets/imdb) (50 000 reviews)
- **LSTM:** TensorFlow / Keras Bidirectional LSTM
- **BERT:** HuggingFace `bert-base-uncased` (fine-tuned)
- **UI:** Streamlit
- **Metrics:** scikit-learn (Accuracy, Precision, Recall, F1)
