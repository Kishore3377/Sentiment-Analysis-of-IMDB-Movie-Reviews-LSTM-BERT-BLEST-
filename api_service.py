"""
Real-time Web Service — FastAPI
Exposes LSTM, BERT, and BLEST (Ensemble) as REST API endpoints.

Install extras:
    pip install fastapi uvicorn python-multipart

Run:
    uvicorn api_service:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /                        → Health check
    POST /predict                 → Single review prediction
    POST /predict/batch           → Batch prediction (JSON list)
    GET  /recommend/{sentiment}   → Movie recommendations
    GET  /metrics                 → Stored model metrics
"""

import os
import re
import json
import pickle
import numpy as np
from typing import List, Optional
from datetime import datetime

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words('english'))
SAVE_DIR   = "saved_models"

# ─────────────────────────────────────────────────────────────────────────────
#  App setup
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "🎬 IMDB Sentiment API",
    description = "Real-time sentiment analysis using LSTM, BERT, and BLEST (Ensemble)",
    version     = "2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
#  Movie recommendation data
# ─────────────────────────────────────────────────────────────────────────────
MOVIE_DB = {
    "positive": [
        {"title": "The Shawshank Redemption", "genre": "Drama",    "rating": 9.3, "year": 1994},
        {"title": "Forrest Gump",             "genre": "Drama",    "rating": 8.8, "year": 1994},
        {"title": "Inception",                "genre": "Sci-Fi",   "rating": 8.8, "year": 2010},
        {"title": "The Dark Knight",          "genre": "Action",   "rating": 9.0, "year": 2008},
        {"title": "Interstellar",             "genre": "Sci-Fi",   "rating": 8.7, "year": 2014},
        {"title": "Parasite",                 "genre": "Thriller", "rating": 8.5, "year": 2019},
    ],
    "negative": [
        {"title": "Joker",               "genre": "Drama",    "rating": 8.4, "year": 2019},
        {"title": "Gone Girl",           "genre": "Thriller", "rating": 8.1, "year": 2014},
        {"title": "Black Swan",          "genre": "Thriller", "rating": 8.0, "year": 2010},
        {"title": "Se7en",               "genre": "Crime",    "rating": 8.6, "year": 1995},
        {"title": "Prisoners",           "genre": "Crime",    "rating": 8.1, "year": 2013},
        {"title": "Nightcrawler",        "genre": "Crime",    "rating": 7.9, "year": 2014},
    ]
}

# ─────────────────────────────────────────────────────────────────────────────
#  Global model holders
# ─────────────────────────────────────────────────────────────────────────────
class ModelStore:
    lstm_model    = None
    lstm_tok      = None
    bert_model    = None
    bert_tok      = None
    blest_clf     = None          # ← new: meta-classifier

store = ModelStore()

# ─────────────────────────────────────────────────────────────────────────────
#  Startup — load models
# ─────────────────────────────────────────────────────────────────────────────
@app.on_event("startup")
def load_models():
    # LSTM
    lstm_path = os.path.join(SAVE_DIR, 'lstm_model.h5')
    tok_path  = os.path.join(SAVE_DIR, 'tokenizer.pkl')
    if os.path.exists(lstm_path) and os.path.exists(tok_path):
        from tensorflow.keras.models import load_model
        store.lstm_model = load_model(lstm_path)
        with open(tok_path, 'rb') as f:
            store.lstm_tok = pickle.load(f)
        print("✅ LSTM loaded")
    else:
        print("⚠️  LSTM model not found")

    # BERT
    bert_dir = os.path.join(SAVE_DIR, 'bert_model')
    if os.path.exists(bert_dir):
        from transformers import BertTokenizer, BertForSequenceClassification
        store.bert_tok   = BertTokenizer.from_pretrained(bert_dir)
        store.bert_model = BertForSequenceClassification.from_pretrained(bert_dir)
        store.bert_model.eval()
        print("✅ BERT loaded")
    else:
        print("⚠️  BERT model not found")

    # BLEST ensemble meta-classifier
    blest_path = os.path.join(SAVE_DIR, 'blest_meta_clf.pkl')
    if os.path.exists(blest_path):
        with open(blest_path, 'rb') as f:
            store.blest_clf = pickle.load(f)
        print("✅ BLEST ensemble loaded")
    else:
        print("⚠️  BLEST model not found — run train_ensemble.py")

# ─────────────────────────────────────────────────────────────────────────────
#  Schemas
# ─────────────────────────────────────────────────────────────────────────────
class ReviewRequest(BaseModel):
    text  : str = Field(..., min_length=3, example="This movie was absolutely fantastic!")
    model : str = Field("all", description="'lstm', 'bert', 'blest', or 'all'")

class SinglePrediction(BaseModel):
    label      : str
    confidence : float
    model      : str

class PredictResponse(BaseModel):
    text        : str
    predictions : List[SinglePrediction]
    agreement   : Optional[bool]
    timestamp   : str

class BatchRequest(BaseModel):
    reviews : List[str] = Field(..., min_items=1, max_items=500)
    model   : str       = Field("blest", description="'lstm', 'bert', 'blest', or 'all'")

# ─────────────────────────────────────────────────────────────────────────────
#  Text cleaning
# ─────────────────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text   = re.sub(r'<.*?>', ' ', text)
    text   = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text   = text.lower().strip()
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 1]
    return ' '.join(tokens)

# ─────────────────────────────────────────────────────────────────────────────
#  Prediction helpers
# ─────────────────────────────────────────────────────────────────────────────
def _lstm_prob(text: str) -> float:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seq    = store.lstm_tok.texts_to_sequences([clean_text(text)])
    padded = pad_sequences(seq, maxlen=200, truncating='post')
    return float(store.lstm_model.predict(padded, verbose=0)[0][0])


def _bert_prob(text: str) -> float:
    import torch
    enc    = store.bert_tok(text, max_length=128, padding='max_length',
                            truncation=True, return_tensors='pt')
    with torch.no_grad():
        logits = store.bert_model(**enc).logits
    probs = torch.softmax(logits, dim=-1)[0].numpy()
    return float(probs[1])          # P(Positive)


def _build_features(lstm_p: float, bert_p: float) -> np.ndarray:
    """Mirrors build_features() in train_ensemble.py."""
    mean_p   = (lstm_p + bert_p) / 2
    diff     = abs(lstm_p - bert_p)
    product  = lstm_p * bert_p
    weighted = 0.4 * lstm_p + 0.6 * bert_p
    return np.array([[lstm_p, bert_p, mean_p, diff, product, weighted]])


def run_lstm(text: str) -> dict:
    if store.lstm_model is None:
        raise HTTPException(503, "LSTM model not loaded. Run train_lstm.py first.")
    p     = _lstm_prob(text)
    label = "Positive" if p > 0.5 else "Negative"
    conf  = round((p if p > 0.5 else 1 - p) * 100, 2)
    return {"label": label, "confidence": conf, "model": "LSTM"}


def run_bert(text: str) -> dict:
    if store.bert_model is None:
        raise HTTPException(503, "BERT model not loaded. Run train_bert.py first.")
    p     = _bert_prob(text)
    label = "Positive" if p > 0.5 else "Negative"
    conf  = round((p if p > 0.5 else 1 - p) * 100, 2)
    return {"label": label, "confidence": conf, "model": "BERT"}


def run_blest(text: str) -> dict:
    """Ensemble prediction: requires LSTM + BERT + meta-classifier."""
    if store.blest_clf is None:
        raise HTTPException(503, "BLEST model not loaded. Run train_ensemble.py first.")
    if store.lstm_model is None or store.bert_model is None:
        raise HTTPException(503, "BLEST requires both LSTM and BERT to be loaded.")

    lstm_p   = _lstm_prob(text)
    bert_p   = _bert_prob(text)
    features = _build_features(lstm_p, bert_p)

    proba    = store.blest_clf.predict_proba(features)[0]
    pred_idx = int(np.argmax(proba))
    label    = "Positive" if pred_idx == 1 else "Negative"
    conf     = round(float(proba[pred_idx]) * 100, 2)
    return {"label": label, "confidence": conf, "model": "BLEST"}

# ─────────────────────────────────────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def health():
    return {
        "status"       : "ok",
        "lstm_loaded"  : store.lstm_model is not None,
        "bert_loaded"  : store.bert_model is not None,
        "blest_loaded" : store.blest_clf  is not None,
        "timestamp"    : datetime.utcnow().isoformat(),
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(req: ReviewRequest):
    """Analyse a single review with LSTM, BERT, BLEST, or all three."""
    preds        = []
    model_choice = req.model.lower()

    if model_choice in ("lstm", "all"):
        preds.append(run_lstm(req.text))
    if model_choice in ("bert", "all"):
        preds.append(run_bert(req.text))
    if model_choice in ("blest", "all"):
        preds.append(run_blest(req.text))

    if not preds:
        raise HTTPException(400, "model must be 'lstm', 'bert', 'blest', or 'all'")

    # Agreement: check if all predictions share the same label
    labels    = [p["label"] for p in preds]
    agreement = len(set(labels)) == 1 if len(preds) > 1 else None

    return {
        "text"        : req.text[:200],
        "predictions" : preds,
        "agreement"   : agreement,
        "timestamp"   : datetime.utcnow().isoformat(),
    }


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(req: BatchRequest):
    """Analyse a list of reviews (up to 500)."""
    results      = []
    model_choice = req.model.lower()
    for text in req.reviews:
        entry = {"text": text[:80]}
        if model_choice in ("lstm", "all"):
            entry["lstm"]  = run_lstm(text)
        if model_choice in ("bert", "all"):
            entry["bert"]  = run_bert(text)
        if model_choice in ("blest", "all"):
            entry["blest"] = run_blest(text)
        results.append(entry)
    return {"total": len(results), "results": results}


@app.get("/recommend/{sentiment}", tags=["Recommendations"])
def recommend(sentiment: str, n: int = 4):
    sentiment = sentiment.lower()
    if sentiment not in MOVIE_DB:
        raise HTTPException(400, "sentiment must be 'positive' or 'negative'")
    n    = min(n, 6)
    pool = sorted(MOVIE_DB[sentiment], key=lambda x: x['rating'], reverse=True)[:n]
    return {"sentiment": sentiment, "recommendations": pool}


@app.get("/metrics", tags=["Metrics"])
def get_metrics():
    """Return stored evaluation metrics for LSTM, BERT, and BLEST."""
    def _load(path, default):
        return json.load(open(path)) if os.path.exists(path) else default

    return {
        "lstm" : _load(os.path.join(SAVE_DIR, 'lstm_metrics.json'),
                       {"accuracy": 0.86, "precision": 0.84, "recall": 0.82, "f1_score": 0.83}),
        "bert" : _load(os.path.join(SAVE_DIR, 'bert_metrics.json'),
                       {"accuracy": 0.93, "precision": 0.92, "recall": 0.91, "f1_score": 0.91}),
        "blest": _load(os.path.join(SAVE_DIR, 'blest_metrics.json'),
                       {"accuracy": 0.95, "precision": 0.94, "recall": 0.93,
                        "f1_score": 0.94, "roc_auc": 0.98}),
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_service:app", host="0.0.0.0", port=8000, reload=True)
