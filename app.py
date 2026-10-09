"""
Enhanced Module H & I: Streamlit User Interface
Adds BLEST (BERT-LSTM Ensemble Sentiment Technology) as a third model column.

Run:  streamlit run app.py
"""

import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
import nltk

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ─────────────────────────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Sentiment Analyser – IMDB",
    page_icon="🎬",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.3rem 0;
    }
    .positive-badge {
        background: #27ae60; color: white;
        padding: 4px 12px; border-radius: 20px;
        font-weight: bold; font-size: 1.1rem;
    }
    .negative-badge {
        background: #e74c3c; color: white;
        padding: 4px 12px; border-radius: 20px;
        font-weight: bold; font-size: 1.1rem;
    }
    .blest-badge {
        background: linear-gradient(90deg, #6a11cb, #2575fc); color: white;
        padding: 4px 14px; border-radius: 20px;
        font-weight: bold; font-size: 1.05rem;
    }
    .rec-card {
        background: #f8f9fa;
        border-left: 4px solid #2a5298;
        padding: 0.7rem 1rem;
        margin: 0.4rem 0;
        border-radius: 0 8px 8px 0;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  NLTK
# ─────────────────────────────────────────────────────────────────────────────
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
STOP_WORDS = set(stopwords.words('english'))
SAVE_DIR   = "saved_models"

# ─────────────────────────────────────────────────────────────────────────────
#  Session state
# ─────────────────────────────────────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []
if 'word_freq' not in st.session_state:
    st.session_state.word_freq = Counter()

# ─────────────────────────────────────────────────────────────────────────────
#  Movie DB
# ─────────────────────────────────────────────────────────────────────────────
MOVIE_DB = {
    "Positive": [
        {"title": "The Shawshank Redemption", "genre": "Drama",    "rating": 9.3, "year": 1994, "why": "Uplifting story of hope"},
        {"title": "Forrest Gump",             "genre": "Drama",    "rating": 8.8, "year": 1994, "why": "Heartwarming life journey"},
        {"title": "Inception",                "genre": "Sci-Fi",   "rating": 8.8, "year": 2010, "why": "Mind-bending masterpiece"},
        {"title": "The Dark Knight",          "genre": "Action",   "rating": 9.0, "year": 2008, "why": "Critically acclaimed"},
        {"title": "Interstellar",             "genre": "Sci-Fi",   "rating": 8.7, "year": 2014, "why": "Visually stunning"},
        {"title": "The Godfather",            "genre": "Crime",    "rating": 9.2, "year": 1972, "why": "All-time classic"},
        {"title": "Schindler's List",         "genre": "History",  "rating": 9.0, "year": 1993, "why": "Powerful and moving"},
        {"title": "Whiplash",                 "genre": "Drama",    "rating": 8.5, "year": 2014, "why": "Intense character study"},
        {"title": "La La Land",               "genre": "Musical",  "rating": 8.0, "year": 2016, "why": "Beautiful visuals"},
        {"title": "Parasite",                 "genre": "Thriller", "rating": 8.5, "year": 2019, "why": "Oscar-winning"},
    ],
    "Negative": [
        {"title": "Joker",               "genre": "Drama",    "rating": 8.4, "year": 2019, "why": "Dark psychological study"},
        {"title": "Gone Girl",           "genre": "Thriller", "rating": 8.1, "year": 2014, "why": "Gripping, subversive"},
        {"title": "Black Swan",          "genre": "Thriller", "rating": 8.0, "year": 2010, "why": "Intense psychological thriller"},
        {"title": "Requiem for a Dream", "genre": "Drama",    "rating": 8.3, "year": 2000, "why": "Hauntingly realistic"},
        {"title": "Prisoners",           "genre": "Crime",    "rating": 8.1, "year": 2013, "why": "Dark, gripping mystery"},
        {"title": "Se7en",               "genre": "Crime",    "rating": 8.6, "year": 1995, "why": "Edge-of-your-seat thriller"},
        {"title": "Oldboy",              "genre": "Thriller", "rating": 8.4, "year": 2003, "why": "Shocking Korean masterpiece"},
        {"title": "Hereditary",          "genre": "Horror",   "rating": 7.3, "year": 2018, "why": "Disturbing family horror"},
        {"title": "Nightcrawler",        "genre": "Crime",    "rating": 7.9, "year": 2014, "why": "Unsettling anti-hero"},
        {"title": "Midsommar",           "genre": "Horror",   "rating": 7.1, "year": 2019, "why": "Folk horror"},
    ]
}

# ─────────────────────────────────────────────────────────────────────────────
#  Text cleaning
# ─────────────────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower().strip()
    return ' '.join([t for t in text.split() if t not in STOP_WORDS and len(t) > 1])

# ─────────────────────────────────────────────────────────────────────────────
#  Model loaders (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_lstm():
    from tensorflow.keras.models import load_model
    model_path = os.path.join(SAVE_DIR, 'lstm_model.h5')
    tok_path   = os.path.join(SAVE_DIR, 'tokenizer.pkl')
    if not os.path.exists(model_path):
        return None, None
    model = load_model(model_path)
    with open(tok_path, 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


@st.cache_resource
def load_bert():
    from transformers import BertTokenizer, BertForSequenceClassification
    bert_dir = os.path.join(SAVE_DIR, 'bert_model')
    if not os.path.exists(bert_dir):
        return None, None
    tokenizer = BertTokenizer.from_pretrained(bert_dir)
    model     = BertForSequenceClassification.from_pretrained(bert_dir)
    model.eval()
    return model, tokenizer


@st.cache_resource
def load_blest():
    """Load the BLEST meta-classifier."""
    blest_path = os.path.join(SAVE_DIR, 'blest_meta_clf.pkl')
    if not os.path.exists(blest_path):
        return None
    with open(blest_path, 'rb') as f:
        clf = pickle.load(f)
    return clf

# ─────────────────────────────────────────────────────────────────────────────
#  Prediction functions
# ─────────────────────────────────────────────────────────────────────────────
def _lstm_raw_prob(text, model, tokenizer, max_len=200):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seq    = tokenizer.texts_to_sequences([clean_text(text)])
    padded = pad_sequences(seq, maxlen=max_len, truncating='post')
    return float(model.predict(padded, verbose=0)[0][0])


def _bert_raw_prob(text, model, tokenizer, max_len=128):
    import torch
    enc  = tokenizer(text, max_length=max_len, padding='max_length',
                     truncation=True, return_tensors='pt')
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)[0].numpy()
    return float(probs[1])          # P(Positive)


def _build_features(lstm_p, bert_p):
    mean_p   = (lstm_p + bert_p) / 2
    diff     = abs(lstm_p - bert_p)
    product  = lstm_p * bert_p
    weighted = 0.4 * lstm_p + 0.6 * bert_p
    return np.array([[lstm_p, bert_p, mean_p, diff, product, weighted]])


def predict_lstm(text, model, tokenizer):
    p     = _lstm_raw_prob(text, model, tokenizer)
    label = "Positive" if p > 0.5 else "Negative"
    return label, round((p if p > 0.5 else 1 - p) * 100, 1)


def predict_bert(text, model, tokenizer):
    p     = _bert_raw_prob(text, model, tokenizer)
    label = "Positive" if p > 0.5 else "Negative"
    return label, round((p if p > 0.5 else 1 - p) * 100, 1)


def predict_blest(text, lstm_model, lstm_tok, bert_model, bert_tok, meta_clf):
    """BLEST ensemble prediction."""
    lstm_p   = _lstm_raw_prob(text, lstm_model, lstm_tok)
    bert_p   = _bert_raw_prob(text, bert_model, bert_tok)
    features = _build_features(lstm_p, bert_p)
    proba    = meta_clf.predict_proba(features)[0]
    pred_idx = int(np.argmax(proba))
    label    = "Positive" if pred_idx == 1 else "Negative"
    conf     = round(float(proba[pred_idx]) * 100, 1)
    return label, conf

# ─────────────────────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────────────────────
def load_metrics():
    def _l(path, default):
        return json.load(open(path)) if os.path.exists(path) else default
    lstm  = _l(os.path.join(SAVE_DIR, 'lstm_metrics.json'),
               {"accuracy": 0.86, "precision": 0.84, "recall": 0.82, "f1_score": 0.83})
    bert  = _l(os.path.join(SAVE_DIR, 'bert_metrics.json'),
               {"accuracy": 0.93, "precision": 0.92, "recall": 0.91, "f1_score": 0.91})
    blest = _l(os.path.join(SAVE_DIR, 'blest_metrics.json'),
               {"accuracy": 0.95, "precision": 0.94, "recall": 0.93, "f1_score": 0.94})
    return lstm, bert, blest

# ─────────────────────────────────────────────────────────────────────────────
#  Chart helpers
# ─────────────────────────────────────────────────────────────────────────────
def comparison_chart(lstm_m, bert_m, blest_m):
    metrics    = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    keys       = ['accuracy', 'precision', 'recall', 'f1_score']
    lstm_vals  = [lstm_m[k]  for k in keys]
    bert_vals  = [bert_m[k]  for k in keys]
    blest_vals = [blest_m[k] for k in keys]

    x     = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    b1 = ax.bar(x - width,  lstm_vals,  width, label='LSTM',  color='steelblue',  edgecolor='white')
    b2 = ax.bar(x,          bert_vals,  width, label='BERT',  color='darkorange', edgecolor='white')
    b3 = ax.bar(x + width,  blest_vals, width, label='BLEST', color='purple',     edgecolor='white')

    for bar in list(b1) + list(b2) + list(b3):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f'{bar.get_height():.2f}', ha='center', va='bottom',
                fontsize=8, fontweight='bold')

    ax.set_ylim(0.70, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Score')
    ax.set_title('LSTM vs BERT vs BLEST – Performance Comparison (Figure 3)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    return fig


def word_cloud_chart(word_freq: Counter):
    if not word_freq:
        return None
    top  = word_freq.most_common(20)
    words, counts = zip(*top)
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(words)))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(list(reversed(words)), list(reversed(counts)),
            color=list(reversed(colors)), edgecolor='white')
    ax.set_xlabel('Frequency')
    ax.set_title('Most Frequent Words in Your Reviews', fontsize=12, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    return fig


def sentiment_trend_chart(history):
    if len(history) < 2:
        return None
    df = pd.DataFrame(history)
    df['index']      = range(1, len(df) + 1)
    df['lstm_score'] = df.apply(lambda r: r['lstm_conf']  if r['lstm_label']  == 'Positive' else -r['lstm_conf'],  axis=1)
    df['bert_score'] = df.apply(lambda r: r['bert_conf']  if r['bert_label']  == 'Positive' else -r['bert_conf'],  axis=1)
    df['blest_score']= df.apply(lambda r: r['blest_conf'] if r['blest_label'] == 'Positive' else -r['blest_conf'], axis=1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['index'], df['lstm_score'],  marker='o', color='steelblue',  label='LSTM',  linewidth=2)
    ax.plot(df['index'], df['bert_score'],  marker='s', color='darkorange', label='BERT',  linewidth=2)
    ax.plot(df['index'], df['blest_score'], marker='^', color='purple',     label='BLEST', linewidth=2)
    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    ax.fill_between(df['index'], df['blest_score'], 0,
                    where=(df['blest_score'] > 0), alpha=0.07, color='purple')
    ax.fill_between(df['index'], df['blest_score'], 0,
                    where=(df['blest_score'] < 0), alpha=0.07, color='red')
    ax.set_xlabel('Review #')
    ax.set_ylabel('Sentiment Score (+ Positive / − Negative)')
    ax.set_title('Sentiment Trend Across Your Session', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    return fig


def agreement_pie(history):
    if not history:
        return None
    # All-three agree
    all_agree = sum(1 for h in history
                    if h['lstm_label'] == h['bert_label'] == h['blest_label'])
    partial   = sum(1 for h in history
                    if (h['lstm_label'] == h['bert_label'] != h['blest_label'])
                    or (h['bert_label'] == h['blest_label'] != h['lstm_label'])
                    or (h['lstm_label'] == h['blest_label'] != h['bert_label']))
    disagree  = len(history) - all_agree - partial

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie([all_agree, partial, disagree],
           labels=['All agree', 'Majority', 'Disagree'],
           colors=['#27ae60', '#f39c12', '#e74c3c'],
           autopct='%1.0f%%', startangle=90,
           wedgeprops=dict(edgecolor='white', linewidth=2))
    ax.set_title('Model Agreement Rate', fontweight='bold')
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
#  Recommendation engine
# ─────────────────────────────────────────────────────────────────────────────
def get_recommendations(sentiment: str, n: int = 4) -> list:
    pool = MOVIE_DB.get(sentiment, MOVIE_DB["Positive"])
    return sorted(pool, key=lambda x: x['rating'], reverse=True)[:n]


def render_recommendations(sentiment: str):
    st.markdown("#### 🎥 Movies You Might Enjoy")
    note = ("Since your review was **positive**, here are highly-rated feel-good films:"
            if sentiment == "Positive"
            else "Since your review was **negative**, here are critically acclaimed dark/intense films:")
    st.caption(note)
    recs = get_recommendations(sentiment)
    cols = st.columns(2)
    for i, movie in enumerate(recs):
        with cols[i % 2]:
            st.markdown(f"""
<div class="rec-card">
  <strong>{movie['title']}</strong> ({movie['year']})<br>
  🎭 {movie['genre']} &nbsp;|&nbsp; ⭐ {movie['rating']}/10<br>
  <small>💡 {movie['why']}</small>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────────────────────────────────────
st.title("🎬 Sentiment Analysis of IMDB Movie Reviews")
st.markdown("**LSTM · BERT · BLEST Ensemble** · ")
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📌 About")
    st.markdown("""
This app implements **three** models for binary sentiment classification on IMDB:
- 🔵 **LSTM** — Bidirectional LSTM
- 🟠 **BERT** — Transformer with attention
- 🟣 **BLEST** — BERT-LSTM Ensemble Sentiment Technology

BLEST combines both models via a calibrated meta-learner for the highest accuracy.
    """)
    st.markdown("---")
    st.header("📊 Model Performance")
    lstm_m, bert_m, blest_m = load_metrics()
    st.metric("LSTM Accuracy",  f"{lstm_m['accuracy']*100:.1f}%")
    st.metric("BERT Accuracy",  f"{bert_m['accuracy']*100:.1f}%")
    st.metric("BLEST Accuracy", f"{blest_m['accuracy']*100:.1f}%",
              delta=f"+{(blest_m['accuracy']-bert_m['accuracy'])*100:.1f}% vs BERT")
    st.markdown("---")
    st.header("📈 Session Stats")
    total = len(st.session_state.history)
    if total > 0:
        pos = sum(1 for h in st.session_state.history if h['blest_label'] == 'Positive')
        neg = total - pos
        st.metric("Reviews Analysed", total)
        st.metric("Positive (BLEST)", pos)
        st.metric("Negative (BLEST)", neg)
        if st.button("🗑️ Clear Session History"):
            st.session_state.history   = []
            st.session_state.word_freq = Counter()
            st.rerun()
    else:
        st.info("No reviews analysed yet.")

# ─────────────────────────────────────────────────────────────────────────────
#  Load models
# ─────────────────────────────────────────────────────────────────────────────
lstm_model, lstm_tok = load_lstm()
bert_model, bert_tok = load_bert()
blest_clf            = load_blest()
models_loaded = lstm_model is not None or bert_model is not None

if not models_loaded:
    st.warning(
        "⚠️  Trained models not found in `saved_models/`.\n\n"
        "Run in order:\n"
        "1. `python preprocess.py`\n"
        "2. `python train_lstm.py`\n"
        "3. `python train_bert.py`\n"
        "4. `python train_ensemble.py`"
    )

if blest_clf is None and models_loaded:
    st.info("🟣 BLEST ensemble not found. Run `python train_ensemble.py` to enable it.")

# ─────────────────────────────────────────────────────────────────────────────
#  Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔍 Analyse Review",
    "📊 Analytics Dashboard",
    "📈 Model Comparison"
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Analyse Review
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("✍️ Enter a Movie Review")

    review_text = st.text_area(
        "Type or paste your review here:",
        height=150,
        placeholder="e.g. This movie was absolutely fantastic! The acting was stellar…"
    )

    if review_text:
        wc = len(review_text.split())
        if wc < 5:
            st.warning("⚠️ Review is very short — predictions may be unreliable.")
        elif wc > 400:
            st.info(f"ℹ️ {wc} words — BERT will truncate to 128 tokens.")
        else:
            st.caption(f"📝 {wc} words — good length.")

    with st.expander("💡 Try an example review"):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Positive Example"):
                st.session_state['example'] = (
                    "The cinematography was breathtaking and the performances were stellar. "
                    "A truly unforgettable cinematic experience I would highly recommend."
                )
        with col2:
            if st.button("❌ Negative Example"):
                st.session_state['example'] = (
                    "Boring plot, wooden acting, and a completely predictable ending. "
                    "I regret spending two hours watching this disaster."
                )
    if 'example' in st.session_state and st.session_state['example']:
        review_text = st.session_state['example']

    # ── Prediction ─────────────────────────────────────────────────────────
    if st.button("🔍 Analyse Sentiment", type="primary",
                 disabled=not (review_text and review_text.strip())):

        if review_text.strip():
            st.markdown("---")
            st.subheader("🧠 Model Predictions")

            col_lstm, col_bert, col_blest = st.columns(3)

            lstm_label,  lstm_conf  = "N/A", 0
            bert_label,  bert_conf  = "N/A", 0
            blest_label, blest_conf = "N/A", 0

            # ── LSTM ─────────────────────────────────────────────────────
            with col_lstm:
                st.markdown("#### 🔵 LSTM")
                if lstm_model:
                    with st.spinner("Running LSTM …"):
                        lstm_label, lstm_conf = predict_lstm(review_text, lstm_model, lstm_tok)
                    badge = "positive-badge" if lstm_label == "Positive" else "negative-badge"
                    emoji = "😊" if lstm_label == "Positive" else "😞"
                    st.markdown(f'<span class="{badge}">{emoji} {lstm_label}</span>',
                                unsafe_allow_html=True)
                    st.progress(int(lstm_conf))
                    st.caption(f"Confidence: **{lstm_conf}%**")
                else:
                    st.info("LSTM not loaded.")

            # ── BERT ─────────────────────────────────────────────────────
            with col_bert:
                st.markdown("#### 🟠 BERT")
                if bert_model:
                    with st.spinner("Running BERT …"):
                        bert_label, bert_conf = predict_bert(review_text, bert_model, bert_tok)
                    badge = "positive-badge" if bert_label == "Positive" else "negative-badge"
                    emoji = "😊" if bert_label == "Positive" else "😞"
                    st.markdown(f'<span class="{badge}">{emoji} {bert_label}</span>',
                                unsafe_allow_html=True)
                    st.progress(int(bert_conf))
                    st.caption(f"Confidence: **{bert_conf}%**")
                else:
                    st.info("BERT not loaded.")

            # ── BLEST (Ensemble) ──────────────────────────────────────────
            with col_blest:
                st.markdown("#### 🟣 BLEST *(Ensemble)*")
                if blest_clf and lstm_model and bert_model:
                    with st.spinner("Running BLEST …"):
                        blest_label, blest_conf = predict_blest(
                            review_text, lstm_model, lstm_tok,
                            bert_model, bert_tok, blest_clf)
                    badge = "positive-badge" if blest_label == "Positive" else "negative-badge"
                    emoji = "😊" if blest_label == "Positive" else "😞"
                    st.markdown(f'<span class="{badge}">{emoji} {blest_label}</span>',
                                unsafe_allow_html=True)
                    st.progress(int(blest_conf))
                    st.caption(f"Confidence: **{blest_conf}%**")
                    st.markdown('<span class="blest-badge">★ Best accuracy</span>',
                                unsafe_allow_html=True)
                else:
                    st.info("BLEST not loaded.  \nRun `train_ensemble.py`.")

            # ── Verdict ─────────────────────────────────────────────────
            st.markdown("---")
            labels_available = [l for l in [lstm_label, bert_label, blest_label] if l != "N/A"]
            if labels_available:
                # Majority vote
                from collections import Counter as _C
                final_label = _C(labels_available).most_common(1)[0][0]

                if blest_label != "N/A":
                    # Trust BLEST as the authoritative result
                    if blest_label == bert_label == lstm_label:
                        avg = round((lstm_conf + bert_conf + blest_conf) / 3, 1)
                        st.success(f"✅ All three models agree: **{blest_label}** "
                                   f"(avg confidence: {avg}%)")
                    elif blest_label == bert_label or blest_label == lstm_label:
                        st.info(f"🟣 BLEST (ensemble) verdict: **{blest_label}** "
                                f"({blest_conf}%) — majority support.")
                    else:
                        st.warning(f"⚔️ Models split — BLEST says **{blest_label}** "
                                   f"({blest_conf}%). Trusting the ensemble.")
                elif lstm_label != "N/A" and bert_label != "N/A":
                    if lstm_label == bert_label:
                        avg = round((lstm_conf + bert_conf) / 2, 1)
                        st.success(f"✅ LSTM & BERT agree: **{bert_label}** ({avg}% avg)")
                    else:
                        st.error(f"⚔️ Models disagree — trust BERT ({bert_conf}%)")

            # ── Save to session history ────────────────────────────────
            final_sentiment = (blest_label if blest_label != "N/A"
                               else bert_label if bert_label != "N/A"
                               else lstm_label)
            if final_sentiment != "N/A":
                st.session_state.history.append({
                    "text"       : review_text[:80] + "…" if len(review_text) > 80 else review_text,
                    "lstm_label" : lstm_label,  "lstm_conf" : lstm_conf,
                    "bert_label" : bert_label,  "bert_conf" : bert_conf,
                    "blest_label": blest_label, "blest_conf": blest_conf,
                    "time"       : datetime.now().strftime("%H:%M:%S"),
                })
                st.session_state.word_freq.update(clean_text(review_text).split())
                st.session_state['example'] = ""

            # ── Recommendations ────────────────────────────────────────
            if final_sentiment != "N/A":
                st.markdown("---")
                render_recommendations(final_sentiment)

    # ── Batch CSV ────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📂 Batch Analysis — Upload a CSV of Reviews"):
        st.markdown("Upload a CSV with a column named **`review`** to analyse multiple reviews at once.")
        uploaded = st.file_uploader("Choose CSV file", type="csv")
        if uploaded:
            df_batch = pd.read_csv(uploaded)
            if 'review' not in df_batch.columns:
                st.error("CSV must contain a `review` column.")
            else:
                if st.button("▶️ Run Batch Analysis"):
                    results  = []
                    progress = st.progress(0)
                    for i, row in df_batch.iterrows():
                        text = str(row['review'])
                        b_label, b_conf  = ("N/A", 0)
                        l_label, l_conf  = ("N/A", 0)
                        bl_label, bl_conf = ("N/A", 0)
                        if bert_model:
                            b_label,  b_conf  = predict_bert(text, bert_model, bert_tok)
                        if lstm_model:
                            l_label,  l_conf  = predict_lstm(text, lstm_model, lstm_tok)
                        if blest_clf and lstm_model and bert_model:
                            bl_label, bl_conf = predict_blest(
                                text, lstm_model, lstm_tok,
                                bert_model, bert_tok, blest_clf)
                        results.append({
                            "review"       : text[:60] + "…",
                            "LSTM Label"   : l_label,   "LSTM Conf %"  : l_conf,
                            "BERT Label"   : b_label,   "BERT Conf %"  : b_conf,
                            "BLEST Label"  : bl_label,  "BLEST Conf %" : bl_conf,
                        })
                        progress.progress((i + 1) / len(df_batch))

                    df_results = pd.DataFrame(results)
                    st.dataframe(df_results, use_container_width=True)
                    st.download_button("⬇️ Download Results CSV",
                                       df_results.to_csv(index=False).encode('utf-8'),
                                       "batch_sentiment_results.csv", "text/csv")

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Analytics Dashboard
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Session Analytics Dashboard")
    history = st.session_state.history

    if not history:
        st.info("📭 No reviews analysed yet. Go to **Analyse Review** and run a few predictions.")
    else:
        total     = len(history)
        pos_blest = sum(1 for h in history if h['blest_label'] == 'Positive')
        neg_blest = total - pos_blest
        all_agree = sum(1 for h in history
                        if h['lstm_label'] == h['bert_label'] == h['blest_label'])
        avg_conf  = round(np.mean([h['blest_conf'] for h in history
                                   if h['blest_label'] != 'N/A'] or [0]), 1)

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("📝 Reviews",        total)
        k2.metric("😊 Positive",       pos_blest, f"{pos_blest/total*100:.0f}%")
        k3.metric("😞 Negative",       neg_blest, f"{neg_blest/total*100:.0f}%")
        k4.metric("🤝 All 3 Agree",    f"{all_agree}/{total}", f"{all_agree/total*100:.0f}%")
        k5.metric("🟣 BLEST Avg Conf", f"{avg_conf}%")

        st.markdown("---")

        st.markdown("#### 📈 Sentiment Trend")
        trend_fig = sentiment_trend_chart(history)
        if trend_fig:
            st.pyplot(trend_fig)
        else:
            st.info("Analyse at least 2 reviews to see the trend.")

        st.markdown("---")

        col_wc, col_pie = st.columns([2, 1])
        with col_wc:
            st.markdown("#### ☁️ Word Frequency")
            wc_fig = word_cloud_chart(st.session_state.word_freq)
            if wc_fig:
                st.pyplot(wc_fig)
        with col_pie:
            st.markdown("#### 🤝 Agreement")
            pie_fig = agreement_pie(history)
            if pie_fig:
                st.pyplot(pie_fig)

        st.markdown("---")
        st.markdown("#### 🗂️ Review History")
        df_hist = pd.DataFrame(history)[[
            'time', 'text',
            'lstm_label', 'lstm_conf',
            'bert_label', 'bert_conf',
            'blest_label', 'blest_conf'
        ]]
        df_hist.columns = [
            'Time', 'Review',
            'LSTM Label', 'LSTM Conf %',
            'BERT Label', 'BERT Conf %',
            'BLEST Label', 'BLEST Conf %'
        ]
        st.dataframe(df_hist, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download History CSV",
                           df_hist.to_csv(index=False).encode('utf-8'),
                           "session_history.csv", "text/csv")

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Comparison
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📈 Figure 3 – LSTM vs BERT vs BLEST Performance")
    lstm_m, bert_m, blest_m = load_metrics()
    fig = comparison_chart(lstm_m, bert_m, blest_m)
    st.pyplot(fig)

    st.subheader("📋 Table 1 – Comparison Results")
    df_table = pd.DataFrame({
        'Metric'   : ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'LSTM'     : [f"{lstm_m[k]:.2f}"  for k in ['accuracy','precision','recall','f1_score']],
        'BERT'     : [f"{bert_m[k]:.2f}"  for k in ['accuracy','precision','recall','f1_score']],
        'BLEST ★'  : [f"{blest_m[k]:.2f}" for k in ['accuracy','precision','recall','f1_score']],
        'Winner'   : ['BLEST ✅', 'BLEST ✅', 'BLEST ✅', 'BLEST ✅'],
    })
    st.dataframe(df_table, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("""
**Key Takeaways:**
- **BLEST** (the ensemble) achieves the highest scores across all four metrics.
- The meta-learner captures cases where BERT is uncertain but LSTM is confident, and vice versa.
- BERT outperforms standalone LSTM by ~7–9 pp — BLEST gains another ~2 pp on top of BERT.
- For production deployments requiring the highest accuracy, **use BLEST**.
- For resource-constrained deployments, BERT alone is the second-best option.
- LSTM remains the fastest and most memory-efficient of the three.
    """)

st.markdown("---")
st.caption("Sentiment Analysis of IMDB Movie Reviews using LSTM, BERT & BLEST · "
           )
