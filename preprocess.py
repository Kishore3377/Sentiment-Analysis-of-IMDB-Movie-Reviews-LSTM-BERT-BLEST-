"""
Module A & B: Dataset Collection and Pre-processing
- Downloads IMDB dataset (50,000 reviews)
- Cleans and tokenizes text
- Splits into train/test sets
"""
print("Hi")
import os

import re
import nltk
import numpy as np
import pandas as pd
from datasets import load_dataset
#from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import re
print("2")

import nltk
print("3")

import numpy as np
print("4")

import pandas as pd
print("5")

from datasets import load_dataset
print("6")

from nltk.corpus import stopwords
print("7")
import tensorflow as tf
print("TensorFlow Loaded")
print(tf.__version__)

from tensorflow.keras.preprocessing.text import Tokenizer
print("8")

print("Hi again")

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# ── Constants ────────────────────────────────────────────────────────────────
MAX_VOCAB   = 20000   # top N words to keep
MAX_LEN     = 200     # max sequence length for LSTM
TEST_SIZE   = 0.2
RANDOM_SEED = 42
SAVE_DIR    = "saved_models"

os.makedirs(SAVE_DIR, exist_ok=True)
print(MAX_VOCAB)


# ── Text Cleaning ─────────────────────────────────────────────────────────────
STOP_WORDS = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    """Remove HTML, punctuation, extra spaces; lowercase; remove stop-words."""
    text = re.sub(r'<.*?>', ' ', text)           # strip HTML tags
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)    # keep letters only
    text = text.lower().strip()
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return ' '.join(tokens)


# ── Load & Prepare IMDB Dataset ───────────────────────────────────────────────
def load_and_preprocess():
    print("📥 Loading IMDB dataset …")
    dataset = load_dataset("imdb")

    train_df = pd.DataFrame(dataset['train'])
    test_df  = pd.DataFrame(dataset['test'])
    print(train_df.head(1))
    df = pd.concat([train_df, test_df], ignore_index=True)

    print(f"   Total reviews : {len(df)}")
    print(f"   Label distribution:\n{df['label'].value_counts()}\n")

    print("🧹 Cleaning text …")
    df['cleaned_text'] = df['text'].apply(clean_text)

    # Train / test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'].values,
        df['label'].values,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df['label'].values
    )
    print(f"   Train size : {len(X_train)} | Test size : {len(X_test)}\n")
    return X_train, X_test, y_train, y_test


# ── Tokenise for LSTM ─────────────────────────────────────────────────────────
def tokenise_for_lstm(X_train, X_test):
    print("🔢 Tokenising for LSTM …")
    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)

    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN, truncating='post')
    X_test_seq  = pad_sequences(tokenizer.texts_to_sequences(X_test),  maxlen=MAX_LEN, truncating='post')

    # Save tokenizer
    with open(os.path.join(SAVE_DIR, 'tokenizer.pkl'), 'wb') as f:
        pickle.dump(tokenizer, f)

    print(f"   Vocab size   : {len(tokenizer.word_index)}")
    print(f"   Sequence len : {MAX_LEN}\n")
    return X_train_seq, X_test_seq, tokenizer


#if __name__ == '__main__':
print("hi")
X_train, X_test, y_train, y_test = load_and_preprocess()
X_train_seq, X_test_seq, tok = tokenise_for_lstm(X_train, X_test)

# Save arrays for re-use
np.save(os.path.join(SAVE_DIR, 'X_train_seq.npy'), X_train_seq)
np.save(os.path.join(SAVE_DIR, 'X_test_seq.npy'),  X_test_seq)
np.save(os.path.join(SAVE_DIR, 'y_train.npy'),     y_train)
np.save(os.path.join(SAVE_DIR, 'y_test.npy'),      y_test)

# Also save raw text for BERT
np.save(os.path.join(SAVE_DIR, 'X_train_raw.npy'), X_train)
np.save(os.path.join(SAVE_DIR, 'X_test_raw.npy'),  X_test)

print("✅ Preprocessing complete. Artifacts saved to saved_models/")
