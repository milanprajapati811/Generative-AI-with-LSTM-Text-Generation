
import os
import re
import pickle
import urllib.request
import numpy as np


DATASET_URL  = "https://www.gutenberg.org/files/100/100-0.txt"
DATASET_PATH = "data/shakespeare.txt"
VOCAB_PATH   = "models/vocab.pkl"



def download_dataset(url: str = DATASET_URL,
                     save_path: str = DATASET_PATH) -> None:
   
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not os.path.exists(save_path):
        print(f"[Preprocess] Downloading dataset from:\n  {url}")
        urllib.request.urlretrieve(url, save_path)
        print(f"[Preprocess] Saved to: {save_path}")
    else:
        print(f"[Preprocess] Dataset already exists: {save_path}")


def load_text(path: str = DATASET_PATH) -> str:
   
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    print(f"[Preprocess] Characters loaded: {len(text):,}")
    return text



def preprocess_text(raw_text: str, max_chars: int = 500_000) -> str:
    
    
    text = raw_text[:max_chars]

    
    text = text.lower()

    
    text = re.sub(r"[^a-z0-9\s]", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    print(f"[Preprocess] Characters after cleaning: {len(text):,}")
    return text




def build_vocabulary(text: str):
    
    words    = text.split()                           
    vocab    = sorted(set(words))                    
    word2idx = {w: i for i, w in enumerate(vocab)}   
    idx2word = {i: w for w, i in word2idx.items()}   

    print(f"[Preprocess] Total tokens  : {len(words):,}")
    print(f"[Preprocess] Unique tokens : {len(vocab):,}")
    return words, vocab, word2idx, idx2word


def save_vocabulary(word2idx: dict, idx2word: dict,
                    path: str = VOCAB_PATH) -> None:
   
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"word2idx": word2idx, "idx2word": idx2word}, f)
    print(f"[Preprocess] Vocabulary saved to: {path}")


def load_vocabulary(path: str = VOCAB_PATH):
   
    with open(path, "rb") as f:
        vocab = pickle.load(f)
    print(f"[Preprocess] Vocabulary loaded from: {path}")
    return vocab["word2idx"], vocab["idx2word"]




def create_sequences(words: list,
                     word2idx: dict,
                     seq_length: int = 30):
   

    indices = [word2idx[w] for w in words]

    X, y = [], []
    for i in range(len(indices) - seq_length):
        X.append(indices[i : i + seq_length])  
        y.append(indices[i + seq_length])       

    X = np.array(X, dtype=np.int32)
    y = np.array(y, dtype=np.int32)

    print(f"[Preprocess] Sequences created : {len(X):,}")
    print(f"[Preprocess] X shape: {X.shape}  |  y shape: {y.shape}")
    return X, y




def run_preprocessing(seq_length: int = 30,
                      max_chars: int = 500_000):
    
    download_dataset()
    raw_text   = load_text()
    clean_text = preprocess_text(raw_text, max_chars=max_chars)

    words, vocab, word2idx, idx2word = build_vocabulary(clean_text)
    save_vocabulary(word2idx, idx2word)

    X, y = create_sequences(words, word2idx, seq_length=seq_length)
    vocab_size = len(vocab)

    return X, y, word2idx, idx2word, vocab_size
