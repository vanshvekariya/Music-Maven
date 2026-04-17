"""
Music Maven - Group 5
FastAPI Wrapper: Lyrics Genre Classification + Hybrid Mood-Based Search
Author: Evidence Obojade
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
# Update this path to wherever you store Maria's artifact files
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "models")

EMBED_MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
EMOTION_MODEL_NAME = "bhadresh-savani/distilbert-base-uncased-emotion"
TOP_N = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─── NUMPY COMPATIBILITY HELPER ───────────────────────────────────────────────
# Required due to numpy version differences between Maria's Colab env and local
class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "numpy._core.numeric":
            module = "numpy.core.numeric"
        elif module == "numpy._core":
            module = "numpy._core"
        return super().find_class(module, name)

def safe_load(path: str):
    with open(path, "rb") as f:
        return RenameUnpickler(f).load()

# ─── LOAD ARTIFACTS ───────────────────────────────────────────────────────────
print("Loading artifacts...")

label_encoder   = safe_load(os.path.join(ARTIFACTS_DIR, "label_encoder.pkl"))
svd_model       = safe_load(os.path.join(ARTIFACTS_DIR, "svd_model.pkl"))
lgbm_model      = safe_load(os.path.join(ARTIFACTS_DIR, "lightgbm_fast_model.pkl"))
linear_svc      = safe_load(os.path.join(ARTIFACTS_DIR, "linear_svc_model.pkl"))
df_search       = safe_load(os.path.join(ARTIFACTS_DIR, "lyrics_with_embeddings.pkl"))

# TF-IDF vectorizer — optional, needed for classification endpoint
tfidf_path = os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.pkl")
if os.path.exists(tfidf_path):
    tfidf_vectorizer = safe_load(tfidf_path)
    print("TF-IDF vectorizer loaded.")
else:
    tfidf_vectorizer = None
    print("WARNING: tfidf_vectorizer.pkl not found. Classification endpoint will be unavailable.")

# Pre-computed emotion & search matrices
test_emotion    = np.load(os.path.join(ARTIFACTS_DIR, "test_emotion.npy"))
test_valence    = np.load(os.path.join(ARTIFACTS_DIR, "test_valence.npy"))
test_mood       = np.load(os.path.join(ARTIFACTS_DIR, "test_mood.npy"), allow_pickle=True)
lyrics_embs_matrix = np.array(df_search["lyrics_embedding"].tolist())

print(f"Search database loaded: {len(df_search)} songs")
print(f"Embedding matrix shape: {lyrics_embs_matrix.shape}")

# ─── LOAD MODELS ──────────────────────────────────────────────────────────────
print("Loading SBERT model...")
sbert_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
sbert_model     = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(device).eval()

print("Loading emotion model...")
emotion_tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_NAME)
emotion_model     = AutoModelForSequenceClassification.from_pretrained(
    EMOTION_MODEL_NAME).to(device).eval()

EMOTION_COLS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
print("All models loaded successfully.")

# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────
def embed_text(text: str) -> np.ndarray:
    """Embed a single text string using SBERT."""
    with torch.no_grad():
        enc = sbert_tokenizer(
            [text], padding=True, truncation=True,
            return_tensors="pt", max_length=512
        ).to(device)
        outputs = sbert_model(**enc)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

def get_emotion_probs(text: str) -> np.ndarray:
    """Get 6-class emotion probability vector for a text."""
    with torch.no_grad():
        enc = emotion_tokenizer(
            [text], return_tensors="pt",
            padding=True, truncation=True, max_length=512
        ).to(device)
        out = emotion_model(**enc)
        return torch.softmax(out.logits, dim=1).cpu().numpy()

def clean_text(text: str) -> str:
    """Basic text cleaning — mirrors Maria's preprocessing."""
    import re
    text = re.sub(r'\[.*?\]', '', text)   # remove [Verse]/[Chorus] tags
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip()

# ─── FASTAPI APP ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Music Maven API",
    description="Lyrics genre classification and mood-based semantic search — Group 5",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── REQUEST / RESPONSE MODELS ────────────────────────────────────────────────
class LyricsRequest(BaseModel):
    lyrics: str
    model: Optional[str] = "lightgbm"  # "lightgbm" or "linear_svc"

class ClassificationResult(BaseModel):
    predicted_genre: str
    model_used: str
    input_length: int

class SearchRequest(BaseModel):
    query: str
    top_n: Optional[int] = 5
    mood_filter: Optional[str] = None  # e.g. "positive", "negative", "neutral"

class SongResult(BaseModel):
    song: str
    artist: str
    genre: str
    top_emotion: str
    valence: float
    mood: str
    match_score: float
    lyrics_preview: str

class SearchResponse(BaseModel):
    query: str
    results: List[SongResult]

# ─── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "Music Maven API is running",
        "endpoints": {
            "POST /classify": "Predict genre from lyrics",
            "POST /search":   "Hybrid mood-based song search",
            "GET  /health":   "Health check"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(device),
        "songs_in_database": len(df_search),
        "tfidf_available": tfidf_vectorizer is not None
    }

@app.post("/classify", response_model=ClassificationResult)
def classify_genre(request: LyricsRequest):
    """
    Predict the genre of a song from its lyrics.
    Requires tfidf_vectorizer.pkl to be present in models/.
    Pipeline: clean → TF-IDF → SBERT → style features → concat → SVD → predict
    """
    if tfidf_vectorizer is None:
        raise HTTPException(
            status_code=503,
            detail="tfidf_vectorizer.pkl not found. Classification unavailable."
        )

    cleaned = clean_text(request.lyrics)

    # 1. TF-IDF features
    tfidf_vec = tfidf_vectorizer.transform([cleaned])

    # 2. SBERT embedding
    sbert_vec = embed_text(cleaned)  # shape (1, 384)

    # 3. Style features
    import re
    words = cleaned.split()
    lines = [l.strip() for l in cleaned.split("\n") if l.strip()]
    slang_list = ['ya', 'yo', 'nah', 'shit', 'bitch', 'nigga', 'fuck', 'yeah']

    avg_line_len      = np.mean([len(l.split()) for l in lines]) if lines else 0
    ttr               = len(set(words)) / (len(words) + 1e-5)
    slang_count       = sum(words.count(s) for s in slang_list)
    word_counts       = {}
    for w in words:
        word_counts[w] = word_counts.get(w, 0) + 1
    repetition_score  = np.mean(list(word_counts.values())) if word_counts else 0
    lyrics_length     = len(words)
    unique_lines      = len(set(lines))
    repeated_line_ratio = 1 - (unique_lines / (len(lines) + 1e-5)) if lines else 0

    style_vec = np.array([[
        ttr, avg_line_len, slang_count,
        repetition_score, lyrics_length, repeated_line_ratio
    ]])

    # 4. Combine: TF-IDF (sparse) + SBERT (384) + style (6) → SVD (300)
    import scipy.sparse
    tfidf_dense = tfidf_vec  # keep sparse for hstack
    sbert_sparse = scipy.sparse.csr_matrix(sbert_vec)
    style_sparse = scipy.sparse.csr_matrix(style_vec)
    combined = scipy.sparse.hstack([tfidf_dense, sbert_sparse, style_sparse])

    # 5. Dimensionality reduction
    reduced = svd_model.transform(combined)

    # 6. Predict
    if request.model == "linear_svc":
        pred = linear_svc.predict(reduced)
        model_used = "Linear SVC"
    else:
        pred = lgbm_model.predict(reduced)
        model_used = "LightGBM"

    genre = label_encoder.inverse_transform(pred)[0]

    return ClassificationResult(
        predicted_genre=genre,
        model_used=model_used,
        input_length=len(request.lyrics)
    )


@app.post("/search", response_model=SearchResponse)
def search_songs(request: SearchRequest):
    """
    Hybrid mood-based semantic search.
    Combines SBERT semantic similarity + DistilBERT emotion similarity.
    Optionally filter by mood: very_positive, positive, neutral, negative, very_negative
    """
    top_n = min(request.top_n, 20)  # cap at 20

    # 1. Semantic embedding of query
    query_vec = embed_text(request.query)
    sim_semantic = cosine_similarity(query_vec, lyrics_embs_matrix)[0]

    # 2. Emotion vector of query
    query_emotion = get_emotion_probs(request.query)
    song_emotions = df_search[EMOTION_COLS].values
    sim_emotion = (query_emotion @ song_emotions.T)[0]

    # 3. Weighted fusion (50/50 semantic + emotion)
    final_score = 0.5 * sim_semantic + 0.5 * sim_emotion

    # 4. Optional mood filter
    if request.mood_filter:
        mood_mask = np.array([
            request.mood_filter.lower() in m.lower() for m in test_mood
        ])
        final_score = np.where(mood_mask, final_score, -1)

    # 5. Rank
    top_idx = final_score.argsort()[::-1][:top_n]
    results_df = df_search.iloc[top_idx].copy()
    scores = final_score[top_idx]

    results = []
    for i, (_, row) in enumerate(results_df.iterrows()):
        results.append(SongResult(
            song=str(row.get("Song", "Unknown")),
            artist=str(row.get("Artist", "Unknown")),
            genre=str(row.get("Genre", "Unknown")),
            top_emotion=str(row.get("top_emotion", "unknown")),
            valence=float(row.get("valence", 0.0)),
            mood=str(test_mood[df_search.index.get_loc(row.name)]
                     if row.name in df_search.index else "unknown"),
            match_score=float(scores[i]),
            lyrics_preview=str(row.get("lyrics_clean", ""))[:200] + "..."
        ))

    return SearchResponse(query=request.query, results=results)
