# Music Maven — Group 5: Lyrics Preprocessing & Semantic Search
**Author: Evidence Obojade**
**Student ID: V00917275**
**Milestone 3 — Group 5**

## My Contributions

### Milestone 1 — Data Preprocessing
- Loaded 109,269 lyrics files from Music4All dataset
- Cleaned lyrics: removed structural tags, lowercased, stripped punctuation, removed stopwords
- Extracted TF-IDF keywords (99,242 × 5,000 feature matrix, ngram 1-2)
- Generated SBERT embeddings using all-MiniLM-L6-v2 (99,242 × 384 dimensions, ~14 mins on CPU)
- Merged genre labels from Music4All metadata (99,242/99,242 matched)
- Output files: lyrics_preprocessed.csv, tfidf_matrix.pkl, tfidf_vectorizer.pkl, embeddings.npy

### Milestone 3 — FastAPI Wrapper
Built a REST API wrapping the full classification and search pipeline:

| Endpoint | Method | Description |
|---|---|---|
| `/classify` | POST | Predict genre from lyrics using LightGBM or Linear SVC |
| `/search` | POST | Hybrid mood-based semantic song search |
| `/health` | GET | Health check and status |

## Setup

### 1. Install dependencies
### 2. Add model artifacts
Create a `models/` folder and place the following files inside:
- label_encoder.pkl, svd_model.pkl, lightgbm_fast_model.pkl
- linear_svc_model.pkl, tfidf_vectorizer.pkl
- lyrics_with_embeddings.pkl, test_emotion.npy
- test_valence.npy, test_mood.npy

### 3. Run the API
### 4. Open interactive docs
Visit: http://localhost:8000/docs

## How it works

**Classification:** clean lyrics → TF-IDF (10K features) → SBERT (384-dim) → style features (6) → TruncatedSVD (300 components) → LightGBM prediction

**Search:** SBERT semantic similarity + DistilBERT emotion similarity, weighted 50/50, optional mood filter

## Dataset
Music4All — 109,269 songs, lyrics-only subset
Genre labels sourced from Music4All metadata (id_genres.csv)
