# Music Maven — FastAPI Wrapper
**Author: Evidence Obojade**
**Milestone 3 — Group 5**

---

## Overview

This FastAPI application wraps the Music Maven lyrics classification and mood-based search pipeline into a REST API with two endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/classify` | POST | Predict genre from lyrics text |
| `/search` | POST | Hybrid mood-based semantic song search |
| `/health` | GET | Health check + status |

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements_api.txt
```

### 2. Add model artifacts
Create a `models/` folder in the same directory as `main.py` and place the following files inside:

**Required for search endpoint:**
- `label_encoder.pkl`
- `svd_model.pkl`
- `lightgbm_fast_model.pkl`
- `linear_svc_model.pkl`
- `lyrics_with_embeddings.pkl`
- `test_emotion.npy`
- `test_valence.npy`
- `test_mood.npy`

**Required for classification endpoint (additionally):**
- `tfidf_vectorizer.pkl`

> Note: These files are not committed to GitHub due to size. Download from the shared Google Drive folder.

### 3. Run the API
```bash
uvicorn main:app --reload --port 8000
```

### 4. Open interactive docs
Visit: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## API Usage

### Classify genre from lyrics
```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"lyrics": "I walk a lonely road the only one that I have ever known", "model": "lightgbm"}'
```

**Response:**
```json
{
  "predicted_genre": "Rock",
  "model_used": "LightGBM",
  "input_length": 62
}
```

### Search songs by mood/theme
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "love and heartbreak", "top_n": 5}'
```

**Response:**
```json
{
  "query": "love and heartbreak",
  "results": [
    {
      "song": "My Love",
      "artist": "Suleiman, Ady",
      "genre": "Indie",
      "top_emotion": "sadness",
      "valence": -0.45,
      "mood": "negative",
      "match_score": 0.571,
      "lyrics_preview": "..."
    }
  ]
}
```

### Optional mood filter
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "feeling good", "top_n": 5, "mood_filter": "positive"}'
```

---

## How it works

### Classification pipeline
1. Clean lyrics (remove tags, lowercase, strip punctuation)
2. TF-IDF vectorisation (10,000 features, ngram 2-3)
3. SBERT embedding (384-dim, all-MiniLM-L6-v2)
4. Style features (TTR, line length, slang count, repetition score)
5. Concatenate → TruncatedSVD (300 components)
6. Predict with LightGBM or Linear SVC

### Search pipeline
1. Embed query with SBERT (semantic similarity)
2. Infer emotion probabilities with DistilBERT
3. Weighted fusion: 50% semantic + 50% emotion similarity
4. Optional mood filter
5. Return top-N ranked results

---

## Notes
- The `RenameUnpickler` class handles numpy version compatibility between Maria's Colab environment and local Python environments
- Classification requires `tfidf_vectorizer.pkl` — search works without it
- Models are loaded once at startup for fast inference
