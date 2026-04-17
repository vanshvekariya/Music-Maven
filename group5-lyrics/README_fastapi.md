# Music Maven — FastAPI Wrapper
**Author: Evidence Obojade**
**Milestone 3 — Group 5**

## Overview

This FastAPI application wraps the Music Maven lyrics classification and mood-based search pipeline into a REST API with two endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/classify` | POST | Predict genre from lyrics text |
| `/search` | POST | Hybrid mood-based semantic song search |
| `/health` | GET | Health check + status |

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

**Classification:** clean lyrics → TF-IDF → SBERT → style features → SVD → LightGBM

**Search:** SBERT semantic similarity + DistilBERT emotion similarity, weighted 50/50
