# Hybrid Mood-Based Music Search Engine
An end-to-end Python pipeline for semantic and emotional music discovery using Transformer-based models.

## Dataset: Multilingual Lyrics Dataset from Kaggle: 
https://www.kaggle.com/datasets/mateibejan/multilingual-lyrics-for-genre-classification 
containing 10 genres, with 290183 songs in Training set and 7935 songs in Test set.

## Tech Stack
* PyTorch (Half-precision inference)
* Sentence-Transformers (all-MiniLM-L6-v2)
* DistilBERT (bhadresh-savani/distilbert-base-uncased-emotion)
* Scikit-Learn (Cosine Similarity)
* Pandas & NumPy

## Overview
This project implements a search engine that retrieves songs based on both semantic meaning and emotional resonance. By combining SBERT embeddings with emotion classification probabilities, the system understands the nuanced "vibe" of a query rather than just keyword matching.

## Key Features
1. Emotion Inference Pipeline:
   - Uses DistilBERT to extract probability vectors for Sadness, Joy, Love, Anger, Fear, and Surprise.
   - Implements custom Valence mapping math to categorize songs into mood labels.
2. Hybrid Ranking Algorithm:
   - Fusion of Semantic Similarity (50%) and Emotional Alignment (50%).
   - Uses dot product and cosine similarity for high-precision ranking.

## Technical Implementation
- Optimization: Uses .half() precision for faster GPU inference.
- Data Handling: Custom RenameUnpickler ensures cross-version NumPy compatibility for serialized data.
- Interface: Includes an interactive CLI loop for real-time testing.
- Embeddings for large train dataset. (Optional)
- FastAPI wrapper. (optional)

## Usage
Run the end-to-end pipeline and search interface with your dataset:
python mood_search_ete.py

## List of files:
Input:
1. train_1.csv (input to lyrics_classification_ete_v1.py pipeline)
2. test_1.csv (input to lyrics_classification_ete_v1.py pipeline)
3. train_clean.pkl (cleaned file as first output file of lyrics_classification_ete_v1.py pipeline)
4. test_clean.pkl (cleaned file as first output file of lyrics_classification_ete_v1.py pipeline)

Output:
1. test_val_mod_merged.pkl
2. lyrics_with_embeddings.pkl
3. test_valence.npy
4. test_mood.npy
5. test_mood_encoded.npy
6. test_emotion.npy