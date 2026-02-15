# Music Maven (G5) - Project Proposal

# Lyrics Classification & Search Module 

# Scope:

We propose to build a lyrics-specific module for the Music Retrieval LLM Chatbot that can:
- Classify lyrics into genres, moods, or thematic categories.
- Enable semantic lyrics search to find songs matching text queries or other lyrics.
- Provide structured outputs in JSON for integration with the LLM.

The system will support basic keyword-based classification, improved features via embeddings, and optionally multi-language support, enabling queries such as:
- “Find happy rap songs.”
- “Songs about heartbreak and hope.”
- “Lyrics similar to this verse.”

# High-Level Design: The system has three main components:

# 1. Lyrics Feature Extraction
- Basic: Extract 5-10 keywords per song using TF-IDF or frequency analysis.
- Expected: Generate vector embeddings (Sentence-BERT) to capture semantic meaning.
- Advanced: Include multi-language embeddings for non-English lyrics (if time permits).

# 2. Lyrics Classification
- Train multiple ML models to classify lyrics:
- Logistic Regression
- SVM
- KNN
- Random Forest

- Evaluate and compare models using accuracy and F1-score.

# 3. Semantic Lyrics Search
- Compute similarity between query text and lyrics embeddings using cosine similarity.
- Return top-k similar songs.
- Support optional filters (genre, mood, explicit content).


# Integration
- Wrap classification & search in a FastAPI endpoint:
    - Accept structured JSON queries
    - Return JSON results compatible with LLM chatbot

JSON query example:
{
  "type": "lyrics_search",
  "query": "happy rap songs",
  "k": 5,
  "include": ["love", "party"],
  "exclude": ["explicit"]
}


# Milestones & Objectives (2-weeks sprints)

# Milestone 1 - Dataset & Preprocessing (Feb 15 - Feb 28)

Goal: Obtain clean lyrics data and extract basic features.
- Load Music4All dataset.
- Clean lyrics (punctuation, stopwords, lowercase).
- Extract 5-10 keywords per song.
- Generate TF-IDF vectors / embeddings for expected features.

Deliverables:
- Preprocessed lyrics dataset in PyNote
- Feature vectors for classification and search

# Milestone 2 - Classification & Model Comparison (Mar 1 - Mar 15)

Goal: Build and evaluate lyrics classification models.
- Train multiple models (Logistic Regression, SVM, KNN, Random Forest).
- Compare performance (accuracy, F1-score) to select best model.
- Prototype baseline semantic search using keywords.

Deliverables:
- Comparison report of models
- Saved best-performing classifier
- Preliminary search results for top-k queries

# Milestone 3 - Semantic Search & API Integration (Mar 16 - Mar 31)

Goal: Enable top-k search with embeddings and API integration.
- Implement semantic search using embeddings.
- Add optional filters (genre, mood, include/exclude).
- Wrap module in FastAPI endpoint.
- Evaluate end-to-end performance (classification + search).

Deliverables:
- Functional module in PyNote
- FastAPI microservice for chatbot queries
- GitHub repository with final code and instructions in README.md

# Team Roles & Detailed Objectives

# Maria Malini Anthony - Classification and Integration

Objective 1: Lyrics Classification
- PI.1 (basic): Extract 5-10 keywords per song.
- PI.2 (expected): Generate TF-IDF / embeddings for each song.
- PI.3 (expected): Train multiple classifiers (Logistic Regression, SVM, KNN/Random Forest).
- PI.4 (expected): Evaluate models (accuracy, F1-score), select best-performing.
- PI.5 (advanced): Multi-label classification, hyperparameter tuning.

Objective 2: Feature Analysis & Integration
- PI.1 (basic): Analyze feature importance, compare keyword features vs embeddings for classification performance.
- PI.2 (expected): Save trained classifier and feature vectors in a structured format for API use (.pkl files).
- PI.3 (expected): Integrate classifier with search component and run preliminary end-to-end tests on sample queries.
- PI.4 (advanced): Test full module (classification + semantic search) on a held-out dataset to verify accuracy and search relevance.

Final: Push final version to GitHub with clear repository structure and instructions for teammate/API use. 

# Evidence Obojade - Search & API Integration 

Objective 1: Semantic Search
- PI.1 (basic): Implement keyword-based top-k search.
- PI.2 (expected): Implement embedding-based semantic search.
- PI.3 (expected): Add top-k retrieval with optional filters (genre, mood).
- PI.4 (advanced): Support multilingual lyrics search.

Objective 2: API Integration
- PI.1 (basic): Set up FastAPI microservice.
- PI.2 (basic): Define JSON query format for structured requests.
- PI.3 (expected): Connect classifier & embedding search to API.
- PI.4 (advanced): Ensure API works with chatbot queries, test end-to-end.

Final: Push final version to GitHub with clear repository structure and instructions for teammate/API use. 

# Tools, Dataset & Libraries

- Dataset: Music4All (primary), optional extra dataset (if time permits)
- Development: PyNote notebooks, GitHub for collaboration
- Libraries: Pandas, NumPy, Scikit-learn, Sentence-Transformers, FastAPI, Matplotlib/Seaborn


# Bibliography
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP 2019.
- Manning, C., Raghavan, P., & Schütze, H. (2008). Introduction to Information Retrieval. Cambridge University Press.
- ISMIR papers on lyrics retrieval and classification:
    - Musical genre classification of audio signals. IEEE Transactions on Speech and Audio Processing, 10(5), 293–302.
    - Relevant recent papers on lyric-based retrieval or embeddings from ISMIR conferences.

