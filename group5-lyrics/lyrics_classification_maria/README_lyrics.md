# Multi-Modal Lyrics Genre Classification
An advanced classification framework leveraging statistical, semantic, and linguistic features to categorize music genres.

## Dataset: Multilingual Lyrics Dataset from Kaggle: 
https://www.kaggle.com/datasets/mateibejan/multilingual-lyrics-for-genre-classification 
containing 10 genres, with 290183 songs in Training set and 7935 songs in Test set.

## Feature Stacking Strategy
This project achieves high accuracy by concatenating three distinct feature sets:
1. Statistical (TF-IDF): Captures keyword importance and n-gram patterns (bi-grams and tri-grams).
2. Semantic (SBERT): Transformer-based embeddings capturing deep contextual meaning.
3. Linguistic (Style): Custom metrics including Type-Token Ratio (TTR), Slang density, and Word Repetition scores.

## Workflow Pipeline
1. Data Engineering:
   - Aggressive cleaning using Regex.
   - Genre Merging strategy (e.g., merging Country/Folk into 'Acoustic') to handle sparsity.
2. Dimensionality Reduction:
   - Utilizes Truncated SVD to compress the high-dimensional stacked feature matrix into 300 latent components.
   - Optimizes training speed and prevents overfitting.
3. Model Benchmarking:
   - Linear SVC: High-speed training for sparse text data.
   - LightGBM: GPU-accelerated gradient boosting for non-linear patterns.
   - SGD Classifier: Scalable linear modeling.

## Key Findings
The project demonstrates the trade-off between speed and precision. While Linear SVC provides rapid inference, LightGBM with GPU support captures subtle linguistic nuances that traditional models miss.
  
## Usage
Run the end-to-end pipeline and search interface with your dataset:
python lyrics_classification_ete_v1.py
  
## List of files:
Input:
1. train_1.csv
2. test_1.csv

Output:
1. train_clean.pkl
2. test_clean.pkl
3. train_clean_merged.pkl
4. test_clean_merged.pkl
5. y_train.npy
6. y_test.npy
7. label_encoder.pkl
8. tfidf_vectorizer.pkl
9. X_train_tfidf.npz
10. X_test_tfidf.npz
11. sbert_train.npy
12. sbert_test.npy
13. train_style_features.npy
14. test_style_features.npy
15. X_train_combined.npz
16. X_test_combined.npz
17. X_train_reduced.npy
18. X_test_reduced.npy
19. svd_model.pkl
20. linear_svc_model.pkl
21. sgd_linear_svc_model.pkl
22. lightgbm_fast_model.pkl