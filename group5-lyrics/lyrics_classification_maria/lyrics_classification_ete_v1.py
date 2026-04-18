
# MM G5_Lyrics Classification End To End, dataset used is added to data folder.

import pandas as pd
import numpy as np
import torch
import pickle
import scipy.sparse
import re
import time # Import time module
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import lightgbm as lgb

print("All necessary libraries imported.")



PROJECT_PATH = "./data/" # path of your datafiles

# Ensure the PROJECT_PATH exists
os.makedirs(PROJECT_PATH, exist_ok=True)

print(f"Project path set to: {PROJECT_PATH}")



# File paths
train_path = PROJECT_PATH + "train_1.csv"
test_path = PROJECT_PATH + "test_1.csv"

# Load datasets
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

train_df.head()

# Function to clean lyrics dataset


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)   # remove punctuation
    text = re.sub(r'\s+', ' ', text)          # collapse spaces
    return text.strip()

# Create copies so originals stay untouched
train_clean_df = train_df.copy()
test_clean_df = test_df.copy()

# Apply cleaning
train_clean_df["lyrics_clean"] = train_clean_df["Lyrics"].apply(clean_text)
test_clean_df["lyrics_clean"] = test_clean_df["Lyrics"].apply(clean_text)

# Save cleaned versions
train_clean_df.to_pickle(PROJECT_PATH + "train_clean.pkl")
test_clean_df.to_pickle(PROJECT_PATH + "test_clean.pkl")

print("Cleaning complete and files saved.")
print("Columns", train_clean_df.columns.tolist())
print("Columns", test_clean_df.columns.tolist())


## Step 1: Load and Clean Data

# Load cleaned datasets
print("Loading cleaned datasets...")
train_df = pd.read_pickle(PROJECT_PATH + "train_clean.pkl")
test_df = pd.read_pickle(PROJECT_PATH + "test_clean.pkl")

print("Datasets loaded successfully.")
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

## Step 2: Merge Genres

genre_map = {
    "Country": "Acoustic",
    "Folk": "Acoustic",
    "Indie": "Acoustic",
    "Rock": "Rock",
    "Metal": "Rock",
    "Pop": "Pop",
    "R&B": "Pop",
    "Electronic": "Pop",
    "Hip-Hop": "Hip-Hop",
    "Jazz": "Jazz"
}

train_df["Genre_merged"] = train_df["Genre"].map(genre_map)
test_df["Genre_merged"] = test_df["Genre"].map(genre_map)

# Save merged datasets
train_df.to_pickle(PROJECT_PATH + "train_clean_merged.pkl")
test_df.to_pickle(PROJECT_PATH + "test_clean_merged.pkl")

print("Merged genre datasets saved!")
print("Train genre distribution after merge:")
print(train_df["Genre_merged"].value_counts())

## Step 3: Prepare and Encode Labels

# Load merged datasets if not already in memory
try:
    if 'train_df' not in locals() or 'test_df' not in locals() or train_df.empty:
        train_df = pd.read_pickle(PROJECT_PATH + "train_clean_merged.pkl")
        test_df = pd.read_pickle(PROJECT_PATH + "test_clean_merged.pkl")
except FileNotFoundError:
    print("Merged dataframes not found. Please ensure Step 2 was executed.")
    exit()

# Create label encoder
label_encoder = LabelEncoder()

# Fit encoder on train labels
y_train = label_encoder.fit_transform(train_df["Genre_merged"])
y_test = label_encoder.transform(test_df["Genre_merged"])

print("Label encoding completed.")
print("Classes:", label_encoder.classes_)

# Save training labels
np.save(PROJECT_PATH + "y_train.npy", y_train)
np.save(PROJECT_PATH + "y_test.npy", y_test)
print("Labels saved!")

# Save the encoder for future use
with open(PROJECT_PATH + "label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
    print("Label encoder saved!")

## Step 4: TF-IDF Feature Extraction

print("Starting TF-IDF vectorization...")

tfidf_vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words="english",
    ngram_range=(2,3),
    min_df=5
)

# Fit on train lyrics
X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['lyrics_clean'])
print("Train TF-IDF shape:", X_train_tfidf.shape)

# Transform test lyrics
X_test_tfidf = tfidf_vectorizer.transform(test_df['lyrics_clean'])
print("Test TF-IDF shape:", X_test_tfidf.shape)

# Save the TF-IDF vectorizer and matrices
with open(PROJECT_PATH + "tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf_vectorizer, f)
print("TF-IDF vectorizer saved!")

scipy.sparse.save_npz(PROJECT_PATH + "X_train_tfidf.npz", X_train_tfidf)
scipy.sparse.save_npz(PROJECT_PATH + "X_test_tfidf.npz", X_test_tfidf)
print("TF-IDF matrices saved!")

## Step 5: SBERT Embeddings

print("Checking GPU availability...")
print("GPU available:", torch.cuda.is_available())

print("Loading SBERT model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
model.max_seq_length = 256
print("Model loaded. Max sequence length:", model.max_seq_length)

# Chunked encoding to prevent crashes
chunk_size = 20000  # adjust if you have GPU memory issues
train_embeddings = []

print("Starting SBERT encoding for train set...")
for start in range(0, len(train_df), chunk_size):
    end = min(start + chunk_size, len(train_df))
    print(f"Encoding train chunk {start} -> {end}")
    chunk_texts = train_df['lyrics_clean'][start:end].tolist()
    emb = model.encode(chunk_texts, batch_size=256, show_progress_bar=True)
    train_embeddings.append(emb)
    # Removed redundant print for chunk completion to reduce output verbosity

# Combine all chunks
sbert_train = np.vstack(train_embeddings)
print("Train SBERT shape:", sbert_train.shape)
np.save(PROJECT_PATH + "sbert_train.npy", sbert_train)
print("Train embeddings saved!")

# Encode test set
print("\nEncoding test set...")
sbert_test = model.encode(
    test_df['lyrics_clean'].tolist(),
    batch_size=256,
    show_progress_bar=True
)
sbert_test = np.array(sbert_test)
print("Test SBERT shape:", sbert_test.shape)
np.save(PROJECT_PATH + "sbert_test.npy", sbert_test)
print("Test embeddings saved!")
print("\nSBERT embeddings extraction completed successfully.")

## Step 6: Style Feature Extraction

print("Starting style feature extraction...")

# Example slang list for demonstration (can expand)
slang_list = ['ya', 'yo', 'nah', 'shit', 'bitch', 'nigga', 'fuck', 'yeah']

def compute_style_features(text):
    # Clean text lines
    lines = [l.strip() for l in text.split("\n") if l.strip() != ""]

    # Average line length
    avg_line_len = np.mean([len(l.split()) for l in lines]) if lines else 0

    # TTR (Type-Token Ratio)
    words = text.split()
    ttr = len(set(words)) / (len(words)+1e-5)

    # Slang count
    slang_count = sum([words.count(s) for s in slang_list])

    # Word repetition score
    word_counts = {}
    for w in words:
        word_counts[w] = word_counts.get(w,0)+1
    repetition_score = np.mean([c for c in word_counts.values()]) if word_counts else 0

    # Lyrics length (words)
    lyrics_length = len(words)

    # Repeated line ratio
    unique_lines = len(set(lines))
    repeated_line_ratio = 1 - (unique_lines / (len(lines)+1e-5)) if lines else 0

    return [ttr, avg_line_len, slang_count, repetition_score, lyrics_length, repeated_line_ratio]

# Compute features for train set
train_style_features = np.array([compute_style_features(t) for t in train_df['lyrics_clean']])
print("Train style features shape:", train_style_features.shape)

# Compute features for test set
test_style_features = np.array([compute_style_features(t) for t in test_df['lyrics_clean']])
print("Test style features shape:", test_style_features.shape)

# Save style features
np.save(PROJECT_PATH + "train_style_features.npy", train_style_features)
np.save(PROJECT_PATH + "test_style_features.npy", test_style_features)
print("Style features saved successfully!")

## Step 7: Combine TF-IDF + SBERT + Style Features

print("Loading saved features...")

# TF-IDF matrices (sparse)
X_train_tfidf = scipy.sparse.load_npz(PROJECT_PATH + "X_train_tfidf.npz")
X_test_tfidf = scipy.sparse.load_npz(PROJECT_PATH + "X_test_tfidf.npz")

# SBERT embeddings
sbert_train = np.load(PROJECT_PATH + "sbert_train.npy")
sbert_test = np.load(PROJECT_PATH + "sbert_test.npy")

# Style features
train_style_features = np.load(PROJECT_PATH + "train_style_features.npy")
test_style_features = np.load(PROJECT_PATH + "test_style_features.npy")

print("All features loaded.")

# Convert style features and SBERT to sparse for stacking with TF-IDF
from scipy.sparse import hstack, csr_matrix
X_train_combined = hstack([X_train_tfidf, csr_matrix(sbert_train), csr_matrix(train_style_features)])
X_test_combined = hstack([X_test_tfidf, csr_matrix(sbert_test), csr_matrix(test_style_features)])

# Save combined features (sparse) before SVD
scipy.sparse.save_npz(PROJECT_PATH + "X_train_combined.npz", X_train_combined)
scipy.sparse.save_npz(PROJECT_PATH + "X_test_combined.npz", X_test_combined)
print("Combined TF-IDF + SBERT + Style features saved (pre-reduction).")

print("Combined feature shapes:")
print("Train:", X_train_combined.shape)
print("Test:", X_test_combined.shape)

## Step 8: Reduce Dimensions with Truncated SVD

# Load combined features
X_train_combined = scipy.sparse.load_npz(PROJECT_PATH + "X_train_combined.npz")
X_test_combined = scipy.sparse.load_npz(PROJECT_PATH + "X_test_combined.npz")

# Apply TruncatedSVD for dimensionality reduction
svd_components = 300  # adjust if needed
print(f"Applying TruncatedSVD with {svd_components} components...")

svd = TruncatedSVD(n_components=svd_components, random_state=42)
X_train_reduced = svd.fit_transform(X_train_combined)
X_test_reduced = svd.transform(X_test_combined)

print("Reduced feature shapes:")
print("Train:", X_train_reduced.shape)
print("Test:", X_test_reduced.shape)

# Save reduced feature matrices
np.save(PROJECT_PATH + "X_train_reduced.npy", X_train_reduced)
np.save(PROJECT_PATH + "X_test_reduced.npy", X_test_reduced)
with open(PROJECT_PATH + "svd_model.pkl", "wb") as f:
    pickle.dump(svd, f)
print("TruncatedSVD reduced features saved successfully!")

## Step 9: Model Training - Logistic Regression (time consuming)
"""
# Load reduced features and labels
X_train_reduced = np.load(PROJECT_PATH + "X_train_reduced.npy")
X_test_reduced = np.load(PROJECT_PATH + "X_test_reduced.npy")
y_train = np.load(PROJECT_PATH + "y_train.npy")
y_test = np.load(PROJECT_PATH + "y_test.npy")
label_encoder = pickle.load(open(PROJECT_PATH + "label_encoder.pkl", "rb"))

model_name = "Logistic Regression"
model = LogisticRegression(
    C=1.0,
    solver='saga',
    multi_class='multinomial',
    class_weight='balanced',
    max_iter=250,
    n_jobs=-1, verbose=0 # Reduced verbosity for script
)

print(f"\nTraining {model_name}...")
start_time = time.time() # Start time
model.fit(X_train_reduced, y_train)
end_time = time.time() # End time
print(f"Training time for {model_name}: {end_time - start_time:.2f} seconds")

y_pred = model.predict(X_test_reduced)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print(f"\n{model_name} Results:")
print("Accuracy:", acc)
print("Weighted F1-score:", f1)
print("\nClassification Report:\n", report)

model_file = PROJECT_PATH + f"{model_name.replace(' ', '_').lower()}_model.pkl"
with open(model_file, 'wb') as f:
    pickle.dump(model, f)
print(f"{model_name} model saved to {model_file}!")
"""
## Step 10: Model Training - Linear SVC

# Load reduced features and labels (if not already loaded in memory)
X_train_reduced = np.load(PROJECT_PATH + "X_train_reduced.npy")
X_test_reduced = np.load(PROJECT_PATH + "X_test_reduced.npy")
y_train = np.load(PROJECT_PATH + "y_train.npy")
y_test = np.load(PROJECT_PATH + "y_test.npy")
label_encoder = pickle.load(open(PROJECT_PATH + "label_encoder.pkl", "rb"))

model_name = "Linear SVC"
model = LinearSVC(dual=False, C=1.0, class_weight='balanced', max_iter=2000)

print(f"\nTraining {model_name}...")
start_time = time.time() # Start time
model.fit(X_train_reduced, y_train)
end_time = time.time() # End time
print(f"Training time for {model_name}: {end_time - start_time:.2f} seconds")

y_pred = model.predict(X_test_reduced)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print(f"\n{model_name} Results:")
print(f"Accuracy: {acc:.2%}")
print("Weighted F1-score:", f1)
print("\nClassification Report:\n", report)

model_file = PROJECT_PATH + f"{model_name.replace(' ', '_').lower()}_model.pkl"
with open(model_file, 'wb') as f:
     pickle.dump(model, f)
print(f"{model_name} model saved to {model_file}!")

## Step 11: Model Training - SGDClassifier (Logistic Regression approximation, time consuming)
"""
# Load reduced dense features and labels
X_train = np.load(PROJECT_PATH + "X_train_reduced.npy")
X_test  = np.load(PROJECT_PATH + "X_test_reduced.npy")
y_train = np.load(PROJECT_PATH + "y_train.npy")
y_test  = np.load(PROJECT_PATH + "y_test.npy")
label_encoder = pickle.load(open(PROJECT_PATH + "label_encoder.pkl", "rb"))

# Initialize SGDClassifier as Logistic Regression
sgd_log = SGDClassifier(
    loss='log_loss',
    max_iter=1000,
    tol=1e-3,
    class_weight='balanced',
    random_state=42,
    verbose=0, # Reduced verbosity for script
    n_jobs=-1
)

print("\nTraining SGD Logistic Regression...")
start_time = time.time() # Start time
sgd_log.fit(X_train, y_train)
end_time = time.time() # End time
print(f"Training time for SGD Logistic Regression: {end_time - start_time:.2f} seconds")

y_pred = sgd_log.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print(f"\nSGD Logistic Regression Results:")
print("Accuracy:", acc)
print("Weighted F1-score:", f1)
print("\nClassification Report:\n", report)

model_file = PROJECT_PATH + "sgd_logistic_model.pkl"
with open(model_file, 'wb') as f:
    pickle.dump(sgd_log, f)
print(f"SGD Logistic Regression model saved to {model_file}!")
"""
## Step 12: Model Training - SGDClassifier (Linear SVC approximation)

# Load reduced dense features and labels
X_train = np.load(PROJECT_PATH + "X_train_reduced.npy")
X_test  = np.load(PROJECT_PATH + "X_test_reduced.npy")
y_train = np.load(PROJECT_PATH + "y_train.npy")
y_test  = np.load(PROJECT_PATH + "y_test.npy")
label_encoder = pickle.load(open(PROJECT_PATH + "label_encoder.pkl", "rb"))

# Initialize SGDClassifier as Linear SVC
sgd_svc = SGDClassifier(
    loss='hinge',
    max_iter=1000,
    tol=1e-3,
    class_weight='balanced',
    random_state=42,
    verbose=0, # Reduced verbosity for script
    n_jobs=-1
)

print("\nTraining SGD Linear SVC...")
start_time = time.time() # Start time
sgd_svc.fit(X_train, y_train)
end_time = time.time() # End time
print(f"Training time for SGD Linear SVC: {end_time - start_time:.2f} seconds")

y_pred = sgd_svc.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print(f"\nSGD Linear SVC Results:")
print(f"Accuracy: {acc:.2%}")
print("Weighted F1-score:", f1)
print("\nClassification Report:\n", report)

model_file = PROJECT_PATH + "sgd_linear_svc_model.pkl"
with open(model_file, 'wb') as f:
    pickle.dump(sgd_svc, f)
print(f"SGD Linear SVC model saved to {model_file}!")

## Step 13: Model Training - LightGBM Classifier

# Load reduced dense features and labels
X_train = np.load(PROJECT_PATH + "X_train_reduced.npy")
X_test  = np.load(PROJECT_PATH + "X_test_reduced.npy")
y_train = np.load(PROJECT_PATH + "y_train.npy")
y_test  = np.load(PROJECT_PATH + "y_test.npy")
label_encoder = pickle.load(open(PROJECT_PATH + "label_encoder.pkl", "rb"))

# Initialize LightGBM classifier (fast version)
lgb_fast = lgb.LGBMClassifier(
    n_estimators=150,
    learning_rate=0.1,
    num_leaves=31,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced',
    device='gpu', # Enable GPU training
    # n_jobs=-1, # Remove n_jobs when using GPU
    random_state=42,
    verbose=-1
)

print("\nTraining Fast LightGBM classifier...")
start_time = time.time() # Start time
lgb_fast.fit(X_train, y_train)
end_time = time.time() # End time
print(f"Training time for Fast LightGBM: {end_time - start_time:.2f} seconds")

y_pred = lgb_fast.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print(f"\nFast LightGBM Results:")
print(f"Accuracy: {acc:.2%}")
print("Weighted F1-score:", f1)
print("\nClassification Report:\n", report)

model_file = PROJECT_PATH + "lightgbm_fast_model.pkl"
with open(model_file, 'wb') as f:
    pickle.dump(lgb_fast, f)
print(f"Fast LightGBM model saved to {model_file}!")

## Step 14: Semantic Lyrics Search Prototype

print("Loading SBERT model for semantic search...")
sbert_model_search = SentenceTransformer('all-MiniLM-L6-v2')  # same model used for embeddings
print("SBERT model loaded!")

print("Loading saved SBERT embeddings and metadata...")
X_train_sbert_search = np.load(PROJECT_PATH + "sbert_train.npy")

# Load song info
train_df_search = pd.read_pickle(PROJECT_PATH + "train_clean.pkl")   # contains ['Artist', 'Song', 'Genre', ...]

song_titles = train_df_search['Song'].tolist()
song_artists = train_df_search['Artist'].tolist()
song_genres  = train_df_search['Genre'].tolist()
print(f"Loaded {len(song_titles)} songs for search.")

def search_lyrics(query, top_k=5, sbert_model=sbert_model_search, embeddings=X_train_sbert_search, titles=song_titles, artists=song_artists, genres=song_genres):
    # Semantic search using SBERT embeddings
    # query: string, top_k: number of results to return

    # Encode query
    query_vec = sbert_model.encode([query])

    # Compute cosine similarity with all song embeddings
    similarities = cosine_similarity(query_vec, embeddings).flatten()

    # Get top k indices
    top_idx = similarities.argsort()[-top_k:][::-1]

    # Display results
    print(f"\nTop {top_k} results for query: '{query}'\n")
    for i in top_idx:
        print(f"{artists[i]} - {titles[i]} [{genres[i]}] (score: {similarities[i]:.3f})")

# Example search, interactive loop in next cell
search_lyrics("love and heartbreak", top_k=10)
search_lyrics("blue skies", top_k=5)
search_lyrics("upbeat party song", top_k=5)

# Emotion Inference and Mood Mapping - handled separately.

"""## Step 15: Interactive Semantic Lyrics Search

This section provides an interactive loop to perform semantic searches on lyrics. You can continuously enter queries until you type 'exit' to quit.
"""

print("Starting interactive semantic lyrics search. Type 'exit' to quit.")
while True:
    query = input("\nEnter your search query (or 'exit' to quit): ")
    if query.lower() == 'exit':
        print("Exiting interactive search.")
        break
    if query.strip(): # Ensure query is not empty
        search_lyrics(query, top_k=5) # Using the previously defined search_lyrics function
    else:
        print("Query cannot be empty. Please try again.")

## Checking pkl file created

import pickle
import os

model_path = PROJECT_PATH + "lightgbm_fast_model.pkl"
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)
    print(f"\nSuccessfully loaded {os.path.basename(model_path)}")
    print("Loaded model type:", type(loaded_model))
    print("\n--- Model Details ---")
    print("Parameters:", loaded_model.get_params())
    if hasattr(loaded_model, 'best_iteration_'):
        print("Best iteration:", loaded_model.best_iteration_)
    elif hasattr(loaded_model, 'n_estimators'):
        print("Number of estimators (trees):", loaded_model.n_estimators)
    # You can add more attributes to inspect if needed, e.g., feature_importances_
else:
    print(f"\nError: {model_path} not found.")