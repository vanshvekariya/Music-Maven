
# Due to the volume of the training dataset (290k songs), we used only test dataset of ~8k songs.

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import os
import pickle # For RenameUnpickler
from torch.utils.data import DataLoader, Dataset # For efficient emotion inference
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm # For progress bar

print("All necessary libraries imported.")


# --- Configuration ---
PROJECT_PATH = "./data/"

# Ensure the PROJECT_PATH exists
os.makedirs(PROJECT_PATH, exist_ok=True)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMOTION_MODEL_NAME = "bhadresh-savani/distilbert-base-uncased-emotion"
TOP_N = 5
BATCH_SIZE_EMOTION_INFERENCE = 128 # For the emotion inference step
BATCH_SIZE_SBERT_EMBEDDING = 64 # For SBERT embeddings computation

# Use GPU if available, else CPU for efficiency
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Helper: Custom Unpickler for NumPy Compatibility ---
class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "numpy._core.numeric": module = "numpy.core.numeric"
        elif module == "numpy._core": module = "numpy._core"
        return super().find_class(module, name)

# --- Helper: Dataset Class for Emotion Inference ---
class LyricsDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        item = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding='max_length', # Use 'max_length' for consistent batching
            max_length=512,
            return_tensors="pt"
        )
        return {key: val.squeeze(0) for key, val in item.items()}

    def __len__(self):
        return len(self.texts)

# ==============================================================================
# STAGE 1: Emotion Inference (from test_clean.pkl to test_emotion.npy)
# ==============================================================================
print("\n--- STAGE 1: Emotion Inference ---")
print("Loading emotion model and tokenizer for inference...")
emotion_tokenizer_inf = AutoTokenizer.from_pretrained(EMOTION_MODEL_NAME)
emotion_model_inf = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL_NAME).to(device).half() # Half-precision for speed
emotion_model_inf.eval()

# Load cleaned test data (from original path)
with open(os.path.join(PROJECT_PATH, "test_clean.pkl"), 'rb') as f:
    df_clean = RenameUnpickler(f).load()

dataset_inf = LyricsDataset(df_clean["lyrics_clean"].astype(str).tolist(), emotion_tokenizer_inf)
loader_inf = DataLoader(dataset_inf, batch_size=BATCH_SIZE_EMOTION_INFERENCE, shuffle=False, num_workers=0) # num_workers=0 based on env

all_probs = []
print(f"Starting emotion inference for {len(df_clean)} songs...")
with torch.no_grad():
    for batch in tqdm(loader_inf, desc="Emotion Inference"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = emotion_model_inf(**batch)
        probs = torch.softmax(outputs.logits, dim=1).float().cpu().numpy()
        all_probs.append(probs)

emotions_npy = np.vstack(all_probs)
np.save(os.path.join(PROJECT_PATH, "test_emotion.npy"), emotions_npy)
print(f"Emotion probabilities saved to {os.path.join(PROJECT_PATH, 'test_emotion.npy')}")

# ==============================================================================
# STAGE 2: Valence, Mood Mapping & Encoding
# ==============================================================================
print("\n--- STAGE 2: Valence, Mood Mapping & Encoding ---")

# Load emotion model config to get label mapping
# The full model is not re-loaded, only its config for label mapping
model_config_for_labels = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL_NAME)
label_to_index = {label.lower(): int(idx) for idx, label in model_config_for_labels.config.id2label.items()}

# Correct index mapping
sad = label_to_index["sadness"]
joy = label_to_index["joy"]
love = label_to_index["love"]
anger = label_to_index["anger"]
fear = label_to_index["fear"]
surprise = label_to_index["surprise"]

# VALENCE calculation
valence = (
    +1.0 * emotions_npy[:, joy] +
    +0.9 * emotions_npy[:, love] +
    +0.1 * emotions_npy[:, surprise] -
    1.0 * emotions_npy[:, sad] -
    0.8 * emotions_npy[:, anger] -
    0.7 * emotions_npy[:, fear]
)
valence = np.clip(valence, -1, 1)
np.save(os.path.join(PROJECT_PATH, "test_valence.npy"), valence)
print(f"Valence scores saved to {os.path.join(PROJECT_PATH, 'test_valence.npy')}")

# Mood mapping function
def map_mood(v):
    if v > 0.6: return "very_positive"
    elif v > 0.2: return "positive"
    elif v < -0.6: return "very_negative"
    elif v < -0.2: return "negative"
    else: return "neutral"

mood = np.array([map_mood(v) for v in valence])
np.save(os.path.join(PROJECT_PATH, "test_mood.npy"), mood)
print(f"Mood labels saved to {os.path.join(PROJECT_PATH, 'test_mood.npy')}")

# Mood encoding
mood_to_int = {
    "very_negative": 0, "negative": 1, "neutral": 2,
    "positive": 3, "very_positive": 4
}
mood_encoded = np.array([mood_to_int[m] for m in mood], dtype=np.int8)
np.save(os.path.join(PROJECT_PATH, "test_mood_encoded.npy"), mood_encoded)
print(f"Mood encoded labels saved to {os.path.join(PROJECT_PATH, 'test_mood_encoded.npy')}")

# ==============================================================================
# STAGE 3: Merge all features into a DataFrame
# ==============================================================================
print("\n--- STAGE 3: Merging all features ---")
emotion_cols = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
top_idx = emotions_npy.argmax(axis=1)
top_emotion = [emotion_cols[i] for i in top_idx]

df_merged_features = pd.concat([
    df_clean.reset_index(drop=True),
    pd.DataFrame(emotions_npy, columns=emotion_cols),
    pd.DataFrame({'valence': valence, 'mood_encoded': mood_encoded, 'top_emotion': top_emotion})
], axis=1)

df_merged_features.to_pickle(os.path.join(PROJECT_PATH, "test_val_mod_merged.pkl"))
print(f"Merged DataFrame saved to {os.path.join(PROJECT_PATH, 'test_val_mod_merged.pkl')}")

# ==============================================================================
# STAGE 4: Lyrics Embeddings (SBERT)
# ==============================================================================
print("\n--- STAGE 4: Generating Lyrics Embeddings ---")
print("Loading SBERT model and tokenizer...")
sbert_tokenizer_sbert = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
sbert_model_sbert = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(device).eval()

# Helper: get sentence embeddings for SBERT
def embed_texts_sbert(texts, batch_size=BATCH_SIZE_SBERT_EMBEDDING):
    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="SBERT Embedding"):
            batch = texts[i:i+batch_size]
            enc = sbert_tokenizer_sbert(batch, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
            outputs = sbert_model_sbert(**enc)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            all_embeddings.append(embeddings.cpu().numpy())
    return np.vstack(all_embeddings)

print("Computing SBERT embeddings for all lyrics...")
df_merged_features["lyrics_embedding"] = list(embed_texts_sbert(df_merged_features["lyrics_clean"].astype(str).tolist()))
df_merged_features.to_pickle(os.path.join(PROJECT_PATH, "lyrics_with_embeddings.pkl"))
print(f"Lyrics with embeddings saved to {os.path.join(PROJECT_PATH, 'lyrics_with_embeddings.pkl')}")

# ==============================================================================
# STAGE 5: Hybrid Search Functionality
# ==============================================================================
print("\n--- STAGE 5: Initializing Hybrid Search ---")

# Reload df with embeddings (or just use the one already in memory, df_merged_features)
df_final = df_merged_features.copy() # Use a copy to avoid SettingWithCopyWarning if any further modifications
lyrics_embs_matrix = np.array(df_final["lyrics_embedding"].tolist())

# Load models and tokenizers for search (they might have been unloaded if previous stages took too much memory, so re-initialize)
print("Loading search models for interactive search...")
search_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
search_model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(device).eval()

emotion_tokenizer_search = AutoTokenizer.from_pretrained(EMOTION_MODEL_NAME)
emotion_model_search = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL_NAME).to(device).eval()

# Helper: Embed Query Text for search
def embed_query_search(text):
    with torch.no_grad():
        enc = search_tokenizer([text], padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
        outputs = search_model(**enc)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

# Hybrid Search Function
def search_songs_hybrid(query, top_n=TOP_N):
    # --- A. Semantic Meaning (MiniLM) ---
    query_search_vec = embed_query_search(query)
    sim_semantic = cosine_similarity(query_search_vec, lyrics_embs_matrix)[0]

    # --- B. Emotion Vector (DistilBERT) ---
    with torch.no_grad():
        inputs_e = emotion_tokenizer_search([query], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        out_e = emotion_model_search(**inputs_e)
        query_emotion_vec = torch.softmax(out_e.logits, dim=1).cpu().numpy()

    emotion_cols = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    song_emotions = df_final[emotion_cols].values

    # Calculate emotion similarity using dot product
    sim_emotion_dot_product = (query_emotion_vec @ song_emotions.T)[0]

    # Calculate emotion similarity using cosine similarity
    sim_emotion_cosine = cosine_similarity(query_emotion_vec, song_emotions)[0]

    # --- C. Weighted Fusion (using dot product for ranking) ---
    final_score = (0.5 * sim_semantic) + (0.5 * sim_emotion_dot_product)

    # --- D. Rank and Return ---
    top_idx = final_score.argsort()[::-1][:top_n]
    results = df_final.iloc[top_idx].copy()
    results["match_score"] = final_score[top_idx]
    results["emotion_sim_dot_product"] = sim_emotion_dot_product[top_idx]
    results["emotion_sim_cosine"] = sim_emotion_cosine[top_idx]

    return results

# --- Interactive Test Loop ---
print("\n--- Starting Interactive Search ---")
print("Enter a lyrics snippet or emotion (e.g., 'I feel sad', 'love', 'The sun is shining')")
print("Type 'exit' or 'quit' to end the search.")
while True:
    query = input("\nEnter your query: ").strip()
    if query.lower() in ["exit", "quit"]:
        print("Exiting interactive search.")
        break

    results = search_songs_hybrid(query, top_n=TOP_N)
    print(f"\nTop matches for '{query}':\n")

    for i, row in results.iterrows():
        print(f"Song: {row['Song']} - {row['Artist']}")
        print(f"Emotion: {row['top_emotion'].upper()} (Valence: {row.get('valence', 0):.2f})")
        print(f"Lyrics: {row['lyrics_clean'][:150]}...")
        print(f"Match Score (Hybrid): {row['match_score']:.3f}")
        print(f"Emotion Similarity (Dot Product): {row['emotion_sim_dot_product']:.3f}")
        print(f"Emotion Similarity (Cosine): {row['emotion_sim_cosine']:.3f}")
        print("-" * 50)