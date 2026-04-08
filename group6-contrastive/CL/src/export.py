import torch
import numpy as np
import librosa
import os
from sklearn.model_selection import train_test_split
from encoder import AudioEncoder
from precompute import audio_to_melspectrogram, fix_length
from dataset import load_all_files

GTZAN_PATH = "../data/genres"
EMBEDDING_DIM = 128
RANDOM_STATE = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load encoder
encoder = AudioEncoder(embedding_dim=EMBEDDING_DIM).to(device)
encoder.load_state_dict(torch.load("../outputs/simclr_encoder.pt", map_location=device))
encoder.eval()
print("Encoder loaded.")

# Load all files and labels
file_paths, labels = load_all_files(GTZAN_PATH)
print(f"Total files: {len(file_paths)}")

# 80/10/10 stratified split
train_idx, temp_idx = train_test_split(
    np.arange(len(file_paths)),
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=labels
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.5,
    random_state=RANDOM_STATE,
    stratify=labels[temp_idx]
)

print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

# Extract embedding for a single file
def get_embedding(path):
    y, sr = librosa.load(path, sr=22050, mono=True, duration=30)
    spec = audio_to_melspectrogram(y, sr)
    spec = fix_length(spec)
    x = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = encoder(x, return_features=True).cpu().numpy()[0]
    return emb

# Extract embeddings for a split
def extract_split(indices, split_name):
    embeddings = []
    split_labels = []
    for i, idx in enumerate(indices):
        emb = get_embedding(file_paths[idx])
        embeddings.append(emb)
        split_labels.append(labels[idx])
        if (i + 1) % 50 == 0:
            print(f"  [{split_name}] {i+1}/{len(indices)} done...")
    return np.array(embeddings), np.array(split_labels)

print("\nExtracting train embeddings...")
X_train, y_train = extract_split(train_idx, "train")

print("\nExtracting val embeddings...")
X_val, y_val = extract_split(val_idx, "val")

print("\nExtracting test embeddings...")
X_test, y_test = extract_split(test_idx, "test")

# Save
os.makedirs("../outputs", exist_ok=True)
np.save("../outputs/X_train_embeddings.npy", X_train)
np.save("../outputs/X_val_embeddings.npy", X_val)
np.save("../outputs/X_test_embeddings.npy", X_test)
np.save("../outputs/y_train.npy", y_train)
np.save("../outputs/y_val.npy", y_val)
np.save("../outputs/y_test.npy", y_test)

print("\nDone! Files saved to outputs/")
print(f"X_train: {X_train.shape}")
print(f"X_val:   {X_val.shape}")
print(f"X_test:  {X_test.shape}")
