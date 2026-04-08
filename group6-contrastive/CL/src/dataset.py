import os
import glob
import numpy as np
import torch
import librosa
from torch.utils.data import Dataset
from precompute import audio_to_melspectrogram, fix_length

GENRE_MAP = {
    'blues': 0, 'classical': 1, 'country': 2, 'disco': 3,
    'hiphop': 4, 'jazz': 5, 'metal': 6, 'pop': 7,
    'reggae': 8, 'rock': 9
}

def load_all_files(gtzan_path):
    file_paths = sorted(glob.glob(os.path.join(gtzan_path, "**/*.wav"), recursive=True))
    file_paths = [p for p in file_paths if "jazz.00054" not in p]
    labels = []
    for path in file_paths:
        genre = os.path.basename(os.path.dirname(path))
        labels.append(GENRE_MAP[genre])
    return file_paths, np.array(labels)


# 🔥 Spectrogram augmentations (keep these)
def augment_spec(spec):
    spec = spec.copy()

    # Add small noise
    if np.random.rand() < 0.3:
        noise = np.random.randn(*spec.shape) * 0.01
        spec += noise

    # Time masking
    if np.random.rand() < 0.3:
        t = spec.shape[1]
        mask_size = np.random.randint(5, 20)
        start = np.random.randint(0, max(1, t - mask_size))
        spec[:, start:start + mask_size] = 0

    # Frequency masking
    if np.random.rand() < 0.3:
        f = spec.shape[0]
        mask_size = np.random.randint(5, 20)
        start = np.random.randint(0, max(1, f - mask_size))
        spec[start:start + mask_size, :] = 0

    return spec

def augment_audio(y, sr):
    # Pitch shift
    if np.random.rand() < 0.3:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=np.random.randint(-1, 2))
    return y


class SimCLRDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]

        # Load raw audio
        y, sr = librosa.load(path, sr=22050)

        # Create two augmented audio views
        y1 = augment_audio(y.copy(), sr)
        y2 = augment_audio(y.copy(), sr)

        # Convert to mel spectrograms
        spec1 = audio_to_melspectrogram(y1, sr)
        spec2 = audio_to_melspectrogram(y2, sr)

        # Fix length
        spec1 = fix_length(spec1)
        spec2 = fix_length(spec2)

        # Optional: apply spectrogram augmentations
        spec1 = augment_spec(spec1)
        spec2 = augment_spec(spec2)

        # Convert to tensors
        t1 = torch.tensor(spec1, dtype=torch.float32).unsqueeze(0)
        t2 = torch.tensor(spec2, dtype=torch.float32).unsqueeze(0)

        return t1, t2