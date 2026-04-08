import numpy as np
import librosa

def audio_to_melspectrogram(y, sr=22050, n_mels=128, hop_length=512):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, hop_length=hop_length
    )
    return librosa.power_to_db(S, ref=np.max)

def fix_length(spec, target=128):
    if spec.shape[1] >= target:
        return spec[:, :target]
    pad = target - spec.shape[1]
    return np.pad(spec, ((0, 0), (0, pad)))