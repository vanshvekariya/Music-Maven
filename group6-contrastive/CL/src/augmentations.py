import numpy as np
import librosa
import random

def augment_waveform(y, sr=22050):
    # Pitch shift
    if random.random() > 0.5:
        n_steps = random.uniform(-2, 2)
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

    # Time stretch
    if random.random() > 0.5:
        rate = random.uniform(0.8, 1.2)
        y = librosa.effects.time_stretch(y, rate=rate)

    # Add noise
    if random.random() > 0.5:
        noise = np.random.randn(len(y)) * 0.005
        y = y + noise

    return y
