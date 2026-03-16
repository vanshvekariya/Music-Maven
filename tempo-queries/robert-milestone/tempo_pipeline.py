import os
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATASET = "genres_original"

results = []
failed_files = []
saved_onset_example = False


def estimate_tempo_autocorr(onset_env, sr, hop_length=512):
    """
    Simple autocorrelation-based tempo estimation using the onset envelope.
    """
    ac = librosa.autocorrelate(onset_env, max_size=len(onset_env) // 2)

    # Convert BPM search range to lag range
    bpm_min = 40
    bpm_max = 200

    lag_min = int(np.floor(60.0 * sr / (hop_length * bpm_max)))
    lag_max = int(np.ceil(60.0 * sr / (hop_length * bpm_min)))

    lag_min = max(lag_min, 1)
    lag_max = min(lag_max, len(ac) - 1)

    if lag_min >= lag_max:
        return np.nan

    ac_segment = ac[lag_min:lag_max + 1]
    best_lag = np.argmax(ac_segment) + lag_min
    tempo_bpm = 60.0 * sr / (hop_length * best_lag)

    return float(tempo_bpm)


for genre in os.listdir(DATASET):
    genre_path = os.path.join(DATASET, genre)

    if not os.path.isdir(genre_path):
        continue

    for file in os.listdir(genre_path):
        if not file.endswith(".wav"):
            continue

        path = os.path.join(genre_path, file)

        try:
            # PI.1: audio loading + sample-rate normalization
            y, sr = librosa.load(path, sr=22050, mono=True)

            # PI.2: onset strength envelope
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)

            # PI.3: algorithm 1 - librosa beat tracker
            tempo_librosa, beats = librosa.beat.beat_track(y=y, sr=sr)
            tempo_librosa = float(np.asarray(tempo_librosa).squeeze())

            # PI.3: algorithm 2 - autocorrelation
            tempo_autocorr = estimate_tempo_autocorr(onset_env, sr)

            # PI.5: confidence score
            # Higher variance in onset strength = more rhythmic variation / ambiguity
            confidence = float(np.std(onset_env))

            results.append({
                "song": file,
                "genre": genre,
                "tempo_librosa": tempo_librosa,
                "tempo_autocorr": tempo_autocorr,
                "confidence": confidence
            })

            print(file, tempo_librosa, tempo_autocorr, confidence)

            # Save one onset-envelope example plot for the report
            if not saved_onset_example:
                plt.figure(figsize=(10, 4))
                plt.plot(onset_env)
                plt.title(f"Onset Strength Envelope Example: {file}")
                plt.xlabel("Frame")
                plt.ylabel("Onset Strength")
                plt.tight_layout()
                plt.savefig("onset_example.png", dpi=300, bbox_inches="tight")
                plt.close()
                saved_onset_example = True

        except Exception as e:
            print(f"Skipping {path} -> {e}")
            failed_files.append({
                "song": file,
                "genre": genre,
                "error": str(e)
            })

# PI.4: caching results
df = pd.DataFrame(results)
df.to_csv("tempo_results.csv", index=False)

failed_df = pd.DataFrame(failed_files)
failed_df.to_csv("failed_files.csv", index=False)

print("Saved tempo_results.csv")
print("Saved failed_files.csv")

# Figure 1: histogram of librosa tempo estimates
plt.figure(figsize=(8, 5))
plt.hist(df["tempo_librosa"], bins=30)
plt.title("Distribution of Estimated Tempo (Librosa)")
plt.xlabel("Tempo (BPM)")
plt.ylabel("Number of Tracks")
plt.tight_layout()
plt.savefig("tempo_histogram.png", dpi=300, bbox_inches="tight")
plt.close()

# Figure 2: comparison of two tempo estimators
plt.figure(figsize=(6, 6))
plt.scatter(df["tempo_librosa"], df["tempo_autocorr"], alpha=0.5)
plt.xlabel("Librosa BPM")
plt.ylabel("Autocorrelation BPM")
plt.title("Tempo Estimator Comparison")
plt.tight_layout()
plt.savefig("tempo_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

# Summary table for console
genre_summary = df.groupby("genre")[["tempo_librosa", "tempo_autocorr", "confidence"]].mean()
print("\nMean values by genre:")
print(genre_summary.round(2))