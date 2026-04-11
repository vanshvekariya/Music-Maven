"""
Run the full beat tracking + semantic mapping pipeline on GTZAN dataset.
Generates all charts and graphs for the presentation.
"""

import os

# ── 1. CRITICAL: STOP NUMPY THREAD THRASHING ──
# Must be set before numpy/scipy/madmom are imported
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import time
from pathlib import Path
import librosa
from concurrent.futures import ProcessPoolExecutor, as_completed

# =============================================
# BEAT TRACKER (with madmom support)
# =============================================

try:
    from madmom.features.downbeats import (
        RNNDownBeatProcessor,
        DBNDownBeatTrackingProcessor,
    )
    MADMOM_AVAILABLE = True
    print("✓ Madmom loaded — using RNN beat tracking")
except ImportError:
    MADMOM_AVAILABLE = False
    print("✗ Madmom not available — falling back to librosa")

# Global variables for the multiprocessing workers
worker_rnn_processor = None
worker_dbn_processor = None

def _init_worker():
    """
    Runs once per CPU core. Loads the heavy models into RAM permanently
    for that specific worker process.
    """
    global worker_rnn_processor, worker_dbn_processor
    if MADMOM_AVAILABLE:
        # Initialize only once per core!
        worker_rnn_processor = RNNDownBeatProcessor(fps=100)
        worker_dbn_processor = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)

def track_beats(file_path: str, fps: int = 100) -> dict:
    """Track beats and downbeats in an audio file."""
    start = time.time()
    if MADMOM_AVAILABLE:
        return _track_madmom(file_path, fps, start)
    else:
        return _track_librosa(file_path, start)

def _track_madmom(file_path: str, fps: int, start: float) -> dict:
    """RNN-based beat and downbeat tracking via cached madmom processors."""
    global worker_rnn_processor, worker_dbn_processor
    
    activations = worker_rnn_processor(file_path)
    beat_info = worker_dbn_processor(activations)
    
    all_beats = beat_info[:, 0]
    beat_positions = beat_info[:, 1]
    downbeats = all_beats[beat_positions == 1]
    
    act_1d = activations.sum(axis=1) if activations.ndim > 1 else activations
    
    bpm = 0.0
    if len(all_beats) > 1:
        ibis = np.diff(all_beats)
        median_ibi = np.median(ibis)
        if median_ibi > 0:
            bpm = 60.0 / median_ibi
            
    meter = 4
    if len(downbeats) >= 2:
        bpb_estimates = []
        for i in range(len(downbeats) - 1):
            bar_beats = np.sum((all_beats >= downbeats[i]) & (all_beats < downbeats[i + 1]))
            bpb_estimates.append(bar_beats)
        if bpb_estimates:
            meter = int(round(np.median(bpb_estimates)))
            
    if len(act_1d) > 0:
        peak_thresh = np.percentile(act_1d, 90)
        peak_ratio = np.mean(act_1d > peak_thresh)
        norm = min(1.0, np.max(act_1d) / (np.mean(act_1d) + 1e-8) / 10.0)
        confidence = float(np.clip((norm + peak_ratio) / 2, 0, 1))
    else:
        confidence = 0.0
        
    return {
        "file_path": file_path,
        "beats": all_beats,
        "downbeats": downbeats,
        "activations": act_1d,
        "bpm": bpm,
        "meter": meter,
        "confidence": confidence,
        "time": time.time() - start,
        "backend": "madmom",
    }

def _track_librosa(file_path: str, start: float) -> dict:
    """Librosa fallback for beat tracking."""
    import warnings
    warnings.filterwarnings("ignore") # suppress librosa warnings in workers
    
    y, sr = librosa.load(file_path, sr=22050, mono=True)
    duration = len(y) / sr
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames", onset_envelope=onset_env)
    if hasattr(tempo, '__len__'):
        tempo = float(tempo[0]) if len(tempo) > 0 else 0.0
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    if len(beat_times) == 0 and tempo > 0:
        beat_interval = 60.0 / tempo
        beat_times = np.arange(beat_interval, duration, beat_interval)
        
    downbeats = beat_times[::4] if len(beat_times) >= 4 else beat_times[:1]
    
    bpm = 0.0
    if len(beat_times) > 1:
        ibis = np.diff(beat_times)
        median_ibi = np.median(ibis)
        if median_ibi > 0:
            bpm = 60.0 / median_ibi
            
    confidence = 0.5
    
    return {
        "file_path": file_path,
        "beats": beat_times,
        "downbeats": downbeats,
        "activations": onset_env,
        "bpm": bpm,
        "meter": 4,
        "confidence": confidence,
        "time": time.time() - start,
        "backend": "librosa",
    }

# =============================================
# SEMANTIC MAPPER
# =============================================

def classify_tempo(bpm):
    if bpm < 60: return ("Largo", "Very slow, broad")
    elif bpm < 80: return ("Adagio", "Slow, restful")
    elif bpm < 100: return ("Andante", "Walking pace")
    elif bpm < 120: return ("Moderato", "Moderate")
    elif bpm < 156: return ("Allegro", "Fast, lively")
    elif bpm < 176: return ("Vivace", "Brisk, vibrant")
    elif bpm < 210: return ("Presto", "Very fast")
    else: return ("Prestissimo", "Extremely fast")

def classify_energy(bpm, onset_density=None):
    bpm_norm = max(0, min(1, (bpm - 40) / (210 - 40)))
    if onset_density is not None:
        score = bpm_norm * 0.6 + onset_density * 0.4
    else:
        score = bpm_norm
        
    if score < 0.15: return "Very Low"
    elif score < 0.35: return "Low"
    elif score < 0.55: return "Moderate"
    elif score < 0.75: return "High"
    else: return "Very High"

def classify_mood(bpm, energy):
    speed = "slow" if bpm < 80 else "medium" if bpm < 120 else "fast" if bpm < 176 else "very_fast"
    elevel = "low" if energy in ["Very Low", "Low"] else "mid" if energy == "Moderate" else "high"
    
    mood_map = {
        ("slow", "low"): "Serene", ("slow", "mid"): "Melancholy", ("slow", "high"): "Melancholy",
        ("medium", "low"): "Chill", ("medium", "mid"): "Groovy", ("medium", "high"): "Groovy",
        ("fast", "low"): "Chill", ("fast", "mid"): "Upbeat", ("fast", "high"): "Intense",
        ("very_fast", "low"): "Upbeat", ("very_fast", "mid"): "Intense", ("very_fast", "high"): "Frantic",
    }
    return mood_map.get((speed, elevel), "Groovy")

# =============================================
# PARALLEL WORKER WRAPPER
# =============================================

def process_single_file_job(args):
    """Wrapper to handle the try/except logic inside the worker pool."""
    index, genre, fpath = args
    try:
        r = track_beats(fpath)
        act = r["activations"]
        if len(act) > 0:
            onset_density = float(np.mean(act > np.percentile(act, 75)))
        else:
            onset_density = 0.5
            
        tempo_label, tempo_desc = classify_tempo(r["bpm"])
        energy = classify_energy(r["bpm"], onset_density)
        mood = classify_mood(r["bpm"], energy)
        
        result_dict = {
            "file": Path(fpath).name,
            "genre": genre,
            "bpm": round(r["bpm"], 1),
            "beats": len(r["beats"]),
            "downbeats": len(r["downbeats"]),
            "meter": r["meter"],
            "confidence": round(r["confidence"], 3),
            "tempo_label": tempo_label,
            "energy": energy,
            "mood": mood,
            "time": round(r["time"], 2),
            "backend": r["backend"],
        }
        return ("success", result_dict)
    except Exception as e:
        return ("error", {"file": fpath, "genre": genre, "error": str(e)})

# =============================================
# MAIN: Process GTZAN
# =============================================

def main():
    print("=" * 65)
    print("  GTZAN Beat Tracking + Semantic Mapping Pipeline (FAST MODE)")
    print("=" * 65)
    
    gtzan_dir = Path("data/gtzan/genres_original")
    output_dir = Path("output/gtzan_figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not gtzan_dir.exists():
        print(f"ERROR: GTZAN not found at {gtzan_dir}")
        return
        
    genres = sorted([d.name for d in gtzan_dir.iterdir() if d.is_dir()])
    print(f"\nFound genres: {genres}")
    
    all_files = []
    for genre in genres:
        genre_dir = gtzan_dir / genre
        files = sorted(genre_dir.glob("*.wav"))[:100]
        for f in files:
            all_files.append((genre, str(f)))
            
    total_files = len(all_files)
    print(f"Total files to process: {total_files}")
    
    results = []
    failed = []
    
    # Use max cores minus 1 so the OS doesn't lock up
    cores_to_use = max(1, os.cpu_count() - 1)
    print(f"\nSpinning up {cores_to_use} parallel workers...")
    
    # Prepare arguments for the workers
    job_args = [(i, genre, fpath) for i, (genre, fpath) in enumerate(all_files)]
    
    # Run the parallel pool
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=cores_to_use, initializer=_init_worker) as executor:
        futures = {executor.submit(process_single_file_job, arg): arg for arg in job_args}
        
        completed_count = 0
        for future in as_completed(futures):
            completed_count += 1
            status, data = future.result()
            
            if status == "success":
                results.append(data)
            else:
                failed.append(data)
                if len(failed) <= 5:
                    print(f"    ✗ Failed: {Path(data['file']).name} — {data['error']}")
                    
            if completed_count % 50 == 0 or completed_count == total_files:
                print(f"  Processed {completed_count}/{total_files} files...")

    total_wall_time = time.time() - start_time
    
    print(f"\n{'='*65}")
    print(f"  Parallel Processing Complete in {total_wall_time:.1f} seconds!")
    print(f"  Processed: {len(results)} | Failed: {len(failed)}")
    print(f"  Backend: {results[0]['backend'] if results else 'N/A'}")
    print(f"{'='*65}")
    
    # ── Save raw results ──
    with open(output_dir / "gtzan_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir / 'gtzan_results.json'}")
    
    # ══════════════════════════════════════════
    # GENERATE FIGURES (Unchanged, done sequentially at the end)
    # ══════════════════════════════════════════
    
    import pandas as pd
    df = pd.DataFrame(results)
    
    # ── FIGURE 1: BPM Distribution by Genre ──
    print("\n[1] Generating BPM distribution by genre...")
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    fig.suptitle("BPM Distribution Across GTZAN Genres", fontsize=16, fontweight="bold")
    
    genre_order = df.groupby("genre")["bpm"].median().sort_values().index.tolist()
    positions = range(len(genre_order))
    
    bp = ax.boxplot(
        [df[df["genre"] == g]["bpm"].values for g in genre_order],
        positions=positions,
        patch_artist=True,
        widths=0.6,
    )
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(genre_order)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        
    ax.set_xticks(positions)
    ax.set_xticklabels(genre_order, fontsize=11, fontweight="bold")
    ax.set_ylabel("BPM", fontsize=12)
    ax.set_xlabel("Genre", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    
    for i, g in enumerate(genre_order):
        med = df[df["genre"] == g]["bpm"].median()
        ax.text(i, med + 3, f"{med:.0f}", ha="center", fontsize=9, fontweight="bold")
        
    fig.savefig(str(output_dir / "bpm_by_genre.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved: bpm_by_genre.png")
    
    # ── FIGURE 2: Semantic Mapping Overview (Tempo class distribution) ──
    print("[2] Generating tempo class distribution...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle("Semantic Mapping: GTZAN Dataset", fontsize=16, fontweight="bold")
    
    tempo_counts = df["tempo_label"].value_counts()
    tempo_order = ["Largo", "Adagio", "Andante", "Moderato", "Allegro", "Vivace", "Presto", "Prestissimo"]
    tempo_counts = tempo_counts.reindex([t for t in tempo_order if t in tempo_counts.index])
    
    tempo_colors = ["#81D4FA", "#4FC3F7", "#29B6F6", "#03A9F4", "#039BE5", "#0288D1", "#0277BD", "#01579B"]
    ax1.pie(
        tempo_counts.values, labels=tempo_counts.index, autopct="%1.0f%%",
        colors=tempo_colors[:len(tempo_counts)], startangle=90,
        textprops={"fontsize": 10}
    )
    ax1.set_title("Tempo Classification", fontsize=13, fontweight="bold")
    
    mood_counts = df["mood"].value_counts()
    mood_colors_map = {
        "Serene": "#81D4FA", "Melancholy": "#B39DDB", "Chill": "#80CBC4",
        "Groovy": "#FFD54F", "Upbeat": "#FFB74D", "Intense": "#E57373", "Frantic": "#F44336",
    }
    bar_colors = [mood_colors_map.get(m, "#999999") for m in mood_counts.index]
    
    ax2.barh(mood_counts.index, mood_counts.values, color=bar_colors, edgecolor="white")
    ax2.set_xlabel("Number of Tracks", fontsize=11)
    ax2.set_title("Mood Classification", fontsize=13, fontweight="bold")
    ax2.grid(True, axis="x", alpha=0.3)
    
    for i, (v, label) in enumerate(zip(mood_counts.values, mood_counts.index)):
        ax2.text(v + 5, i, str(v), va="center", fontsize=10, fontweight="bold")
        
    fig.savefig(str(output_dir / "semantic_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved: semantic_distribution.png")
    
    # ── FIGURE 3: Energy by Genre heatmap ──
    print("[3] Generating energy-genre heatmap...")
    energy_order = ["Very Low", "Low", "Moderate", "High", "Very High"]
    cross = pd.crosstab(df["genre"], df["energy"])
    cross = cross.reindex(columns=[e for e in energy_order if e in cross.columns], fill_value=0)
    cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100
    
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    im = ax.imshow(cross_pct.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(cross_pct.columns)))
    ax.set_xticklabels(cross_pct.columns, fontsize=11)
    ax.set_yticks(range(len(cross_pct.index)))
    ax.set_yticklabels(cross_pct.index, fontsize=11)
    ax.set_title("Energy Level Distribution by Genre (%)", fontsize=14, fontweight="bold")
    
    for i in range(len(cross_pct.index)):
        for j in range(len(cross_pct.columns)):
            val = cross_pct.values[i, j]
            if val > 0:
                color = "white" if val > 40 else "black"
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=9, color=color, fontweight="bold")
                
    fig.colorbar(im, ax=ax, label="% of genre", shrink=0.8)
    fig.savefig(str(output_dir / "energy_by_genre.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved: energy_by_genre.png")
    
    # ── FIGURE 4: Beat tracking detail for one track per genre ──
    print("[4] Generating per-genre beat tracking examples...")
    fig, axes = plt.subplots(2, 5, figsize=(20, 7), constrained_layout=True)
    fig.suptitle("Beat Tracking Across GTZAN Genres (Mel-Spectrogram + Beats)", fontsize=16, fontweight="bold")
    
    for idx, genre in enumerate(genres[:10]):
        ax = axes[idx // 5][idx % 5]
        genre_files = [f for f in all_files if f[0] == genre]
        if not genre_files: continue
        
        fpath = genre_files[0][1]
        try:
            y, sr = librosa.load(fpath, sr=22050, duration=30.0, mono=True)
            mel_spec = librosa.power_to_db(
                librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512),
                ref=np.max,
            )
            librosa.display.specshow(mel_spec, sr=sr, hop_length=512, x_axis="time", y_axis="mel", ax=ax, cmap="magma")
            
            genre_result = [r for r in results if r["file"] == Path(fpath).name]
            if genre_result:
                bpm_val = genre_result[0]["bpm"]
                ax.set_title(f"{genre}\n{bpm_val:.0f} BPM", fontsize=10, fontweight="bold")
            else:
                ax.set_title(genre, fontsize=10, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("")
        except Exception as e:
            ax.set_title(f"{genre}\n(error)", fontsize=10)
            
    fig.savefig(str(output_dir / "spectrograms_by_genre.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved: spectrograms_by_genre.png")
    
    # ── FIGURE 5: Confidence by Genre ──
    print("[5] Generating confidence by genre...")
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    fig.suptitle("Beat Tracking Confidence by Genre", fontsize=16, fontweight="bold")
    
    genre_conf = df.groupby("genre")["confidence"].agg(["mean", "std"]).sort_values("mean", ascending=True)
    colors = ["#F44336" if m < 0.3 else "#FF9800" if m < 0.5 else "#4CAF50" for m in genre_conf["mean"]]
    
    bars = ax.barh(genre_conf.index, genre_conf["mean"], xerr=genre_conf["std"],
                   color=colors, alpha=0.85, capsize=3, edgecolor="white")
    ax.set_xlabel("Mean Confidence Score", fontsize=12)
    ax.set_xlim(0, 1)
    ax.grid(True, axis="x", alpha=0.3)
    
    for i, (mean, std) in enumerate(zip(genre_conf["mean"], genre_conf["std"])):
        ax.text(mean + std + 0.02, i, f"{mean:.2f}", va="center", fontsize=10, fontweight="bold")
        
    fig.savefig(str(output_dir / "confidence_by_genre.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved: confidence_by_genre.png")
    
    # ── FIGURE 6: Detailed beat tracking for one good track ──
    print("[6] Generating detailed beat tracking figure...")
    good_track = None
    for genre, fpath in all_files:
        if genre in ["disco", "rock"]:
            good_track = (genre, fpath)
            break
            
    if good_track:
        genre, fpath = good_track
        
        # Re-initialize processors for this single sequential call
        if MADMOM_AVAILABLE:
            _init_worker()
        
        # Run beat tracking fresh on this one track to get full arrays
        r = track_beats(fpath)
        
        y, sr_val = librosa.load(fpath, sr=22050, duration=30.0, mono=True)
        duration = len(y) / sr_val
        time_axis = np.linspace(0, duration, len(y))
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 9), constrained_layout=True)
        fig.suptitle(
            f"Beat & Downbeat Tracking: {Path(fpath).name}\n"
            f"BPM={r['bpm']:.1f} | Meter={r['meter']}/4 | Confidence={r['confidence']:.3f} | Backend={r['backend']}",
            fontsize=13, fontweight="bold",
        )
        
        # Waveform + beats
        axes[0].plot(time_axis, y, linewidth=0.3, color="#90CAF9", alpha=0.7)
        for bt in r["beats"]:
            if bt <= duration:
                axes[0].axvline(x=bt, color="#2196F3", linewidth=0.5, alpha=0.4)
        for db in r["downbeats"]:
            if db <= duration:
                axes[0].axvline(x=db, color="#F44336", linewidth=1.5, alpha=0.8)
        axes[0].set_ylabel("Amplitude")
        axes[0].set_title("Waveform with Beats (blue) and Downbeats (red)")
        axes[0].set_xlim(0, duration)
        
        from matplotlib.lines import Line2D
        legend = [
            Line2D([0], [0], color="#2196F3", linewidth=1.5, label=f"Beats ({len(r['beats'])})"),
            Line2D([0], [0], color="#F44336", linewidth=2.5, label=f"Downbeats ({len(r['downbeats'])})"),
        ]
        axes[0].legend(handles=legend, loc="upper right", fontsize=9)
        
        # Spectrogram + downbeats
        mel_spec = librosa.power_to_db(
            librosa.feature.melspectrogram(y=y, sr=sr_val, n_mels=128, hop_length=512),
            ref=np.max,
        )
        librosa.display.specshow(mel_spec, sr=sr_val, hop_length=512, x_axis="time", y_axis="mel", ax=axes[1], cmap="magma")
        for db in r["downbeats"]:
            if db <= duration:
                axes[1].axvline(x=db, color="#00E676", linewidth=1.2, alpha=0.7)
        axes[1].set_title("Mel-Spectrogram with Downbeats (green)")
        
        # Activation function
        act = r["activations"]
        if len(act) > 0:
            act_time = np.linspace(0, duration, len(act))
            axes[2].plot(act_time, act, color="#FF9800", linewidth=0.8)
            axes[2].fill_between(act_time, 0, act, color="#FF9800", alpha=0.2)
            for db in r["downbeats"]:
                if db <= duration:
                    axes[2].axvline(x=db, color="#F44336", linewidth=1.0, alpha=0.6)
        title_suffix = "RNN Beat Activation Function" if r["backend"] == "madmom" else "Onset Strength Envelope"
        axes[2].set_title(title_suffix)
        axes[2].set_xlabel("Time (s)")
        axes[2].set_xlim(0, duration)
        
        fig.savefig(str(output_dir / "beat_tracking_detail.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → Saved: beat_tracking_detail.png")
            
    # ── SUMMARY TABLE ──
    print("\n" + "=" * 65)
    print("  GENRE SUMMARY")
    print("=" * 65)
    print(f"\n  {'Genre':<12s} {'Tracks':>6s} {'Mean BPM':>9s} {'Std BPM':>8s} {'Conf':>6s} {'Top Mood':<12s} {'Top Tempo':<10s}")
    print("  " + "-" * 65)
    
    for genre in genres:
        g = df[df["genre"] == genre]
        top_mood = g["mood"].mode().iloc[0] if len(g) > 0 else "N/A"
        top_tempo = g["tempo_label"].mode().iloc[0] if len(g) > 0 else "N/A"
        print(f"  {genre:<12s} {len(g):>6d} {g['bpm'].mean():>9.1f} {g['bpm'].std():>8.1f} {g['confidence'].mean():>6.2f} {top_mood:<12s} {top_tempo:<10s}")
        
    print(f"\n  Total GPU/CPU crunch time (sum of all track times): {df['time'].sum():.1f}s")
    print(f"  Average per track compute: {df['time'].mean():.2f}s")
    print(f"  Actual Wall Clock Time: {total_wall_time:.1f}s ({total_wall_time/60:.1f} min)")
    
    print(f"\n  All figures saved to: {output_dir.resolve()}")
    print("=" * 65)

if __name__ == "__main__":
    # Required for safe multiprocessing on Windows/WSL
    import multiprocessing
    multiprocessing.freeze_support()
    main()