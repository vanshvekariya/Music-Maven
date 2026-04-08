"""
Demo script for PI.3 – RNN Beat & Downbeat Detection.
Generates report-ready figures showing beat tracking results.

Run:
    1. python generate_test_audio.py   (if not already done)
    2. python demo_pi3.py
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pathlib import Path
from app.beat_tracker import BeatTracker
from app.spectrogram import SpectrogramPipeline
import json


def plot_beat_tracking_result(result, audio_path, save_path=None, show=True):
    """Generate a 3-panel figure: waveform+beats, spectrogram+downbeats, activations.

    This is the key figure for the ISMIR report showing PI.3 results.
    """
    y, sr = librosa.load(audio_path, sr=22050, duration=30.0)
    duration = len(y) / sr
    time_axis = np.linspace(0, duration, len(y))

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), constrained_layout=True)
    fig.suptitle(
        f"Beat & Downbeat Tracking: {Path(audio_path).name}\n"
        f"BPM={result.bpm:.1f} | Meter={result.meter}/4 | "
        f"Confidence={result.confidence:.3f}",
        fontsize=13, fontweight="bold",
    )

    # ── Panel 1: Waveform with beat markers ──
    axes[0].plot(time_axis, y, linewidth=0.3, color="#90CAF9", alpha=0.7)
    # Mark beats as thin blue lines
    for bt in result.beats:
        if bt <= duration:
            axes[0].axvline(x=bt, color="#2196F3", linewidth=0.6, alpha=0.5)
    # Mark downbeats as thick red lines
    for db in result.downbeats:
        if db <= duration:
            axes[0].axvline(x=db, color="#F44336", linewidth=1.5, alpha=0.8)

    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Waveform with Beats (blue) and Downbeats (red)")
    axes[0].set_xlim(0, duration)
    axes[0].grid(True, alpha=0.2)

    # ── Panel 2: Spectrogram with downbeat markers ──
    mel_spec = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512),
        ref=np.max,
    )
    librosa.display.specshow(
        mel_spec, sr=sr, hop_length=512,
        x_axis="time", y_axis="mel", ax=axes[1], cmap="magma",
    )
    for db in result.downbeats:
        if db <= duration:
            axes[1].axvline(x=db, color="#00E676", linewidth=1.2, alpha=0.7)
    axes[1].set_title("Mel-Spectrogram with Downbeats (green)")

    # ── Panel 3: Beat activation function ──
    if result.beat_activations is not None and len(result.beat_activations) > 0:
        act = result.beat_activations
        act_time = np.linspace(0, duration, len(act))
        axes[2].plot(act_time, act, color="#FF9800", linewidth=0.8)
        axes[2].fill_between(act_time, 0, act, color="#FF9800", alpha=0.2)
        # Mark beats on activation plot too
        for bt in result.beats:
            if bt <= duration:
                axes[2].axvline(x=bt, color="#2196F3", linewidth=0.4, alpha=0.4)
        for db in result.downbeats:
            if db <= duration:
                axes[2].axvline(x=db, color="#F44336", linewidth=1.0, alpha=0.6)
        axes[2].set_ylabel("Activation")
        axes[2].set_title("Beat Activation Function (RNN output)")
        axes[2].set_xlim(0, duration)
        axes[2].grid(True, alpha=0.2)
    else:
        axes[2].text(0.5, 0.5, "No activation data available",
                     ha="center", va="center", transform=axes[2].transAxes)
        axes[2].set_title("Beat Activation Function")

    axes[2].set_xlabel("Time (s)")

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#2196F3", linewidth=1.5, label=f"Beats ({len(result.beats)})"),
        Line2D([0], [0], color="#F44336", linewidth=2.5, label=f"Downbeats ({len(result.downbeats)})"),
    ]
    axes[0].legend(handles=legend_elements, loc="upper right", fontsize=9)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  → Figure saved: {save_path}")

    if show:
        plt.show()
    plt.close(fig)


def plot_tempo_accuracy(results, ground_truths, save_path=None, show=True):
    """Bar chart comparing detected BPM vs ground truth for each test file."""
    names = [Path(r.file_path).stem for r in results]
    detected = [r.bpm for r in results]
    errors = [abs(d - gt) for d, gt in zip(detected, ground_truths)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle("Tempo Detection Accuracy", fontsize=14, fontweight="bold")

    x = np.arange(len(names))
    width = 0.35

    # Left panel: detected vs ground truth
    bars1 = ax1.bar(x - width/2, ground_truths, width, label="Ground Truth",
                     color="#4CAF50", alpha=0.8)
    bars2 = ax1.bar(x + width/2, detected, width, label="Detected",
                     color="#2196F3", alpha=0.8)
    ax1.set_xlabel("Test File")
    ax1.set_ylabel("BPM")
    ax1.set_title("Ground Truth vs Detected BPM")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax1.legend()
    ax1.grid(True, axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                 f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                 f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)

    # Right panel: absolute error
    colors = ["#4CAF50" if e < 5 else "#FF9800" if e < 15 else "#F44336" for e in errors]
    ax2.bar(x, errors, color=colors, alpha=0.8)
    ax2.set_xlabel("Test File")
    ax2.set_ylabel("Absolute Error (BPM)")
    ax2.set_title("Tempo Detection Error")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax2.axhline(y=5, color="gray", linestyle="--", alpha=0.5, label="5 BPM threshold")
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  → Figure saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    print("=" * 65)
    print("  Music Maven G3 – Beat & Downbeat Detection Demo (PI.3)")
    print("=" * 65)

    fig_dir = Path("output/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    tracker = BeatTracker()
    test_dir = Path("data/test_audio")

    if not test_dir.exists():
        print("ERROR: Run 'python generate_test_audio.py' first!")
        return

    # ── 1. Track all test files ──
    print("\n[1] Beat tracking all test files...")
    test_files = {
        "sine_120bpm.wav": 120,
        "drums_100bpm.wav": 100,
        "complex_128bpm.wav": 128,
        "slow_60bpm.wav": 60,
        "fast_180bpm.wav": 180,
    }

    results = []
    ground_truths = []
    print(f"\n  {'File':<25s} {'GT BPM':>7s} {'Det BPM':>8s} {'Error':>7s} "
          f"{'Beats':>6s} {'DBs':>5s} {'Meter':>6s} {'Conf':>6s} {'Time':>7s}")
    print("  " + "-" * 80)

    for fname, gt_bpm in test_files.items():
        path = test_dir / fname
        if not path.exists():
            continue

        result = tracker.track(str(path))
        results.append(result)
        ground_truths.append(gt_bpm)

        error = abs(result.bpm - gt_bpm)
        print(f"  {fname:<25s} {gt_bpm:>7.0f} {result.bpm:>8.1f} {error:>7.1f} "
              f"{len(result.beats):>6d} {len(result.downbeats):>5d} "
              f"{result.meter:>5d}/4 {result.confidence:>6.3f} {result.processing_time:>6.2f}s")

    # ── 2. Generate detailed figure for drums (best figure for the report) ──
    print("\n[2] Generating detailed beat tracking figure (drums)...")
    drums_result = [r for r in results if "drums" in r.file_path]
    if drums_result:
        plot_beat_tracking_result(
            drums_result[0],
            str(test_dir / "drums_100bpm.wav"),
            save_path=str(fig_dir / "beat_tracking_drums.png"),
            show=False,
        )

    # ── 3. Generate detailed figure for complex signal ──
    print("[3] Generating detailed beat tracking figure (complex)...")
    complex_result = [r for r in results if "complex" in r.file_path]
    if complex_result:
        plot_beat_tracking_result(
            complex_result[0],
            str(test_dir / "complex_128bpm.wav"),
            save_path=str(fig_dir / "beat_tracking_complex.png"),
            show=False,
        )

    # ── 4. Accuracy comparison chart ──
    print("[4] Generating tempo accuracy comparison chart...")
    if results:
        plot_tempo_accuracy(
            results, ground_truths,
            save_path=str(fig_dir / "tempo_accuracy.png"),
            show=False,
        )

    # ── 5. Save all results as JSON ──
    json_path = Path("output/beat_tracking_results.json")
    all_results = [r.to_dict() for r in results]
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[5] Results saved: {json_path}")

    # ── 6. Summary statistics ──
    if results:
        errors = [abs(r.bpm - gt) for r, gt in zip(results, ground_truths)]
        print(f"\n  === Summary ===")
        print(f"  Mean absolute error:  {np.mean(errors):.2f} BPM")
        print(f"  Max error:            {np.max(errors):.2f} BPM")
        print(f"  Files within 5 BPM:   {sum(1 for e in errors if e < 5)}/{len(errors)}")
        print(f"  Mean confidence:      {np.mean([r.confidence for r in results]):.4f}")
        print(f"  Mean processing time: {np.mean([r.processing_time for r in results]):.3f}s")

    print("\n" + "=" * 65)
    print("  Demo complete. Check output/figures/ for report images.")
    print("=" * 65)


if __name__ == "__main__":
    main()
