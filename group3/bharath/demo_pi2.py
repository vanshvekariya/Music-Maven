"""
Demo script for PI.2 – Mel-Spectrogram Pipeline.
Generates report-ready figures and prints metadata.

Run:
    1. python generate_test_audio.py     (creates test .wav files)
    2. python demo_pi2.py               (processes them and generates figures)
"""

import json
import numpy as np
from pathlib import Path
from app.spectrogram import SpectrogramPipeline, SpectrogramConfig


def main():
    print("=" * 60)
    print("  Music Maven G3 – Mel-Spectrogram Pipeline Demo (PI.2)")
    print("=" * 60)

    # Create output directory for figures
    fig_dir = Path("output/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    pipeline = SpectrogramPipeline()
    test_dir = Path("data/test_audio")

    if not test_dir.exists():
        print("ERROR: Run 'python generate_test_audio.py' first!")
        return

    # ── 1. Process a single file with full visualization ──
    print("\n[1] Single file analysis: drums_100bpm.wav")
    result = pipeline.process(str(test_dir / "drums_100bpm.wav"))
    meta = result["metadata"]
    print(f"    Duration:    {meta['duration_seconds']}s")
    print(f"    Sample rate: {meta['sample_rate']} Hz")
    print(f"    Spec shape:  {meta['spectrogram_shape']}")
    print(f"    Time frames: {meta['time_frames']}")
    print(f"    Process time: {meta['processing_time_seconds']}s")

    # Save figure (for the ISMIR report)
    pipeline.visualize(result, save_path=str(fig_dir / "drums_spectrogram.png"), show=False)
    print(f"    → Figure saved: {fig_dir / 'drums_spectrogram.png'}")

    # ── 2. Process all test files ──
    print("\n[2] Batch processing all test audio files")
    wav_files = sorted(test_dir.glob("*.wav"))
    batch_results = pipeline.batch_process(
        [str(f) for f in wav_files],
        output_dir="output/spectrograms",
    )

    print("\n    Summary Table:")
    print(f"    {'File':<28s} {'Shape':<18s} {'Duration':>8s}  {'Time':>6s}")
    print("    " + "-" * 64)
    for meta in batch_results:
        if "error" not in meta:
            fname = Path(meta["file_path"]).name
            shape = f"{meta['spectrogram_shape']}"
            dur = f"{meta['duration_seconds']:.1f}s"
            t = f"{meta['processing_time_seconds']:.3f}s"
            print(f"    {fname:<28s} {shape:<18s} {dur:>8s}  {t:>6s}")

    # ── 3. Compare different configs ──
    print("\n[3] Config comparison (same audio, different parameters)")
    audio_file = str(test_dir / "complex_128bpm.wav")

    configs = [
        ("Default (128 mels)", SpectrogramConfig()),
        ("64 mels", SpectrogramConfig(n_mels=64)),
        ("256 mels", SpectrogramConfig(n_mels=256)),
        ("Large FFT (4096)", SpectrogramConfig(n_fft=4096)),
    ]

    print(f"    {'Config':<22s} {'Shape':<18s} {'Time':>8s}")
    print("    " + "-" * 50)
    for name, cfg in configs:
        p = SpectrogramPipeline(cfg)
        r = p.process(audio_file)
        m = r["metadata"]
        print(f"    {name:<22s} {str(m['spectrogram_shape']):<18s} {m['processing_time_seconds']:>7.3f}s")

    # ── 4. Generate comparison figure ──
    print("\n[4] Generating multi-file comparison figure...")

    import matplotlib.pyplot as plt
    import librosa.display

    files_to_compare = [
        ("slow_60bpm.wav", "60 BPM – Largo"),
        ("drums_100bpm.wav", "100 BPM – Moderato"),
        ("complex_128bpm.wav", "128 BPM – Allegro"),
        ("fast_180bpm.wav", "180 BPM – Presto"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    fig.suptitle("Mel-Spectrograms Across Different Tempos", fontsize=14, fontweight="bold")

    for ax, (fname, title) in zip(axes.flat, files_to_compare):
        r = pipeline.process(str(test_dir / fname))
        # Recompute unnormalized for proper dB display
        y = r["waveform"]
        mel_db = librosa.power_to_db(
            librosa.feature.melspectrogram(
                y=y, sr=22050, n_fft=2048, hop_length=512, n_mels=128
            ),
            ref=np.max,
        )
        librosa.display.specshow(
            mel_db, sr=22050, hop_length=512,
            x_axis="time", y_axis="mel", ax=ax, cmap="magma"
        )
        ax.set_title(title)

    fig.savefig(str(fig_dir / "tempo_comparison.png"), dpi=150, bbox_inches="tight")
    print(f"    → Figure saved: {fig_dir / 'tempo_comparison.png'}")

    # ── 5. Save metadata as JSON (for the report) ──
    json_path = Path("output/pipeline_results.json")
    serializable = []
    for meta in batch_results:
        serializable.append(meta)
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n[5] Metadata saved: {json_path}")

    print("\n" + "=" * 60)
    print("  Demo complete. Check output/figures/ for report images.")
    print("=" * 60)


if __name__ == "__main__":
    main()
