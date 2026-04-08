"""
Generate synthetic test audio files for testing the spectrogram pipeline.
These simulate simple musical signals without needing the full Music4All dataset.

Run:  python generate_test_audio.py
"""

import numpy as np
import soundfile as sf
from pathlib import Path


def generate_sine_beat(
    bpm: float = 120,
    duration: float = 10.0,
    sr: int = 22050,
    freq: float = 440.0,
) -> np.ndarray:
    """Generate a sine wave with rhythmic amplitude modulation (fake beats)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Base tone
    tone = np.sin(2 * np.pi * freq * t)

    # Beat envelope: amplitude peaks at each beat
    beat_interval = 60.0 / bpm
    envelope = np.zeros_like(t)
    for beat_time in np.arange(0, duration, beat_interval):
        # Gaussian pulse at each beat
        envelope += np.exp(-((t - beat_time) ** 2) / (0.01))

    envelope = envelope / envelope.max()
    return (tone * envelope).astype(np.float32)


def generate_drum_pattern(
    bpm: float = 100,
    duration: float = 10.0,
    sr: int = 22050,
) -> np.ndarray:
    """Generate a simple kick-snare pattern using noise bursts."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = np.zeros_like(t)
    beat_interval = 60.0 / bpm

    for i, beat_time in enumerate(np.arange(0, duration, beat_interval)):
        idx = int(beat_time * sr)
        burst_len = int(0.05 * sr)  # 50ms burst
        if idx + burst_len > len(signal):
            break

        if i % 2 == 0:
            # Kick: low frequency sine burst
            burst_t = np.linspace(0, 0.05, burst_len)
            burst = np.sin(2 * np.pi * 80 * burst_t) * np.exp(-burst_t * 40)
        else:
            # Snare: noise burst
            burst = np.random.randn(burst_len) * 0.3
            burst *= np.exp(-np.linspace(0, 5, burst_len))

        signal[idx:idx + burst_len] += burst

    signal = signal / (np.abs(signal).max() + 1e-8)
    return signal.astype(np.float32)


def generate_complex_signal(
    bpm: float = 128,
    duration: float = 15.0,
    sr: int = 22050,
) -> np.ndarray:
    """Generate a more complex signal with multiple frequencies and beats."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Chord (multiple frequencies)
    chord = (
        0.4 * np.sin(2 * np.pi * 261.63 * t)   # C4
        + 0.3 * np.sin(2 * np.pi * 329.63 * t)  # E4
        + 0.3 * np.sin(2 * np.pi * 392.00 * t)  # G4
    )

    # Rhythmic envelope
    beat_interval = 60.0 / bpm
    envelope = np.zeros_like(t)
    for beat_time in np.arange(0, duration, beat_interval):
        envelope += np.exp(-((t - beat_time) ** 2) / 0.005)
    envelope = envelope / envelope.max()

    # Add some noise for realism
    noise = np.random.randn(len(t)) * 0.02

    signal = chord * envelope + noise
    signal = signal / (np.abs(signal).max() + 1e-8)
    return signal.astype(np.float32)


def main():
    output_dir = Path("data/test_audio")
    output_dir.mkdir(parents=True, exist_ok=True)

    sr = 22050

    test_files = [
        ("sine_120bpm.wav", generate_sine_beat(bpm=120, sr=sr)),
        ("drums_100bpm.wav", generate_drum_pattern(bpm=100, sr=sr)),
        ("complex_128bpm.wav", generate_complex_signal(bpm=128, sr=sr)),
        ("slow_60bpm.wav", generate_sine_beat(bpm=60, freq=220, sr=sr)),
        ("fast_180bpm.wav", generate_sine_beat(bpm=180, freq=660, sr=sr)),
    ]

    print("Generating test audio files...")
    print(f"Output directory: {output_dir.resolve()}\n")

    for filename, audio in test_files:
        path = output_dir / filename
        sf.write(str(path), audio, sr)
        duration = len(audio) / sr
        print(f"  ✓ {filename:25s}  duration={duration:.1f}s  samples={len(audio)}")

    print(f"\nDone! Generated {len(test_files)} test files.")


if __name__ == "__main__":
    main()
