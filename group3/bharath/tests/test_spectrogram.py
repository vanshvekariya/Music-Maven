"""
Tests for PI.2 – Mel-Spectrogram Pipeline.
Run with: pytest tests/test_spectrogram.py -v
"""

import numpy as np
import pytest
from pathlib import Path
from app.spectrogram import SpectrogramPipeline, SpectrogramConfig


# ── Fixtures ──

@pytest.fixture
def pipeline():
    """Default pipeline with standard config."""
    return SpectrogramPipeline()


@pytest.fixture
def short_pipeline():
    """Pipeline configured for short audio clips."""
    config = SpectrogramConfig(duration=5.0, n_mels=64)
    return SpectrogramPipeline(config)


@pytest.fixture
def test_audio_dir():
    """Path to test audio files. Run generate_test_audio.py first."""
    return Path("data/test_audio")


# ── Config Tests ──

def test_default_config():
    """Default config should have standard MIR values."""
    config = SpectrogramConfig()
    assert config.sr == 22050
    assert config.n_fft == 2048
    assert config.hop_length == 512
    assert config.n_mels == 128
    assert config.normalize is True
    print(f"✓ Default config: sr={config.sr}, n_mels={config.n_mels}")


def test_custom_config():
    """Custom config values should override defaults."""
    config = SpectrogramConfig(sr=44100, n_mels=64, duration=10.0)
    assert config.sr == 44100
    assert config.n_mels == 64
    assert config.duration == 10.0
    print(f"✓ Custom config: sr={config.sr}, n_mels={config.n_mels}")


# ── Audio Loading Tests ──

def test_load_audio(pipeline, test_audio_dir):
    """Should load a WAV file and return correct sample rate."""
    audio_file = test_audio_dir / "sine_120bpm.wav"
    if not audio_file.exists():
        pytest.skip("Run generate_test_audio.py first")

    y, sr = pipeline.load_audio(str(audio_file))
    assert isinstance(y, np.ndarray)
    assert y.ndim == 1  # Mono
    assert sr == 22050
    assert len(y) > 0
    print(f"✓ Loaded audio: {len(y)} samples, sr={sr}")


def test_load_audio_file_not_found(pipeline):
    """Should raise FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        pipeline.load_audio("nonexistent_file.mp3")
    print("✓ FileNotFoundError raised for missing file")


# ── Spectrogram Tests ──

def test_mel_spectrogram_shape(pipeline, test_audio_dir):
    """Spectrogram should have shape (n_mels, time_frames)."""
    audio_file = test_audio_dir / "sine_120bpm.wav"
    if not audio_file.exists():
        pytest.skip("Run generate_test_audio.py first")

    y, sr = pipeline.load_audio(str(audio_file))
    mel_spec = pipeline.compute_mel_spectrogram(y)

    assert mel_spec.ndim == 2
    assert mel_spec.shape[0] == 128  # n_mels
    assert mel_spec.shape[1] > 0     # time frames
    print(f"✓ Spectrogram shape: {mel_spec.shape}")


def test_mel_spectrogram_normalized(pipeline, test_audio_dir):
    """When normalized, values should be in [0, 1] range."""
    audio_file = test_audio_dir / "sine_120bpm.wav"
    if not audio_file.exists():
        pytest.skip("Run generate_test_audio.py first")

    y, _ = pipeline.load_audio(str(audio_file))
    mel_spec = pipeline.compute_mel_spectrogram(y)

    assert mel_spec.min() >= 0.0, f"Min value {mel_spec.min()} < 0"
    assert mel_spec.max() <= 1.0, f"Max value {mel_spec.max()} > 1"
    print(f"✓ Normalized range: [{mel_spec.min():.4f}, {mel_spec.max():.4f}]")


def test_mel_spectrogram_unnormalized(test_audio_dir):
    """Without normalization, values should be in dB scale (negative)."""
    config = SpectrogramConfig(normalize=False)
    pipeline = SpectrogramPipeline(config)

    audio_file = test_audio_dir / "sine_120bpm.wav"
    if not audio_file.exists():
        pytest.skip("Run generate_test_audio.py first")

    y, _ = pipeline.load_audio(str(audio_file))
    mel_spec = pipeline.compute_mel_spectrogram(y)

    assert mel_spec.max() <= 0.0  # dB relative to max, so max is 0
    print(f"✓ Unnormalized dB range: [{mel_spec.min():.1f}, {mel_spec.max():.1f}]")


def test_custom_n_mels(test_audio_dir):
    """Changing n_mels should change the spectrogram height."""
    config = SpectrogramConfig(n_mels=64)
    pipeline = SpectrogramPipeline(config)

    audio_file = test_audio_dir / "sine_120bpm.wav"
    if not audio_file.exists():
        pytest.skip("Run generate_test_audio.py first")

    y, _ = pipeline.load_audio(str(audio_file))
    mel_spec = pipeline.compute_mel_spectrogram(y)
    assert mel_spec.shape[0] == 64
    print(f"✓ Custom n_mels=64, shape: {mel_spec.shape}")


# ── Full Pipeline Tests ──

def test_full_process(pipeline, test_audio_dir):
    """process() should return spectrogram, waveform, and metadata."""
    audio_file = test_audio_dir / "drums_100bpm.wav"
    if not audio_file.exists():
        pytest.skip("Run generate_test_audio.py first")

    result = pipeline.process(str(audio_file))

    assert "mel_spectrogram" in result
    assert "waveform" in result
    assert "metadata" in result

    meta = result["metadata"]
    assert meta["sample_rate"] == 22050
    assert meta["n_mels"] == 128
    assert meta["time_frames"] > 0
    assert meta["processing_time_seconds"] > 0
    print(f"✓ Full pipeline: shape={meta['spectrogram_shape']}, "
          f"time={meta['processing_time_seconds']:.3f}s")


def test_process_all_test_files(pipeline, test_audio_dir):
    """Should process all generated test files without errors."""
    if not test_audio_dir.exists():
        pytest.skip("Run generate_test_audio.py first")

    files = list(test_audio_dir.glob("*.wav"))
    assert len(files) > 0, "No test audio files found"

    for f in files:
        result = pipeline.process(str(f))
        mel = result["mel_spectrogram"]
        assert mel.shape[0] == 128
        assert mel.shape[1] > 0
        print(f"  ✓ {f.name}: shape={mel.shape}")

    print(f"✓ All {len(files)} files processed successfully")


# ── Batch Processing Tests ──

def test_batch_process(pipeline, test_audio_dir, tmp_path):
    """Batch processing should handle multiple files and save .npy outputs."""
    if not test_audio_dir.exists():
        pytest.skip("Run generate_test_audio.py first")

    files = [str(f) for f in test_audio_dir.glob("*.wav")][:3]
    output_dir = str(tmp_path / "spectrograms")

    results = pipeline.batch_process(files, output_dir=output_dir)

    assert len(results) == len(files)
    for meta in results:
        assert "error" not in meta
        assert "saved_to" in meta
        # Verify .npy file was created and is loadable
        saved = np.load(meta["saved_to"])
        assert saved.shape[0] == 128
        print(f"  ✓ Saved: {meta['saved_to']} shape={saved.shape}")

    print(f"✓ Batch processed and saved {len(files)} spectrograms")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
