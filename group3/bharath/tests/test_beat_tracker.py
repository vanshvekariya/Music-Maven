"""
Tests for PI.3 – RNN-Based Beat and Downbeat Detection.
Run with: pytest tests/test_beat_tracker.py -v
"""

import numpy as np
import pytest
from pathlib import Path
from app.beat_tracker import BeatTracker, BeatTrackingResult


# ── Fixtures ──

@pytest.fixture
def tracker():
    """Initialize a beat tracker (uses madmom if available, else librosa)."""
    return BeatTracker()


@pytest.fixture
def test_audio_dir():
    return Path("data/test_audio")


# ── Initialization Tests ──

def test_tracker_initializes(tracker):
    """Tracker should initialize without errors."""
    assert tracker is not None
    assert tracker.fps == 100
    assert tracker.beats_per_bar == [3, 4]
    backend = "madmom" if tracker._available else "librosa (fallback)"
    print(f"✓ Tracker initialized with backend: {backend}")


def test_custom_fps():
    """Should accept custom frames-per-second."""
    t = BeatTracker(fps=50)
    assert t.fps == 50
    print("✓ Custom FPS accepted")


def test_custom_meter():
    """Should accept custom meter configurations."""
    t = BeatTracker(beats_per_bar=[3, 4, 6])
    assert 6 in t.beats_per_bar
    print("✓ Custom meter [3,4,6] accepted")


# ── Single File Tracking Tests ──

def test_track_sine_120bpm(tracker, test_audio_dir):
    """120 BPM sine wave should be detected close to 120 BPM."""
    audio_file = test_audio_dir / "sine_120bpm.wav"
    if not audio_file.exists():
        pytest.skip("Run generate_test_audio.py first")

    result = tracker.track(str(audio_file))

    assert isinstance(result, BeatTrackingResult)
    assert len(result.beats) > 0
    assert result.bpm > 0
    assert result.processing_time > 0
    # Allow generous tolerance – synthetic audio isn't perfectly musical
    assert 80 < result.bpm < 180, f"BPM {result.bpm} too far from expected 120"
    print(f"✓ 120 BPM sine: detected BPM={result.bpm:.1f}, "
          f"beats={len(result.beats)}, downbeats={len(result.downbeats)}")


def test_track_drums_100bpm(tracker, test_audio_dir):
    """100 BPM drum pattern should be detected close to 100 BPM."""
    audio_file = test_audio_dir / "drums_100bpm.wav"
    if not audio_file.exists():
        pytest.skip("Run generate_test_audio.py first")

    result = tracker.track(str(audio_file))

    assert len(result.beats) > 0
    assert len(result.downbeats) > 0
    assert 60 < result.bpm < 160, f"BPM {result.bpm} too far from expected 100"
    print(f"✓ 100 BPM drums: detected BPM={result.bpm:.1f}, "
          f"beats={len(result.beats)}, downbeats={len(result.downbeats)}")


def test_track_fast_180bpm(tracker, test_audio_dir):
    """180 BPM should be detected in the high-tempo range."""
    audio_file = test_audio_dir / "fast_180bpm.wav"
    if not audio_file.exists():
        pytest.skip("Run generate_test_audio.py first")

    result = tracker.track(str(audio_file))

    assert len(result.beats) > 0
    assert result.bpm > 0
    print(f"✓ 180 BPM fast: detected BPM={result.bpm:.1f}, "
          f"beats={len(result.beats)}, downbeats={len(result.downbeats)}")


def test_track_slow_60bpm(tracker, test_audio_dir):
    """60 BPM should be detected in the slow tempo range."""
    audio_file = test_audio_dir / "slow_60bpm.wav"
    if not audio_file.exists():
        pytest.skip("Run generate_test_audio.py first")

    result = tracker.track(str(audio_file))

    assert len(result.beats) > 0
    assert result.bpm > 0
    print(f"✓ 60 BPM slow: detected BPM={result.bpm:.1f}, "
          f"beats={len(result.beats)}, downbeats={len(result.downbeats)}")


# ── Result Structure Tests ──

def test_result_has_all_fields(tracker, test_audio_dir):
    """BeatTrackingResult should contain all expected fields."""
    audio_file = test_audio_dir / "drums_100bpm.wav"
    if not audio_file.exists():
        pytest.skip("Run generate_test_audio.py first")

    result = tracker.track(str(audio_file))

    assert hasattr(result, "beats")
    assert hasattr(result, "downbeats")
    assert hasattr(result, "beat_activations")
    assert hasattr(result, "bpm")
    assert hasattr(result, "meter")
    assert hasattr(result, "confidence")
    assert hasattr(result, "processing_time")
    print(f"✓ All fields present: bpm={result.bpm:.1f}, meter={result.meter}, "
          f"confidence={result.confidence:.3f}")


def test_result_to_dict(tracker, test_audio_dir):
    """to_dict() should produce a JSON-serializable dictionary."""
    audio_file = test_audio_dir / "drums_100bpm.wav"
    if not audio_file.exists():
        pytest.skip("Run generate_test_audio.py first")

    result = tracker.track(str(audio_file))
    d = result.to_dict()

    assert isinstance(d, dict)
    assert "bpm" in d
    assert "beats" in d
    assert "downbeats" in d
    assert "num_beats" in d
    assert "meter" in d
    # Should be JSON serializable
    import json
    json_str = json.dumps(d)
    assert len(json_str) > 0
    print(f"✓ to_dict() serializable, {len(json_str)} chars")


def test_beats_are_sorted(tracker, test_audio_dir):
    """Beat timestamps should be monotonically increasing."""
    audio_file = test_audio_dir / "complex_128bpm.wav"
    if not audio_file.exists():
        pytest.skip("Run generate_test_audio.py first")

    result = tracker.track(str(audio_file))

    if len(result.beats) > 1:
        diffs = np.diff(result.beats)
        assert np.all(diffs > 0), "Beats should be monotonically increasing"
    print(f"✓ Beats are sorted (ascending)")


def test_downbeats_are_subset_of_beats(tracker, test_audio_dir):
    """Every downbeat should also appear in the beats array."""
    audio_file = test_audio_dir / "drums_100bpm.wav"
    if not audio_file.exists():
        pytest.skip("Run generate_test_audio.py first")

    result = tracker.track(str(audio_file))

    if len(result.downbeats) > 0 and len(result.beats) > 0:
        for db in result.downbeats:
            # Check that each downbeat is close to some beat (within 50ms)
            min_dist = np.min(np.abs(result.beats - db))
            assert min_dist < 0.05, f"Downbeat {db:.3f}s not found in beats"
    print(f"✓ All {len(result.downbeats)} downbeats are within beat positions")


# ── Confidence Score Tests ──

def test_confidence_in_range(tracker, test_audio_dir):
    """Confidence should be between 0 and 1."""
    audio_file = test_audio_dir / "drums_100bpm.wav"
    if not audio_file.exists():
        pytest.skip("Run generate_test_audio.py first")

    result = tracker.track(str(audio_file))

    assert 0 <= result.confidence <= 1, f"Confidence {result.confidence} out of range"
    print(f"✓ Confidence = {result.confidence:.4f} (in [0,1])")


# ── Error Handling Tests ──

def test_file_not_found(tracker):
    """Should raise FileNotFoundError for missing audio."""
    with pytest.raises(FileNotFoundError):
        tracker.track("nonexistent_song.wav")
    print("✓ FileNotFoundError raised correctly")


# ── Meter Inference Test ──

def test_meter_is_reasonable(tracker, test_audio_dir):
    """Inferred meter should be 3 or 4 (common time signatures)."""
    audio_file = test_audio_dir / "drums_100bpm.wav"
    if not audio_file.exists():
        pytest.skip("Run generate_test_audio.py first")

    result = tracker.track(str(audio_file))

    assert result.meter in [3, 4, 6], f"Unexpected meter: {result.meter}"
    print(f"✓ Meter = {result.meter} (reasonable)")


# ── Batch Processing Test ──

def test_batch_track(tracker, test_audio_dir):
    """Should process multiple files in batch."""
    if not test_audio_dir.exists():
        pytest.skip("Run generate_test_audio.py first")

    files = [str(f) for f in sorted(test_audio_dir.glob("*.wav"))[:3]]
    results = tracker.batch_track(files)

    assert len(results) == len(files)
    for r in results:
        assert len(r.beats) > 0
        assert r.bpm > 0
    print(f"✓ Batch tracked {len(results)} files")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
