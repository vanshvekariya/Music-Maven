"""
PI.3 – RNN-Based Beat and Downbeat Detection
Uses pre-trained Recurrent Neural Network models from madmom to extract
beat positions and downbeat (bar-start) locations from audio.

Madmom provides pre-trained RNNs that output beat activation functions,
which are then decoded using Dynamic Bayesian Networks (DBNs) to produce
precise beat and downbeat timestamps.

Key concepts:
    - Beat: A regular pulse in music (e.g., every 0.5s at 120 BPM)
    - Downbeat: The first beat of each bar/measure (the "1" in 1-2-3-4)
    - Beat activation function: Neural network output indicating beat likelihood
    - DBN decoding: Probabilistic model that enforces temporal consistency

Usage:
    from app.beat_tracker import BeatTracker
    tracker = BeatTracker()
    result = tracker.track("path/to/audio.wav")

References:
    Böck et al. (2016) "Joint Beat and Downbeat Tracking with Recurrent
    Neural Networks" – ISMIR 2016
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple
import time
import json


@dataclass
class BeatTrackingResult:
    """Container for beat tracking analysis results.

    Attributes:
        file_path: Path to the analyzed audio file.
        beats: Array of beat timestamps in seconds.
        downbeats: Array of downbeat timestamps in seconds.
        beat_activations: Raw RNN activation function (beat probability over time).
        bpm: Estimated tempo from inter-beat intervals.
        meter: Inferred meter (e.g., 4 if 4/4 time).
        confidence: Confidence score based on activation strength.
        processing_time: Time taken for analysis in seconds.
    """
    file_path: str
    beats: np.ndarray
    downbeats: np.ndarray
    beat_activations: Optional[np.ndarray]
    bpm: float
    meter: int
    confidence: float
    processing_time: float

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dictionary."""
        return {
            "file_path": self.file_path,
            "beats": self.beats.tolist(),
            "downbeats": self.downbeats.tolist(),
            "num_beats": len(self.beats),
            "num_downbeats": len(self.downbeats),
            "bpm": round(self.bpm, 2),
            "meter": self.meter,
            "confidence": round(self.confidence, 4),
            "processing_time": round(self.processing_time, 3),
        }


class BeatTracker:
    """Beat and downbeat tracker using madmom's pre-trained RNN models.

    This class wraps madmom's RNNDownBeatProcessor and DBNDownBeatTrackingProcessor
    to provide a clean interface for extracting beat and downbeat positions.

    The pipeline is:
        Audio → RNN (pre-trained) → Beat activation function → DBN decoder → Beat/Downbeat times

    The RNN model was trained on large annotated datasets and generalizes
    across genres. The DBN decoder enforces musical constraints (consistent
    tempo, regular meter) to produce musically plausible results.
    """

    def __init__(self, fps: int = 100, beats_per_bar: Optional[List[int]] = None):
        """Initialize the beat tracker.

        Args:
            fps: Frames per second for the activation function. Higher = finer
                 temporal resolution but slower processing. Default 100 is standard.
            beats_per_bar: Allowed meter configurations. Default [3,4] allows
                          both 3/4 and 4/4 time signatures.
        """
        self.fps = fps
        self.beats_per_bar = beats_per_bar or [3, 4]
        self._processor = None
        self._decoder = None
        self._init_models()

    def _init_models(self):
        """Initialize madmom processors (lazy import to handle missing dependency)."""
        try:
            from madmom.features.downbeats import (
                RNNDownBeatProcessor,
                DBNDownBeatTrackingProcessor,
            )

            # RNN processor: extracts beat activation function from audio
            # Uses pre-trained bidirectional LSTM networks
            self._processor = RNNDownBeatProcessor(fps=self.fps)

            # DBN decoder: converts activations to discrete beat/downbeat positions
            # Uses a Dynamic Bayesian Network with tempo and meter state variables
            self._decoder = DBNDownBeatTrackingProcessor(
                beats_per_bar=self.beats_per_bar,
                fps=self.fps,
            )

            self._available = True
            print("✓ Madmom RNN beat tracker initialized successfully")

        except ImportError:
            self._available = False
            print("✗ Madmom not available – using librosa fallback")

    def _track_with_madmom(self, file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run madmom's RNN + DBN pipeline.

        Args:
            file_path: Path to audio file.

        Returns:
            Tuple of (beats, downbeats, activations)
            - beats: all beat timestamps
            - downbeats: only downbeat (bar-start) timestamps
            - activations: raw RNN output
        """
        # Step 1: Run RNN to get beat activation function
        # The RNN outputs a 2D array: (time_frames, num_beat_positions)
        # Column 0 = downbeat probability, Column 1+ = other beat positions
        activations = self._processor(file_path)

        # Step 2: Decode activations with DBN
        # Returns array of (time, beat_position) pairs
        # beat_position=1 means downbeat, 2/3/4 = other beats in the bar
        beat_info = self._decoder(activations)

        # Extract all beats and downbeats separately
        all_beats = beat_info[:, 0]  # All beat timestamps
        beat_positions = beat_info[:, 1]  # 1=downbeat, 2,3,4=other beats
        downbeats = all_beats[beat_positions == 1]  # Only the "1"s

        # Sum activation columns for a 1D activation function
        activation_1d = activations.sum(axis=1) if activations.ndim > 1 else activations

        return all_beats, downbeats, activation_1d

    def _track_with_librosa(self, file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fallback: use librosa for basic beat tracking (no downbeat distinction).

        This is less accurate than madmom but works without the dependency.
        Librosa uses onset strength + dynamic programming for beat tracking.
        For signals with weak onsets (e.g., pure tones), we fall back to
        autocorrelation-based tempo estimation with synthetic beat placement.
        """
        import librosa

        y, sr = librosa.load(file_path, sr=22050, mono=True)
        duration = len(y) / sr

        # Onset strength as a proxy for activation function
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)

        # Get tempo and beat frames
        tempo, beat_frames = librosa.beat.beat_track(
            y=y, sr=sr, units="frames", onset_envelope=onset_env
        )

        # Handle newer librosa returning tempo as array
        if hasattr(tempo, '__len__'):
            tempo = float(tempo[0]) if len(tempo) > 0 else 0.0
        else:
            tempo = float(tempo)

        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        # If beat tracking returned no beats but we have a tempo estimate,
        # generate synthetic beat positions from the tempo.
        # This handles pure tones and signals with weak transients.
        if len(beat_times) == 0 and tempo > 0:
            beat_interval = 60.0 / tempo
            beat_times = np.arange(beat_interval, duration, beat_interval)

        # If we still have no tempo, use autocorrelation as last resort
        if len(beat_times) == 0:
            tempo_fallback = librosa.beat.tempo(
                onset_envelope=onset_env, sr=sr
            )
            if hasattr(tempo_fallback, '__len__'):
                tempo_fallback = float(tempo_fallback[0]) if len(tempo_fallback) > 0 else 120.0
            else:
                tempo_fallback = float(tempo_fallback) if tempo_fallback > 0 else 120.0

            beat_interval = 60.0 / tempo_fallback
            beat_times = np.arange(beat_interval, duration, beat_interval)

        # Approximate downbeats: every 4th beat (assumes 4/4)
        downbeats = beat_times[::4] if len(beat_times) >= 4 else beat_times[:1]

        return beat_times, downbeats, onset_env

    def track(self, file_path: str) -> BeatTrackingResult:
        """Analyze an audio file for beats and downbeats.

        Automatically uses madmom if available, otherwise falls back to librosa.

        Args:
            file_path: Path to the audio file (.wav, .mp3, .flac)

        Returns:
            BeatTrackingResult with beats, downbeats, BPM, meter, etc.

        Raises:
            FileNotFoundError: If the audio file doesn't exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        start_time = time.time()

        # Run tracking
        if self._available:
            beats, downbeats, activations = self._track_with_madmom(str(path))
        else:
            beats, downbeats, activations = self._track_with_librosa(str(path))

        elapsed = time.time() - start_time

        # Compute BPM from inter-beat intervals
        bpm = self._estimate_bpm(beats)

        # Infer meter from beats-per-bar
        meter = self._infer_meter(beats, downbeats)

        # Confidence: mean peak activation strength
        confidence = self._compute_confidence(activations)

        return BeatTrackingResult(
            file_path=str(file_path),
            beats=beats,
            downbeats=downbeats,
            beat_activations=activations,
            bpm=bpm,
            meter=meter,
            confidence=confidence,
            processing_time=elapsed,
        )

    def _estimate_bpm(self, beats: np.ndarray) -> float:
        """Estimate BPM from the median inter-beat interval.

        Using the median (not mean) is more robust to outliers from
        tracking errors at the start/end of the audio.
        """
        if len(beats) < 2:
            return 0.0
        ibis = np.diff(beats)  # Inter-beat intervals
        median_ibi = np.median(ibis)
        if median_ibi > 0:
            return 60.0 / median_ibi
        return 0.0

    def _infer_meter(self, beats: np.ndarray, downbeats: np.ndarray) -> int:
        """Infer the meter (beats per bar) from beat and downbeat counts.

        If we have N beats and M downbeats, the meter is approximately N/M.
        """
        if len(downbeats) < 2:
            return 4  # Default assumption
        # Average beats between consecutive downbeats
        beats_per_bar_estimates = []
        for i in range(len(downbeats) - 1):
            bar_beats = np.sum(
                (beats >= downbeats[i]) & (beats < downbeats[i + 1])
            )
            beats_per_bar_estimates.append(bar_beats)
        if beats_per_bar_estimates:
            return int(round(np.median(beats_per_bar_estimates)))
        return 4

    def _compute_confidence(self, activations: np.ndarray) -> float:
        """Compute a confidence score from the activation function.

        Higher mean activation with clear peaks = higher confidence.
        Score is normalized to [0, 1].
        """
        if len(activations) == 0:
            return 0.0
        # Ratio of peak energy to total energy (peakiness)
        peak_threshold = np.percentile(activations, 90)
        peak_ratio = np.mean(activations > peak_threshold)
        # Normalize: more concentrated peaks = higher confidence
        normalized = min(1.0, np.max(activations) / (np.mean(activations) + 1e-8) / 10.0)
        return float(np.clip((normalized + peak_ratio) / 2, 0, 1))

    def batch_track(self, file_paths: List[str]) -> List[BeatTrackingResult]:
        """Process multiple audio files.

        Args:
            file_paths: List of audio file paths.

        Returns:
            List of BeatTrackingResult objects.
        """
        results = []
        for i, fp in enumerate(file_paths):
            print(f"  [{i+1}/{len(file_paths)}] Tracking: {Path(fp).name}")
            try:
                result = self.track(fp)
                results.append(result)
                print(f"    → BPM={result.bpm:.1f}, Beats={len(result.beats)}, "
                      f"Downbeats={len(result.downbeats)}, Meter={result.meter}/4")
            except Exception as e:
                print(f"    ✗ Error: {e}")
        return results
