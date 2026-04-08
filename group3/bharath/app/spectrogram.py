"""
PI.2 – Mel-Spectrogram Pipeline
Converts audio files from the Music4All dataset into Mel-spectrograms
suitable for neural network input (RNN/TCN/Transformer models).

Usage:
    from app.spectrogram import SpectrogramPipeline
    pipeline = SpectrogramPipeline()
    mel_spec = pipeline.process("path/to/audio.mp3")
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import json
import time


@dataclass
class SpectrogramConfig:
    """Configuration for Mel-spectrogram generation.

    These parameters control the Short-Time Fourier Transform (STFT)
    and Mel filterbank used to create spectrograms.

    Attributes:
        sr: Target sample rate in Hz. Audio is resampled to this rate.
        n_fft: FFT window size. Larger = better frequency resolution.
        hop_length: Samples between STFT frames. Controls time resolution.
        n_mels: Number of Mel-frequency bands. More = finer frequency detail.
        fmin: Lowest frequency (Hz) for the Mel filterbank.
        fmax: Highest frequency (Hz) for the Mel filterbank.
        duration: Max duration in seconds to load. None = full track.
        normalize: Whether to apply per-spectrogram normalization.
    """
    sr: int = 22050
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    fmin: float = 20.0
    fmax: float = 8000.0
    duration: Optional[float] = 30.0  # Load first 30s by default
    normalize: bool = True


class SpectrogramPipeline:
    """Pipeline to convert audio files into Mel-spectrograms.

    This class handles the full audio → spectrogram workflow:
    1. Load and resample audio to a consistent sample rate
    2. Compute the Mel-spectrogram via STFT + Mel filterbank
    3. Convert power to log scale (dB)
    4. Optionally normalize for neural network input
    5. Save and visualize results
    """

    def __init__(self, config: Optional[SpectrogramConfig] = None):
        self.config = config or SpectrogramConfig()

    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load an audio file and resample to the target sample rate.

        Args:
            file_path: Path to an audio file (.mp3, .wav, .flac, etc.)

        Returns:
            Tuple of (audio waveform as 1D numpy array, sample rate)

        Raises:
            FileNotFoundError: If the audio file does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        y, sr = librosa.load(
            file_path,
            sr=self.config.sr,
            duration=self.config.duration,
            mono=True,
        )
        return y, sr

    def compute_mel_spectrogram(self, y: np.ndarray) -> np.ndarray:
        """Compute a log-scaled Mel-spectrogram from an audio waveform.

        The Mel scale approximates human pitch perception, making it
        well-suited for music analysis tasks. Log scaling (dB) compresses
        the dynamic range for neural network consumption.

        Args:
            y: Audio waveform (1D numpy array)

        Returns:
            Log-Mel spectrogram of shape (n_mels, time_frames)
        """
        # Compute Mel-spectrogram (power)
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.config.sr,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            fmin=self.config.fmin,
            fmax=self.config.fmax,
        )

        # Convert power to log scale (decibels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize to [0, 1] range for neural network input
        if self.config.normalize:
            mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (
                mel_spec_db.max() - mel_spec_db.min() + 1e-8
            )

        return mel_spec_db

    def process(self, file_path: str) -> dict:
        """Full pipeline: load audio → compute Mel-spectrogram → return metadata.

        Args:
            file_path: Path to the audio file.

        Returns:
            Dictionary containing:
                - mel_spectrogram: numpy array of shape (n_mels, time_frames)
                - waveform: raw audio waveform
                - metadata: dict with shape, duration, sample rate, etc.
        """
        start_time = time.time()

        # Step 1: Load audio
        y, sr = self.load_audio(file_path)
        duration = len(y) / sr

        # Step 2: Compute Mel-spectrogram
        mel_spec = self.compute_mel_spectrogram(y)

        elapsed = time.time() - start_time

        # Step 3: Package results
        metadata = {
            "file_path": str(file_path),
            "sample_rate": sr,
            "duration_seconds": round(duration, 2),
            "waveform_samples": len(y),
            "spectrogram_shape": list(mel_spec.shape),
            "n_mels": mel_spec.shape[0],
            "time_frames": mel_spec.shape[1],
            "processing_time_seconds": round(elapsed, 3),
            "config": {
                "sr": self.config.sr,
                "n_fft": self.config.n_fft,
                "hop_length": self.config.hop_length,
                "n_mels": self.config.n_mels,
                "fmin": self.config.fmin,
                "fmax": self.config.fmax,
                "normalized": self.config.normalize,
            },
        }

        return {
            "mel_spectrogram": mel_spec,
            "waveform": y,
            "metadata": metadata,
        }

    def visualize(
        self,
        result: dict,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """Generate a publication-quality figure with waveform and spectrogram.

        Creates a two-panel figure suitable for inclusion in an ISMIR paper:
        - Top: Audio waveform (amplitude vs. time)
        - Bottom: Log-Mel spectrogram (Mel bands vs. time)

        Args:
            result: Output dict from self.process()
            save_path: If provided, save figure to this path (e.g., "output.png")
            show: Whether to display the figure interactively.
        """
        y = result["waveform"]
        mel_spec = result["mel_spectrogram"]
        meta = result["metadata"]

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
        fig.suptitle(
            f"Audio Analysis: {Path(meta['file_path']).name}",
            fontsize=14,
            fontweight="bold",
        )

        # ── Top panel: Waveform ──
        time_axis = np.linspace(0, meta["duration_seconds"], len(y))
        axes[0].plot(time_axis, y, linewidth=0.4, color="#2196F3")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Amplitude")
        axes[0].set_title("Waveform")
        axes[0].set_xlim(0, meta["duration_seconds"])
        axes[0].grid(True, alpha=0.3)

        # ── Bottom panel: Mel-Spectrogram ──
        img = librosa.display.specshow(
            mel_spec if not meta["config"]["normalized"]
            else librosa.power_to_db(
                librosa.feature.melspectrogram(
                    y=y,
                    sr=meta["sample_rate"],
                    n_fft=meta["config"]["n_fft"],
                    hop_length=meta["config"]["hop_length"],
                    n_mels=meta["config"]["n_mels"],
                    fmin=meta["config"]["fmin"],
                    fmax=meta["config"]["fmax"],
                ),
                ref=np.max,
            ),
            sr=meta["sample_rate"],
            hop_length=meta["config"]["hop_length"],
            x_axis="time",
            y_axis="mel",
            ax=axes[1],
            cmap="magma",
        )
        axes[1].set_title("Log-Mel Spectrogram")
        fig.colorbar(img, ax=axes[1], format="%+2.0f dB", label="Power (dB)")

        # ── Annotation box ──
        info_text = (
            f"SR: {meta['sample_rate']} Hz  |  "
            f"FFT: {meta['config']['n_fft']}  |  "
            f"Hop: {meta['config']['hop_length']}  |  "
            f"Mels: {meta['config']['n_mels']}  |  "
            f"Shape: {mel_spec.shape}"
        )
        fig.text(
            0.5, -0.02, info_text,
            ha="center", fontsize=9, style="italic", color="gray"
        )

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Figure saved to: {save_path}")

        if show:
            plt.show()

        plt.close(fig)

    def batch_process(
        self,
        file_paths: List[str],
        output_dir: Optional[str] = None,
    ) -> List[dict]:
        """Process multiple audio files and optionally save spectrograms.

        Args:
            file_paths: List of audio file paths.
            output_dir: If provided, save .npy spectrogram files here.

        Returns:
            List of metadata dicts for each processed file.
        """
        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)

        results_meta = []
        for i, fp in enumerate(file_paths):
            print(f"  [{i+1}/{len(file_paths)}] Processing: {fp}")
            try:
                result = self.process(fp)
                meta = result["metadata"]

                if output_dir:
                    stem = Path(fp).stem
                    npy_path = out / f"{stem}_mel.npy"
                    np.save(npy_path, result["mel_spectrogram"])
                    meta["saved_to"] = str(npy_path)

                results_meta.append(meta)
            except Exception as e:
                print(f"    ✗ Error: {e}")
                results_meta.append({"file_path": fp, "error": str(e)})

        return results_meta
