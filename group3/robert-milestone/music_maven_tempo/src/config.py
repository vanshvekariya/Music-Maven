from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass(slots=True)
class TempoConfig:
    dataset_dir: Path = Path("genres_original")
    cache_dir: Path = Path("cache")
    output_dir: Path = Path("outputs")
    plots_dir: Path = Path("outputs/plots")

    sample_rate: int = 22050
    mono: bool = True
    hop_length: int = 512
    n_fft: int = 2048

    bpm_min: float = 40.0
    bpm_max: float = 220.0
    local_window_seconds: float = 8.0
    local_hop_seconds: float = 4.0

    onset_pre_avg: int = 8
    onset_post_avg: int = 8
    onset_pre_max: int = 8
    onset_post_max: int = 8
    onset_delta: float = 0.15
    onset_wait: int = 3

    confidence_high_threshold: float = 0.75
    confidence_medium_threshold: float = 0.50
    ambiguous_std_threshold: float = 18.0
    fluctuating_std_threshold: float = 12.0

    preferred_extensions: Tuple[str, ...] = (".wav", ".mp3", ".au", ".ogg", ".flac")

    plot_dpi: int = 220
    histogram_bins: int = 30
    max_plot_tracks: int = 1000

    semantic_tempo_bins: Tuple[Tuple[float, str], ...] = field(
        default_factory=lambda: (
            (60.0, "very_slow"),
            (76.0, "slow"),
            (108.0, "moderate"),
            (132.0, "driving"),
            (168.0, "energetic"),
            (240.0, "very_fast"),
        )
    )

    def ensure_directories(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
