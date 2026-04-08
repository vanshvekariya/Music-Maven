from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd

from .audio_io import duration_seconds, file_metadata, infer_genre, load_audio
from .caching import TrackCache
from .config import TempoConfig
from .confidence import compute_confidence_bundle
from .plots import save_genre_boxplot, save_histogram, save_onset_plot, save_scatter
from .tempo_methods import collect_estimates, local_tempo_profile, weighted_consensus


def semantic_tempo_class(bpm: float, config: TempoConfig) -> str:
    if not np.isfinite(bpm):
        return "unknown"
    for threshold, label in config.semantic_tempo_bins:
        if bpm < threshold:
            return label
    return config.semantic_tempo_bins[-1][1]


def simple_meter_guess(beats: np.ndarray) -> str:
    if beats is None or len(beats) < 8:
        return "unknown"
    n = len(beats)
    if n % 3 == 0 and n % 4 != 0:
        return "3/4_candidate"
    if n % 6 == 0:
        return "6/8_candidate"
    return "4/4_candidate"


def onset_statistics(onset_env: np.ndarray, sr: int, hop_length: int) -> Dict[str, float]:
    frames_per_second = sr / hop_length
    peaks = np.where(onset_env >= np.percentile(onset_env, 90))[0]
    density = float(len(peaks) / max(len(onset_env) / frames_per_second, 1e-6))
    return {
        "onset_mean": float(np.mean(onset_env)),
        "onset_std": float(np.std(onset_env)),
        "onset_peak_ratio": float(np.percentile(onset_env, 95) / (np.mean(onset_env) + 1e-9)),
        "onset_density": density,
    }


def analyze_track(path: Path, config: TempoConfig, cache: Optional[TrackCache] = None, dataset_root: Optional[Path] = None) -> Dict[str, Any]:
    if cache is not None:
        cached = cache.load(path)
        if cached is not None:
            cached["cache_hit"] = True
            return cached

    y, sr = load_audio(path=path, sample_rate=config.sample_rate, mono=config.mono)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=config.hop_length)

    estimates, beats = collect_estimates(
        y=y,
        sr=sr,
        onset_env=onset_env,
        hop_length=config.hop_length,
        bpm_min=config.bpm_min,
        bpm_max=config.bpm_max,
        peak_kwargs={
            "pre_max": config.onset_pre_max,
            "post_max": config.onset_post_max,
            "pre_avg": config.onset_pre_avg,
            "post_avg": config.onset_post_avg,
            "delta": config.onset_delta,
            "wait": config.onset_wait,
        },
    )
    consensus = weighted_consensus(estimates, bpm_min=config.bpm_min, bpm_max=config.bpm_max)
    local_tempi = local_tempo_profile(onset_env, sr=sr, hop_length=config.hop_length, bpm_max=config.bpm_max)
    confidence_bundle = compute_confidence_bundle(
        estimates=estimates,
        consensus_bpm=consensus,
        beats=beats,
        sr=sr,
        hop_length=config.hop_length,
        onset_env=onset_env,
        local_tempi=local_tempi,
        high_threshold=config.confidence_high_threshold,
        medium_threshold=config.confidence_medium_threshold,
        ambiguous_std_threshold=config.ambiguous_std_threshold,
        fluctuating_std_threshold=config.fluctuating_std_threshold,
    )

    meta = file_metadata(path)
    result: Dict[str, Any] = {
        "song": path.name,
        "track_path": str(path),
        "genre": infer_genre(path, dataset_root=dataset_root),
        "duration_seconds": duration_seconds(y, sr),
        "sample_rate": sr,
        "tempo_librosa": estimates[0].bpm,
        "tempo_autocorr": estimates[1].bpm,
        "tempo_tempogram": estimates[2].bpm,
        "tempo_onset_peaks": estimates[3].bpm,
        "tempo_consensus": consensus,
        "semantic_tempo_class": semantic_tempo_class(consensus, config),
        "meter_guess": simple_meter_guess(beats),
        "n_beats": int(len(beats)) if beats is not None else 0,
        "cache_hit": False,
        **onset_statistics(onset_env, sr=sr, hop_length=config.hop_length),
        **confidence_bundle,
        "file_size_bytes": meta["size"],
        "file_mtime": meta["mtime"],
    }

    if cache is not None:
        cache.save(path, result)
    return result


def iter_audio_files(dataset_dir: Path, extensions: Tuple[str, ...]) -> Iterable[Path]:
    for path in sorted(dataset_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


def analyze_dataset(config: TempoConfig, save_plots: bool = True, force_recompute: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    config.ensure_directories()
    cache = TrackCache(config.cache_dir)

    results: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []

    first_onset_saved = False

    for audio_path in iter_audio_files(config.dataset_dir, config.preferred_extensions):
        try:
            track_cache = None if force_recompute else cache
            result = analyze_track(audio_path, config=config, cache=track_cache, dataset_root=config.dataset_dir)
            results.append(result)

            if save_plots and not first_onset_saved:
                y, sr = load_audio(path=audio_path, sample_rate=config.sample_rate, mono=config.mono)
                onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=config.hop_length)
                save_onset_plot(
                    onset_env,
                    config.plots_dir / "onset_example.png",
                    f"Onset Strength Envelope Example: {audio_path.name}",
                )
                first_onset_saved = True
        except Exception as exc:
            failed.append(
                {
                    "song": audio_path.name,
                    "track_path": str(audio_path),
                    "genre": infer_genre(audio_path, dataset_root=config.dataset_dir),
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=2),
                }
            )

    results_df = pd.DataFrame(results)
    failed_df = pd.DataFrame(failed)

    if not results_df.empty:
        results_df = results_df.sort_values(["genre", "song"]).reset_index(drop=True)
        results_df.to_csv(config.output_dir / "tempo_results_full.csv", index=False)

        summary = (
            results_df.groupby("genre")
            .agg(
                tracks=("song", "count"),
                tempo_consensus_mean=("tempo_consensus", "mean"),
                tempo_consensus_std=("tempo_consensus", "std"),
                confidence_mean=("confidence", "mean"),
                ambiguous_rate=("tempo_ambiguous", lambda s: float(np.mean(s.astype(float)))),
            )
            .reset_index()
        )
        summary.to_csv(config.output_dir / "genre_summary.csv", index=False)

        if save_plots:
            save_histogram(
                results_df,
                config.plots_dir / "tempo_histogram_consensus.png",
                column="tempo_consensus",
                title="Distribution of Consensus Tempo",
                bins=config.histogram_bins,
            )
            save_scatter(
                results_df,
                config.plots_dir / "tempo_comparison_librosa_vs_autocorr.png",
                x="tempo_librosa",
                y="tempo_autocorr",
                title="Librosa vs Autocorrelation Tempo Estimates",
            )
            save_scatter(
                results_df,
                config.plots_dir / "tempo_comparison_consensus_vs_tempogram.png",
                x="tempo_consensus",
                y="tempo_tempogram",
                title="Consensus vs Tempogram Tempo Estimates",
            )
            save_genre_boxplot(
                results_df,
                config.plots_dir / "tempo_consensus_by_genre.png",
                column="tempo_consensus",
                title="Consensus Tempo by Genre",
            )

    if not failed_df.empty:
        failed_df.to_csv(config.output_dir / "failed_files.csv", index=False)
    else:
        pd.DataFrame(columns=["song", "track_path", "genre", "error", "traceback"]).to_csv(
            config.output_dir / "failed_files.csv", index=False
        )

    return results_df, failed_df
