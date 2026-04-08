from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import librosa
import numpy as np

from .tempo_methods import TempoEstimate, tempo_ratio_distance


def _safe_unit_interval(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def beat_stability_score(beats: np.ndarray, sr: int, hop_length: int) -> float:
    if beats is None or len(beats) < 3:
        return 0.25
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
    ibis = np.diff(beat_times)
    ibis = ibis[ibis > 1e-6]
    if ibis.size < 2:
        return 0.25
    cv = float(np.std(ibis) / (np.mean(ibis) + 1e-9))
    return _safe_unit_interval(1.0 - min(cv, 1.0))


def onset_peak_score(onset_env: np.ndarray) -> float:
    if onset_env.size == 0:
        return 0.0
    p95 = float(np.percentile(onset_env, 95))
    mean = float(np.mean(onset_env)) + 1e-9
    ratio = p95 / mean
    return _safe_unit_interval((ratio - 1.0) / 5.0)


def agreement_score(estimates: List[TempoEstimate], consensus_bpm: float) -> Tuple[float, float]:
    valid = [e for e in estimates if np.isfinite(e.bpm)]
    if not valid or not np.isfinite(consensus_bpm):
        return 0.0, float("inf")
    distances = np.array([tempo_ratio_distance(e.bpm, consensus_bpm) for e in valid], dtype=float)
    weighted_dist = float(np.average(distances, weights=[e.confidence_hint for e in valid]))
    score = float(np.exp(-weighted_dist / 14.0))
    return _safe_unit_interval(score), weighted_dist


def local_tempo_stability(local_tempi: np.ndarray) -> Tuple[float, float]:
    if local_tempi.size < 2:
        return 0.25, float("inf")
    local_tempi = local_tempi[np.isfinite(local_tempi)]
    if local_tempi.size < 2:
        return 0.25, float("inf")
    std = float(np.std(local_tempi))
    mean = float(np.mean(local_tempi)) + 1e-9
    cv = std / mean
    score = _safe_unit_interval(1.0 - min(cv * 2.0, 1.0))
    return score, std


def ambiguity_flags(weighted_distance: float, local_std: float, thresholds: Dict[str, float]) -> Dict[str, bool]:
    return {
        "estimator_disagreement": bool(weighted_distance > thresholds["ambiguous_std_threshold"]),
        "tempo_fluctuating": bool(local_std > thresholds["fluctuating_std_threshold"]),
        "tempo_ambiguous": bool(
            weighted_distance > thresholds["ambiguous_std_threshold"]
            or local_std > thresholds["ambiguous_std_threshold"]
        ),
    }


def confidence_label(score: float, high: float, medium: float) -> str:
    if score >= high:
        return "high"
    if score >= medium:
        return "medium"
    return "low"


def compute_confidence_bundle(
    estimates: List[TempoEstimate],
    consensus_bpm: float,
    beats: np.ndarray,
    sr: int,
    hop_length: int,
    onset_env: np.ndarray,
    local_tempi: np.ndarray,
    high_threshold: float,
    medium_threshold: float,
    ambiguous_std_threshold: float,
    fluctuating_std_threshold: float,
) -> Dict[str, float | bool | str]:
    agree_score, weighted_distance = agreement_score(estimates, consensus_bpm)
    beat_score = beat_stability_score(beats=beats, sr=sr, hop_length=hop_length)
    peak_score = onset_peak_score(onset_env)
    local_score, local_std = local_tempo_stability(local_tempi)

    confidence = float(
        np.clip(
            0.42 * agree_score + 0.24 * beat_score + 0.16 * peak_score + 0.18 * local_score,
            0.0,
            1.0,
        )
    )

    flags = ambiguity_flags(
        weighted_distance=weighted_distance,
        local_std=local_std,
        thresholds={
            "ambiguous_std_threshold": ambiguous_std_threshold,
            "fluctuating_std_threshold": fluctuating_std_threshold,
        },
    )

    return {
        "confidence": confidence,
        "confidence_label": confidence_label(confidence, high_threshold, medium_threshold),
        "agreement_score": agree_score,
        "beat_stability_score": beat_score,
        "onset_peak_score": peak_score,
        "local_tempo_score": local_score,
        "estimator_weighted_distance": weighted_distance,
        "local_tempo_std": local_std,
        **flags,
    }
