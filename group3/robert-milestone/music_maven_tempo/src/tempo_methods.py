from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import librosa
import numpy as np


@dataclass(slots=True)
class TempoEstimate:
    method: str
    bpm: float
    raw_value: float
    confidence_hint: float


def _valid_bpm(bpm: float, bpm_min: float, bpm_max: float) -> bool:
    return np.isfinite(bpm) and bpm_min <= bpm <= bpm_max


def fold_tempo_to_range(bpm: float, bpm_min: float, bpm_max: float) -> float:
    if not np.isfinite(bpm) or bpm <= 0:
        return float("nan")
    while bpm < bpm_min:
        bpm *= 2.0
    while bpm > bpm_max:
        bpm /= 2.0
    return float(bpm)


def tempo_ratio_distance(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0 or b <= 0:
        return float("inf")
    candidates = [b, b / 2.0, b * 2.0, b / 3.0, b * 3.0]
    return float(min(abs(a - c) for c in candidates if c > 0))


def estimate_tempo_librosa(y: np.ndarray, sr: int) -> Tuple[float, np.ndarray]:
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.asarray(tempo).squeeze())
    return tempo, beats


def estimate_tempo_autocorr(onset_env: np.ndarray, sr: int, hop_length: int, bpm_min: float, bpm_max: float) -> float:
    ac = librosa.autocorrelate(onset_env, max_size=max(2, len(onset_env) // 2))
    lag_min = int(np.floor(60.0 * sr / (hop_length * bpm_max)))
    lag_max = int(np.ceil(60.0 * sr / (hop_length * bpm_min)))
    lag_min = max(lag_min, 1)
    lag_max = min(lag_max, len(ac) - 1)
    if lag_min >= lag_max:
        return float("nan")
    segment = ac[lag_min : lag_max + 1]
    best_lag = int(np.argmax(segment) + lag_min)
    bpm = 60.0 * sr / (hop_length * best_lag)
    return float(bpm)


def estimate_tempo_tempogram(onset_env: np.ndarray, sr: int, hop_length: int, bpm_min: float, bpm_max: float) -> float:
    tg = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    if tg.size == 0:
        return float("nan")
    tempo_curve = librosa.feature.tempo(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        aggregate=None,
        max_tempo=bpm_max,
    )
    if tempo_curve is None or len(tempo_curve) == 0:
        return float("nan")
    tempo_curve = np.asarray(tempo_curve, dtype=float)
    tempo_curve = tempo_curve[np.isfinite(tempo_curve)]
    tempo_curve = tempo_curve[(tempo_curve >= bpm_min) & (tempo_curve <= bpm_max)]
    if tempo_curve.size == 0:
        return float("nan")
    return float(np.median(tempo_curve))


def estimate_tempo_from_onset_peaks(
    onset_env: np.ndarray,
    sr: int,
    hop_length: int,
    bpm_min: float,
    bpm_max: float,
    **peak_kwargs,
) -> float:
    peaks = librosa.util.peak_pick(onset_env, **peak_kwargs)
    if len(peaks) < 2:
        return float("nan")
    peak_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
    iois = np.diff(peak_times)
    iois = iois[iois > 0.08]
    if iois.size == 0:
        return float("nan")
    median_ioi = float(np.median(iois))
    bpm = 60.0 / median_ioi
    return fold_tempo_to_range(bpm, bpm_min, bpm_max)


def collect_estimates(
    y: np.ndarray,
    sr: int,
    onset_env: np.ndarray,
    hop_length: int,
    bpm_min: float,
    bpm_max: float,
    peak_kwargs: Dict[str, int | float],
) -> Tuple[List[TempoEstimate], np.ndarray]:
    estimates: List[TempoEstimate] = []

    librosa_bpm, beats = estimate_tempo_librosa(y=y, sr=sr)
    estimates.append(TempoEstimate("librosa_beat_track", fold_tempo_to_range(librosa_bpm, bpm_min, bpm_max), librosa_bpm, 0.9))

    ac_bpm = estimate_tempo_autocorr(onset_env=onset_env, sr=sr, hop_length=hop_length, bpm_min=bpm_min, bpm_max=bpm_max)
    estimates.append(TempoEstimate("autocorrelation", fold_tempo_to_range(ac_bpm, bpm_min, bpm_max), ac_bpm, 0.7))

    tg_bpm = estimate_tempo_tempogram(onset_env=onset_env, sr=sr, hop_length=hop_length, bpm_min=bpm_min, bpm_max=bpm_max)
    estimates.append(TempoEstimate("tempogram", fold_tempo_to_range(tg_bpm, bpm_min, bpm_max), tg_bpm, 0.8))

    peaks_bpm = estimate_tempo_from_onset_peaks(
        onset_env=onset_env,
        sr=sr,
        hop_length=hop_length,
        bpm_min=bpm_min,
        bpm_max=bpm_max,
        **peak_kwargs,
    )
    estimates.append(TempoEstimate("onset_peak_ioi", fold_tempo_to_range(peaks_bpm, bpm_min, bpm_max), peaks_bpm, 0.55))

    return estimates, beats


def weighted_consensus(estimates: List[TempoEstimate], bpm_min: float, bpm_max: float) -> float:
    valid = [e for e in estimates if _valid_bpm(e.bpm, bpm_min, bpm_max)]
    if not valid:
        return float("nan")
    anchors = []
    for anchor in valid:
        aligned_values = []
        weights = []
        for other in valid:
            candidates = np.array([other.bpm, other.bpm / 2.0, other.bpm * 2.0, other.bpm / 3.0, other.bpm * 3.0])
            idx = int(np.argmin(np.abs(candidates - anchor.bpm)))
            aligned = float(candidates[idx])
            if bpm_min <= aligned <= bpm_max * 3:
                aligned_values.append(aligned)
                weights.append(other.confidence_hint)
        if aligned_values:
            anchors.append((anchor.bpm, float(np.average(aligned_values, weights=weights))))
    if not anchors:
        return float(np.mean([e.bpm for e in valid]))
    best_anchor, best_value = min(anchors, key=lambda pair: abs(pair[1] - pair[0]))
    return fold_tempo_to_range(best_value, bpm_min, bpm_max)


def local_tempo_profile(
    onset_env: np.ndarray,
    sr: int,
    hop_length: int,
    bpm_max: float,
) -> np.ndarray:
    tempo_curve = librosa.feature.tempo(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        aggregate=None,
        max_tempo=bpm_max,
    )
    tempo_curve = np.asarray(tempo_curve, dtype=float)
    tempo_curve = tempo_curve[np.isfinite(tempo_curve)]
    return tempo_curve
