from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _normalize_series(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    rng = x.max() - x.min()
    if not np.isfinite(rng) or rng == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - x.min()) / rng


class TemporalQueryEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        if "tempo_consensus" not in self.df.columns:
            raise ValueError("DataFrame must contain tempo_consensus")

    def workout_tracks(self, min_bpm: float = 135.0, min_confidence: float = 0.55, max_results: int = 20) -> pd.DataFrame:
        df = self.df[
            (self.df["tempo_consensus"] >= min_bpm)
            & (self.df["confidence"] >= min_confidence)
            & (~self.df["tempo_ambiguous"].astype(bool))
        ].copy()
        if df.empty:
            return df
        density = _normalize_series(df.get("onset_density", pd.Series(np.zeros(len(df)), index=df.index)))
        energy = _normalize_series(df.get("onset_peak_ratio", pd.Series(np.zeros(len(df)), index=df.index)))
        confidence = _normalize_series(df["confidence"])
        df["workout_score"] = 0.45 * confidence + 0.35 * energy + 0.20 * density
        return df.sort_values(["workout_score", "tempo_consensus"], ascending=[False, False]).head(max_results)

    def similar_groove_but_slower(
        self,
        track_query: str,
        tolerance_bpm: float = 20.0,
        min_slowdown_bpm: float = 8.0,
        max_results: int = 10,
    ) -> pd.DataFrame:
        source = self.df[self.df["song"].str.contains(track_query, case=False, na=False)]
        if source.empty:
            return pd.DataFrame()
        src = source.iloc[0]
        candidates = self.df[self.df["song"] != src["song"]].copy()
        candidates = candidates[candidates["tempo_consensus"] < float(src["tempo_consensus"]) - min_slowdown_bpm]
        if candidates.empty:
            return candidates
        candidates["delta_bpm"] = (candidates["tempo_consensus"] - float(src["tempo_consensus"])) .abs()
        candidates["density_gap"] = (candidates.get("onset_density", 0) - float(src.get("onset_density", 0))).abs()
        candidates["peak_gap"] = (candidates.get("onset_peak_ratio", 0) - float(src.get("onset_peak_ratio", 0))).abs()
        candidates = candidates[candidates["delta_bpm"] <= tolerance_bpm + min_slowdown_bpm]
        if candidates.empty:
            return candidates
        candidates["similarity_score"] = (
            1.0 / (1.0 + candidates["delta_bpm"]) +
            0.7 / (1.0 + candidates["density_gap"]) +
            0.7 / (1.0 + candidates["peak_gap"]) +
            0.5 * candidates["confidence"]
        )
        return candidates.sort_values("similarity_score", ascending=False).head(max_results)

    def time_signature_candidates(self, max_results: int = 20) -> pd.DataFrame:
        df = self.df.copy()
        if "meter_guess" not in df.columns:
            return pd.DataFrame()
        return df.sort_values(["confidence", "tempo_consensus"], ascending=[False, False]).head(max_results)
