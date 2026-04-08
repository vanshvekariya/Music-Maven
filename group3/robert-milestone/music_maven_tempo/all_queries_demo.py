# run dataset: python scripts/run_batch.py --dataset genres_original --output-dir outputs

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.query_engine import TemporalQueryEngine


def print_section(title: str, df: pd.DataFrame, columns: list[str] | None = None, max_results: int = 10) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    if df is None or df.empty:
        print("No results.")
        return
    out = df.copy()
    if columns:
        keep = [c for c in columns if c in out.columns]
        if keep:
            out = out[keep]
    print(out.head(max_results).to_string(index=False))


def chill_tracks(df: pd.DataFrame, max_results: int = 10) -> pd.DataFrame:
    out = df[
        (df["tempo_consensus"] <= 90)
        & (df["confidence"] >= 0.50)
    ].copy()
    if out.empty:
        return out
    return out.sort_values(["tempo_consensus", "confidence"], ascending=[True, False]).head(max_results)


def high_confidence_tracks(df: pd.DataFrame, max_results: int = 10) -> pd.DataFrame:
    out = df.copy()
    return out.sort_values(["confidence", "tempo_consensus"], ascending=[False, False]).head(max_results)


def ambiguous_tracks(df: pd.DataFrame, max_results: int = 10) -> pd.DataFrame:
    out = df[df["tempo_ambiguous"].astype(bool)].copy()
    if out.empty:
        return out
    return out.sort_values(
        ["confidence", "local_tempo_std", "estimator_weighted_distance"],
        ascending=[True, False, False],
    ).head(max_results)


def fluctuating_tracks(df: pd.DataFrame, max_results: int = 10) -> pd.DataFrame:
    out = df[df["tempo_fluctuating"].astype(bool)].copy()
    if out.empty:
        return out
    return out.sort_values(["local_tempo_std", "confidence"], ascending=[False, True]).head(max_results)


def fastest_tracks(df: pd.DataFrame, max_results: int = 10) -> pd.DataFrame:
    out = df.copy()
    return out.sort_values(["tempo_consensus", "confidence"], ascending=[False, False]).head(max_results)


def slowest_tracks(df: pd.DataFrame, max_results: int = 10) -> pd.DataFrame:
    out = df.copy()
    return out.sort_values(["tempo_consensus", "confidence"], ascending=[True, False]).head(max_results)


def by_semantic_class(df: pd.DataFrame, semantic_class: str, max_results: int = 10) -> pd.DataFrame:
    out = df[df["semantic_tempo_class"].astype(str).str.lower() == semantic_class.lower()].copy()
    if out.empty:
        return out
    return out.sort_values(["confidence", "tempo_consensus"], ascending=[False, False]).head(max_results)


def by_genre(df: pd.DataFrame, genre: str, max_results: int = 10) -> pd.DataFrame:
    out = df[df["genre"].astype(str).str.lower() == genre.lower()].copy()
    if out.empty:
        return out
    return out.sort_values(["confidence", "tempo_consensus"], ascending=[False, False]).head(max_results)


def by_meter(df: pd.DataFrame, meter: str, max_results: int = 10) -> pd.DataFrame:
    out = df[df["meter_guess"].astype(str).str.lower() == meter.lower()].copy()
    if out.empty:
        return out
    return out.sort_values(["confidence", "tempo_consensus"], ascending=[False, False]).head(max_results)


def genre_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("genre")
        .agg(
            tracks=("song", "count"),
            tempo_consensus_mean=("tempo_consensus", "mean"),
            tempo_consensus_std=("tempo_consensus", "std"),
            confidence_mean=("confidence", "mean"),
            ambiguous_rate=("tempo_ambiguous", lambda s: float(s.astype(float).mean())),
        )
        .reset_index()
        .sort_values("tempo_consensus_mean", ascending=False)
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run many temporal queries against tempo_results_full.csv")
    parser.add_argument("--csv", type=str, default="outputs/tempo_results_full.csv")
    parser.add_argument("--track", type=str, default="blues.00000")
    parser.add_argument("--genre", type=str, default="reggae")
    parser.add_argument("--semantic-class", type=str, default="energetic")
    parser.add_argument("--meter", type=str, default="4/4_candidate")
    parser.add_argument("--max-results", type=int, default=10)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    engine = TemporalQueryEngine(df)

    common_cols = [
        "song",
        "genre",
        "tempo_consensus",
        "confidence",
        "confidence_label",
        "semantic_tempo_class",
        "meter_guess",
        "tempo_ambiguous",
        "tempo_fluctuating",
        "onset_density",
        "onset_peak_ratio",
        "local_tempo_std",
    ]

    print_section(
        "WORKOUT TRACKS",
        engine.workout_tracks(max_results=args.max_results),
        columns=common_cols + ["workout_score"],
        max_results=args.max_results,
    )

    print_section(
        "CHILL / RELAXED TRACKS",
        chill_tracks(df, max_results=args.max_results),
        columns=common_cols,
        max_results=args.max_results,
    )

    print_section(
        "HIGH CONFIDENCE TRACKS",
        high_confidence_tracks(df, max_results=args.max_results),
        columns=common_cols,
        max_results=args.max_results,
    )

    print_section(
        "AMBIGUOUS TRACKS",
        ambiguous_tracks(df, max_results=args.max_results),
        columns=common_cols + ["estimator_weighted_distance"],
        max_results=args.max_results,
    )

    print_section(
        "TEMPO-FLUCTUATING TRACKS",
        fluctuating_tracks(df, max_results=args.max_results),
        columns=common_cols + ["local_tempo_std"],
        max_results=args.max_results,
    )

    print_section(
        "FASTEST TRACKS",
        fastest_tracks(df, max_results=args.max_results),
        columns=common_cols,
        max_results=args.max_results,
    )

    print_section(
        "SLOWEST TRACKS",
        slowest_tracks(df, max_results=args.max_results),
        columns=common_cols,
        max_results=args.max_results,
    )

    print_section(
        f"TRACKS IN SEMANTIC TEMPO CLASS = {args.semantic_class}",
        by_semantic_class(df, args.semantic_class, max_results=args.max_results),
        columns=common_cols,
        max_results=args.max_results,
    )

    print_section(
        f"TOP TRACKS IN GENRE = {args.genre}",
        by_genre(df, args.genre, max_results=args.max_results),
        columns=common_cols,
        max_results=args.max_results,
    )

    print_section(
        f"METER CANDIDATES = {args.meter}",
        by_meter(df, args.meter, max_results=args.max_results),
        columns=common_cols,
        max_results=args.max_results,
    )

    print_section(
        f"SIMILAR GROOVE BUT SLOWER THAN '{args.track}'",
        engine.similar_groove_but_slower(args.track, max_results=args.max_results),
        columns=common_cols + ["delta_bpm", "density_gap", "peak_gap", "similarity_score"],
        max_results=args.max_results,
    )

    print_section(
        "TIME SIGNATURE CANDIDATES",
        engine.time_signature_candidates(max_results=args.max_results),
        columns=common_cols,
        max_results=args.max_results,
    )

    print_section(
        "GENRE SUMMARY",
        genre_summary(df),
        columns=[
            "genre",
            "tracks",
            "tempo_consensus_mean",
            "tempo_consensus_std",
            "confidence_mean",
            "ambiguous_rate",
        ],
        max_results=100,
    )


if __name__ == "__main__":
    main()
    
#run with python all_queries_demo.py --csv outputs/tempo_results_full.csv
# or specific: python all_queries_demo.py --csv outputs/tempo_results_full.csv --track blues.00000 --genre reggae --semantic-class energetic --meter 4/4_candidate --max-results 15