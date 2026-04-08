from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.query_engine import TemporalQueryEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run temporal queries on analyzed tracks.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--query", choices=["workout", "similar-slower", "meter"], required=True)
    parser.add_argument("--min-bpm", type=float, default=135.0)
    parser.add_argument("--track", type=str, default="")
    parser.add_argument("--max-results", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    engine = TemporalQueryEngine(df)

    if args.query == "workout":
        out = engine.workout_tracks(min_bpm=args.min_bpm, max_results=args.max_results)
    elif args.query == "similar-slower":
        if not args.track:
            raise SystemExit("--track is required for similar-slower")
        out = engine.similar_groove_but_slower(track_query=args.track, max_results=args.max_results)
    else:
        out = engine.time_signature_candidates(max_results=args.max_results)

    if out.empty:
        print("No matching tracks found.")
    else:
        cols = [c for c in ["song", "genre", "tempo_consensus", "confidence", "semantic_tempo_class", "meter_guess"] if c in out.columns]
        print(out[cols].to_string(index=False))


if __name__ == "__main__":
    main()
