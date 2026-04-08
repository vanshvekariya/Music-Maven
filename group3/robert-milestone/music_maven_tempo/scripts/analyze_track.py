from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.analysis import analyze_track
from src.caching import TrackCache
from src.config import TempoConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a single track for tempo and temporal features.")
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TempoConfig(cache_dir=args.cache_dir, sample_rate=args.sample_rate)
    config.ensure_directories()
    result = analyze_track(
        path=args.file,
        config=config,
        cache=TrackCache(args.cache_dir),
        dataset_root=args.dataset_root,
    )
    if args.pretty:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
