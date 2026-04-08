from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.analysis import analyze_dataset
from src.config import TempoConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch tempo analysis for a music dataset.")
    parser.add_argument("--dataset", type=Path, default=Path("genres_original"), help="Dataset root containing genre folders.")
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--force", action="store_true", help="Ignore cache and recompute all tracks.")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TempoConfig(
        dataset_dir=args.dataset,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        plots_dir=args.output_dir / "plots",
        sample_rate=args.sample_rate,
    )
    results_df, failed_df = analyze_dataset(config=config, save_plots=not args.no_plots, force_recompute=args.force)
    print(f"Processed tracks: {len(results_df)}")
    print(f"Failed tracks: {len(failed_df)}")
    if not results_df.empty:
        print(results_df[["song", "genre", "tempo_consensus", "confidence", "confidence_label"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
