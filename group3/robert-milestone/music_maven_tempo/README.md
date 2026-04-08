# Music Maven Tempo & Temporal Query Subsystem

This package expands the original CSC475/575 tempo prototype into a multi-file subsystem focused on Robert Widjaja's assigned scope:
- batch audio ingestion from a music dataset
- onset strength extraction
- multiple classical tempo estimators
- confidence scoring and ambiguity detection
- per-track caching to avoid recomputation
- local tempo fluctuation analysis
- temporal-query utilities built on top of the extracted features

Default dataset layout expected:

```text
genres_original/
  blues/
    blues.00000.wav
  classical/
  country/
  disco/
  hiphop/
  jazz/
  metal/
  pop/
  reggae/
  rock/
```

## Quick start

### 1) Batch-process GTZAN
```bash
python scripts/run_batch.py --dataset genres_original --output-dir outputs
```

### 2) Analyze one track
```bash
python scripts/analyze_track.py --file genres_original/blues/blues.00000.wav --pretty
```

### 3) Run temporal queries on the produced CSV
```bash
python scripts/run_queries.py --csv outputs/tempo_results_full.csv --query workout --min-bpm 135
python scripts/run_queries.py --csv outputs/tempo_results_full.csv --query similar-slower --track "Billie Jean" --max-results 10
```

## Outputs

Main outputs from batch mode:
- `outputs/tempo_results_full.csv`
- `outputs/genre_summary.csv`
- `outputs/failed_files.csv`
- `outputs/plots/*.png`
- `cache/*.json`

## Notes

- Designed to work even when a few dataset files are corrupted.
- Uses only common Python MIR stack pieces: `numpy`, `pandas`, `librosa`, `matplotlib`.
- Caching is per-file and based on file metadata, so unchanged files are not reprocessed.
