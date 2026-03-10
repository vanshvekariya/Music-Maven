# Music4All Data Quality Report

**Date:** February 2026  
**Dataset:** Music4All  
**Source path:** `data/raw/dataset/`

---

## Summary

All 5 CSV files are fully populated with no missing values across 109,269 songs.  
Lyrics coverage is 100%. The dataset is clean and ready for ingestion with no preprocessing required.

---

## File Overview

| File | Rows | Columns |
|------|------|---------|
| `id_information.csv` | 109,269 | id, artist, song, album_name |
| `id_metadata.csv` | 109,269 | id, spotify_id, popularity, release, danceability, energy, key, mode, valence, tempo, duration_ms |
| `id_lang.csv` | 109,269 | id, lang |
| `id_tags.csv` | 109,269 | id, tags |
| `id_genres.csv` | 109,269 | id, genres |

---

## Missing Values

**None.** All columns in all 5 files are fully populated.

---

## Songs & Artists

- **Total songs:** 109,269
- **Unique artists:** 16,269
- **Unique albums:** 38,363
- **Empty artist fields:** 0
- **Empty song name fields:** 0

---

## Metadata Distributions

### Popularity (0–100, from Spotify)
| Stat | Value |
|------|-------|
| Min | 0.0 |
| Max | 95.0 |
| Mean | 35.1 |
| Songs with popularity = 0 | 248 (0.2%) |

> 248 songs have popularity = 0. These are likely obscure or removed tracks. Kept as-is.

### Tempo (BPM)
| Stat | Value |
|------|-------|
| Min | 0.0 |
| Max | 242.9 |
| Mean | 122.8 |

> A small number of songs have tempo = 0 (likely spoken word or ambient). Kept as-is.

### Energy (0.0–1.0)
| Stat | Value |
|------|-------|
| Min | 0.000 |
| Max | 1.000 |

### Danceability (0.0–1.0)
| Stat | Value |
|------|-------|
| Min | 0.000 |
| Max | 0.988 |

### Valence (0.0–1.0)
| Stat | Value |
|------|-------|
| Min | 0.000 |
| Max | 0.998 |

---

## Language Distribution

- **Unique languages:** 46
- Top 10:

| Language | Count |
|----------|-------|
| en (English) | 84,103 |
| INTRUMENTAL | 9,417 |
| pt (Portuguese) | 7,020 |
| es (Spanish) | 3,225 |
| ko (Korean) | 1,145 |
| fr (French) | 994 |
| ja (Japanese) | 615 |
| de (German) | 577 |
| pl (Polish) | 446 |
| it (Italian) | 437 |

> Note: `INTRUMENTAL` is a typo present in the original dataset (not `INSTRUMENTAL`). Not corrected — source data preserved as-is.

---

## Genre Distribution (Top 10)

| Genre | Count |
|-------|-------|
| rock | 25,731 |
| pop | 22,013 |
| electronic | 12,769 |
| alternative rock | 8,103 |
| indie rock | 7,943 |
| metal | 6,459 |
| folk | 6,106 |
| singer-songwriter | 5,793 |
| classic rock | 5,688 |
| soul | 5,458 |

> Genres are stored as a single string per song (comma-separated values possible).

---

## Lyrics Coverage

| Metric | Value |
|--------|-------|
| Songs with lyrics files | 109,269 |
| Total songs | 109,269 |
| Coverage | **100%** |

All songs have a corresponding `.txt` file in `data/raw/dataset/lyrics/`.

---

## Known Issues / Notes

| Issue | Severity | Decision |
|-------|----------|----------|
| 248 songs with popularity = 0 | Low | Keep as-is |
| Some songs with tempo = 0 | Low | Keep as-is |
| `INTRUMENTAL` typo in lang field | Low | Keep as-is (source data) |
| No audio clips folder | N/A | Audio ingestion deferred to Layer 2 |

---

## Conclusion

The dataset is clean and ready for ingestion. No preprocessing or imputation is required for the base system (Layer 1). The processor can merge all 5 CSVs and write directly to SQLite without any data cleaning steps.
