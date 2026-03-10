# SQLite Database Load Report

**Generated:** 2026-03-05  
**Database:** `music4all.db`  
**Source:** `data/raw/dataset/`

---

## Table Sizes

| Table              | Rows       |
|--------------------|------------|
| `songs`            | 109,269    |
| `listening_history`| 5,109,592  |

---

## Songs Table — Schema

| Column           | Type    | Description                          |
|------------------|---------|--------------------------------------|
| song_id          | TEXT    | Unique song identifier               |
| artist           | TEXT    | Artist name                          |
| song_name        | TEXT    | Song title                           |
| album            | TEXT    | Album name                           |
| spotify_id       | TEXT    | Spotify track identifier             |
| popularity       | REAL    | Spotify popularity score (0–100)     |
| release          | REAL    | Release year                         |
| danceability     | REAL    | Danceability score (0.0–1.0)         |
| energy           | REAL    | Energy score (0.0–1.0)               |
| key              | REAL    | Musical key (0–11)                   |
| mode             | REAL    | 1 = major, 0 = minor                 |
| valence          | REAL    | Musical positiveness (0.0–1.0)       |
| tempo            | REAL    | Tempo in BPM                         |
| duration_ms      | REAL    | Track duration in milliseconds       |
| lang             | TEXT    | Language code (e.g. "en", "pt")      |
| tags             | TEXT    | Comma-separated user tags            |
| genres           | TEXT    | Comma-separated genres               |
| has_lyrics       | INTEGER | 1 if lyrics file exists, 0 otherwise |

**Indexes created:** `idx_artist`, `idx_popularity`, `idx_tempo`, `idx_lang`

---

## Sample Row

```
song_id   : 0009fFIM1eYThaPg
artist    : Cheryl
song_name : Rain on Me
lang      : en
popularity: 12.0
tempo     : 110.973 BPM
genres    : pop
```

---

## Top 5 Artists by Song Count

| Artist        | Songs |
|---------------|-------|
| Queen         | 264   |
| David Bowie   | 226   |
| The Beatles   | 213   |
| Madonna       | 200   |
| Kylie Minogue | 171   |

---

## Top 5 Languages

| Language      | Songs  | Notes                          |
|---------------|--------|--------------------------------|
| en            | 84,103 | ~77% of the dataset            |
| INTRUMENTAL   | 9,417  | Dataset typo — no language tag |
| pt            | 7,020  | Portuguese                     |
| es            | 3,225  | Spanish                        |
| ko            | 1,145  | Korean                         |

---

## Notes

- All 109,269 songs have a corresponding lyrics file (`has_lyrics = True` for all rows).
- `INTRUMENTAL` is a dataset-level label (misspelled) used for songs with no lyrics language — treat as a special case when filtering by language.
- `listening_history` table has indexes on both `user` and `song` columns for fast join queries.
