# Project Notes

**Project:** Music Maven — MP2.G1 Knowledge Graph & Architecture  
**Course:** CSC475 / CSC575 – Music Information Retrieval  
**Date:** 2026-03-10

---

## Phase Status

### Phase A — Complete
- Music4All dataset structure analysed and documented
- Data schema mapped across all 6 CSV files and lyrics/audio folders
- Query types confirmed across 3 layers (Basic, Expected, Advanced)
- Requirements captured in `reports/project_plan.md`

### Phase B — Complete
- Music4All CSVs loaded and merged into SQLite (`music4all.db`)
  - `songs` table: 109,269 rows, 18 columns
  - `listening_history` table: 5,109,592 rows
- SQL agent operational — translates natural language to SQL via LangChain + LLM
- FastAPI backend running with `/query`, `/health`, `/examples`, `/system/info` endpoints
- React frontend updated with music-specific UI (`SongCard`, audio feature display)
- Hallucination guardrails added to SQL agent prompt

### Phase C — Complete
- Embedding strategy decided (two Qdrant collections — see `embedding_options_report.md`)
- Qdrant running via Docker (`docker-compose up -d`)
- Embedding pipeline built (`src/data/embedding_pipeline.py`) and executed
  - `music4all_lyrics` collection: lyrics embedded with `paraphrase-multilingual-MiniLM-L12-v2`
  - `music4all_tags` collection: artist/genre/tag metadata embedded with `all-MiniLM-L6-v2`
  - Instrumental placeholders skipped (9,419 songs)
- Vector Agent operational — keyword-based routing between lyrics and tags collections
- Vector search tested end-to-end via frontend
- Cross-lingual retrieval confirmed — English queries return Portuguese, Spanish, Italian songs

### Phase D — Not Started
- Neo4j knowledge graph schema not yet designed
- Artist–album–genre–tag relationship ingestion not built
- Graph agent not implemented

### Phase E — In Progress
- API and UI working for SQL and Vector queries
- Hybrid routing operational (SQL + Vector agents both active)
- **Remaining:** language filter for vector queries, report draft, subgroup plugin endpoints

### Future Phases
- Audio embeddings (CLAP / MERT) — awaiting audio clip availability
- Session memory for multi-turn conversations
- Subgroup module integration (tempo, recommender, artist classifier, lyrics, genre)
- Redis caching, Docker deployment, performance tuning

---

## Layer Completion Status

| Layer | Description | Status |
|---|---|---|
| Layer 1 | SQL analytics, text embeddings, vector search, REST API, basic UI | **Complete** — SQL + vector search + API + UI all operational |
| Layer 2 | Knowledge graph, hybrid retrieval, conversation memory, plugin endpoints | Not started |
| Layer 3 | Session memory, caching, Docker, subgroup integration, performance | Not started |

---

## Process So Far

### 1. Dataset Loading

The Music4All dataset is distributed across six tab-separated CSV files and two folders. The CSVs were loaded and merged on `song_id` into a single SQLite database (`music4all.db`) using a left join chain anchored on `id_information.csv`. Indexes were created on `artist`, `popularity`, `tempo`, and `lang` for fast SQL filtering.

### 2. SQL Agent

A LangChain SQL agent translates natural language queries into SQL against the `songs` table via an LLM (OpenRouter). A hallucination issue was discovered early — the model was appending fabricated streaming metrics (e.g., "≈100M views") not present in the database. Fixed by adding explicit negative examples into the prompt showing both correct and wrong output formats.

### 3. API and Frontend

FastAPI backend exposes the SQL agent. React frontend displays results via a `SongCard` component showing music-specific fields: `popularity`, `tempo`, `energy`, `danceability`, `valence`, `lang`, `mode`, `genres`, `tags`. System currently runs SQL-only; vector search enabled once Phase C is complete.

### 4. Embedding Strategy and Vector Search

Two Qdrant collections created:
- `music4all_lyrics` — `paraphrase-multilingual-MiniLM-L12-v2` (handles 23% non-English songs)
- `music4all_tags` — `all-MiniLM-L6-v2` (tags are ~99.7% English)

The Vector Agent routes queries between collections based on keywords: queries containing mood/lyric words ("sad", "heartbreak", "feel") hit both collections; all others hit tags only. Results are merged by similarity score.

Full model comparison in `embedding_options_report.md`.

### 5. Cross-lingual Retrieval Behaviour

Confirmed that the multilingual model maps French, English, Spanish, and Portuguese queries into the same embedding space — an English query for "sad songs about lost love" returns Portuguese, Spanish, and Italian songs as top results. This is expected and correct behaviour.

**Known limitation:** language keywords in queries (e.g., "French sad songs") are not currently parsed as metadata filters — "French" is treated as semantic context, not a `lang=FR` filter. A fix is planned: detect language words in the Vector Agent and pass them as a Qdrant payload filter, so language-scoped vector searches work correctly.

### 6. Qdrant API Upgrade

The installed Qdrant client (v1.16+) removed the `.search()` method in favour of `.query_points()`. Updated `src/vectordb/operations.py` accordingly. This is a good example of why pinning exact dependency versions in `requirements.txt` matters — a silent upgrade would have broken vector search.

### 7. Codebase Cleanup and GitHub Push

All YouTube-era files, scripts, and references were removed. The project was pushed to a new branch (`phase1-text-search`) on a remote GitHub repository, keeping the existing `main` branch untouched.

---

## To Do / Known Issues

- **Router returns UNKNOWN → crash:** When the LLM is overloaded it returns a garbled classification, the router falls back to `UNKNOWN`, `agents: []`, nothing runs, and `answer = None`. The API then crashes with a Pydantic validation error (`Input should be a valid string`). Two fixes needed:
  1. `src/agents/orchestrator.py` — default to SQL when routing returns `[]` or `UNKNOWN` instead of returning nothing
  2. `src/api/main.py` — add a fallback so `answer` is never `None` (return a user-friendly message instead of a 500 error)

---

## Open Technical Decisions

> **SQLite vs PostgreSQL:** The project specification (`project_plan.md`) targets PostgreSQL as the SQL backend. The current implementation uses SQLite for simplicity during solo development. Revisit this decision before integrating with other subgroups or deploying to a shared server — migration may be necessary at that point.

---

## Note on Instrumental Lyric Files

Approximately 9,419 lyric files contain only the text `"INSTRUMENTAL"` — a dataset placeholder for songs with no lyrics. These are excluded from the lyrics Qdrant collection because:
- All 9,419 would produce nearly identical vectors, causing false matches on any semantic query
- These songs remain discoverable through the `music4all_tags` collection via tags and genres

The `lang` column uses `"INTRUMENTAL"` (dataset typo, missing 'S') for these songs. Preserved as-is — SQL queries must use `'INTRUMENTAL'` to match them.
