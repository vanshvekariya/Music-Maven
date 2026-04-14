# Project Notes

**Project:** Music Maven — MP2.G1 Knowledge Graph & Architecture  
**Course:** CSC475 / CSC575 – Music Information Retrieval  
**Date:** 2026-04-14 (updated)

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

### Phase D — Complete
- Knowledge graph built using NetworkX (in-memory directed graph, serialized to `data/music4all_kg.gpickle`)
- Node types: Song (~109k), Artist (~16k), Genre (~853), Language, Tag, and Tier (popularity/energy/danceability/tempo buckets)
- Relationship edges: `PERFORMED_BY`, `HAS_GENRE`, `IN_LANGUAGE`, `HAS_TAG`, `WORKS_IN_GENRE`, `SINGS_IN`, `RELATED_TO` (genre co-occurrence), `IN_*_TIER`
- Pre-computed aggregations stored as graph-level attributes: artist/song/genre leaderboards, language distribution, global stats, entity lookup tables
- KG Query Engine resolves 17+ template patterns via regex matching — zero LLM calls for factual queries
- KG-aware router (`KGQueryRouter`) replaces the LLM router for classification — queries are routed locally using regex patterns, keyword scoring, and entity recognition
- Orchestrator updated with `kg_agent` node in the LangGraph workflow; supports KG-direct, SQL, Vector, and hybrid (KG + Vector) execution paths
- KG loads automatically on server startup via `load_or_build()` — rebuilds from SQLite if pickle is missing

### Phase E — In Progress (core items landed)
- API and UI working for SQL, Vector, and KG queries
- Hybrid routing operational (KG + Vector, SQL + Vector)
- KG-direct path handles factual queries with zero latency and zero LLM cost
- **Done:** language filter for vector search (`lang_filter` on `/query` + optional inference from query text); subgroup plugin registry and forward (`GET /plugins/status`, `POST /plugins/{module}/forward`); milestone report draft at `reports/phase_e_milestone_report_draft.md`
- **Done:** optional **`use_kg`** on `POST /query` + UI checkbox “Use KG (instant)” to bypass KG-direct routing for SQL/vector testing
- **Done:** **conversation memory** — client `session_id` on `/query`; server stores a **rolling LLM summary** (not raw transcripts) capped by `memory_max_chars`; injected into SQL agent, vector embedding string, and hybrid synthesis; `DELETE /conversation/{session_id}` clears server state; React **New chat** rotates session and clears remote summary; UI copy explains summary-based memory; results metadata shows `memory_mode: rolling_summary` and `memory_summarization: llm | fallback`
- **Done:** **dependency pinning** — tightened `requirements.txt` (`python-dotenv`, `langid`, `pycountry`, `sqlalchemy`, `networkx`, dev tools); **`qdrant-client==1.16.2`** kept explicit (API surface sensitive to version)
- **Done:** **Chat-style UI** — React keeps a **turn list** (user bubble + assistant block per query); **New chat** clears on-screen thread and server summary; sticky input; examples hidden after first turn
- **Done:** **Router continuation** — short follow-ups like “give 10 more please” / “next 10” are classified as **SQL** before KG/vector fallthrough (`KGQueryRouter` in `src/agents/query_router.py`), so they are not misrouted to **vector** keyword search
- **Done:** **SQL agent context safety** — `BoundedSQLDatabase` in `src/agents/sql_agent.py` truncates each `sql_db_query` tool result (~14k chars) so LangChain cannot blow past the LLM context window; stricter prompt rules (always **LIMIT**, **DISTINCT**/**GROUP BY** for dedupe); injected `conversation_context` capped (~3.5k chars); extra SQL keywords (`distinct`, `unique`, `duplicate`, …) for dedupe-style questions
- **Remaining:** finalize report for submission; point plugin URLs at teammate services when APIs are agreed; optional UI polish

### Future Phases
- Audio embeddings (CLAP / MERT) — awaiting audio clip availability
- Subgroup module integration (tempo, recommender, artist classifier, lyrics, genre) — deprioritized unless APIs land
- Redis caching, Docker deployment, performance tuning
- Optional: persist conversation summaries beyond in-memory (e.g. Redis) for multi-worker deploys

---

## Layer Completion Status

| Layer | Description | Status |
|---|---|---|
| Layer 1 | SQL analytics, text embeddings, vector search, REST API, basic UI | **Complete** — SQL + vector search + API + UI all operational |
| Layer 2 | Knowledge graph, hybrid retrieval, conversation memory, plugin endpoints | **Mostly complete** — KG + hybrid + plugins + **rolling-summary session memory** (`session_id`, summarizer in `src/api/conversation_summarizer.py`, store in `src/api/conversation_store.py`) |
| Layer 3 | Caching, Docker, subgroup integration, performance | Not started (memory is in-process only; not Redis-scale) |

---

## Process So Far

### 1. Dataset Loading

The Music4All dataset is distributed across six tab-separated CSV files and two folders. The CSVs were loaded and merged on `song_id` into a single SQLite database (`music4all.db`) using a left join chain anchored on `id_information.csv`. Indexes were created on `artist`, `popularity`, `tempo`, and `lang` for fast SQL filtering.

### 2. SQL Agent

A LangChain SQL agent translates natural language queries into SQL against the `songs` table via an LLM (OpenRouter). A hallucination issue was discovered early — the model was appending fabricated streaming metrics (e.g., "≈100M views") not present in the database. Fixed by adding explicit negative examples into the prompt showing both correct and wrong output formats.

**Context window failure (mitigated):** The agent re-injects **full tool result strings** into the chat each iteration. A broad `SELECT` with no `LIMIT` can return hundreds of thousands of rows → multi-million **character** payloads → OpenRouter errors (“maximum context length … tokens”). Mitigations: **`BoundedSQLDatabase`** truncates each query tool output with a hint to add `WHERE`/`LIMIT`; system prompt requires **LIMIT** on exploratory queries and **DISTINCT**/**GROUP BY** for unique titles; **`conversation_context`** injected into the SQL agent is hard-capped (~3500 chars) independent of `memory_max_chars`.

### 3. API and Frontend

FastAPI backend exposes the multi-agent stack (KG, SQL, vector). React frontend displays results via a `SongCard` component showing music-specific fields: `popularity`, `tempo`, `energy`, `danceability`, `valence`, `lang`, `mode`, `genres`, `tags`. Optional UI language filter maps to `lang_filter` on `/query` for vector search.

**Session memory (rolling summary):** Each browser tab keeps a `sessionId` (UUID); every `POST /query` may send `session_id`, `memory_turns` (>0 enables), and `memory_max_chars` (budget for stored + injected summary). Before the orchestrator runs, the prior summary is injected as `conversation_context`. After the response, the LLM updates that summary so follow-ups (“more”, “same for Spanish”) get intent without re-sending full prior answers. **Trade-off:** one extra LLM call per turn when memory is on. **Clear:** `DELETE /conversation/{session_id}` or UI **New chat**.

**UI metadata note:** Badges such as **Memory: rolling summary (LLM)** refer to **how the post-reply summary was updated** (LLM vs fallback), not to an extra LLM call for **kg_direct** answers themselves. KG-direct answers are still instant; summarization runs **after** the reply when session memory is enabled.

**KG toggle:** `use_kg: false` strips `kg_direct` from routing so aggregate/demo queries can be forced through SQL when needed.

### 4. Embedding Strategy and Vector Search

Two Qdrant collections created:
- `music4all_lyrics` — `paraphrase-multilingual-MiniLM-L12-v2` (handles 23% non-English songs)
- `music4all_tags` — `all-MiniLM-L6-v2` (tags are ~99.7% English)

The Vector Agent routes queries between collections based on keywords: queries containing mood/lyric words ("sad", "heartbreak", "feel") hit both collections; all others hit tags only. Results are merged by similarity score.

Full model comparison in `embedding_options_report.md`.

### 5. Cross-lingual Retrieval Behaviour

Confirmed that the multilingual model maps French, English, Spanish, and Portuguese queries into the same embedding space — an English query for "sad songs about lost love" returns Portuguese, Spanish, and Italian songs as top results. This is expected and correct behaviour.

**Language filter:** Optional `lang_filter` on `POST /query` applies a Qdrant payload match on `lang`. If omitted, a lightweight heuristic may infer a language from the query text (e.g. "Portuguese" → `pt`); explicit `lang_filter` is preferred for demos.

### 6. Knowledge Graph

A NetworkX directed graph is built from the SQLite database by `src/knowledge_graph/kg_builder.py`. The graph contains ~130k+ nodes across six types (Song, Artist, Genre, Language, Tag, Tier) and several hundred thousand edges encoding relationships like `PERFORMED_BY`, `HAS_GENRE`, `IN_LANGUAGE`, `HAS_TAG`, `WORKS_IN_GENRE`, `SINGS_IN`, `RELATED_TO` (genre co-occurrence), and tier membership edges (`IN_POPULARITY_TIER`, `IN_ENERGY_TIER`, etc.).

Pre-computed aggregations (top artists by popularity, top songs, genre/language distributions, global stats, artist/genre/language lookup dictionaries) are stored as graph-level attributes (`G.graph[...]`) so the query engine can answer common questions in O(1) without traversing the graph at query time.

The KG Query Engine (`src/knowledge_graph/kg_query_engine.py`) defines 17+ regex-based templates (e.g., "top N artists", "how many songs in X language", "compare artist A vs B", "songs with energy above N"). Each template is paired with a resolver function that traverses the graph and returns a formatted markdown answer. If no template matches, the query falls through to the SQL or Vector agent.

The `KGQueryRouter` replaces the original LLM-based query router for classification. It first treats **continuation** phrasing (e.g. “give 10 more”, “next 10 artists”) as **SQL** so vague follow-ups are not defaulted to **vector** search. Then it tries to match the query against the KG engine's templates; if that succeeds, the query is classified as `KG_DIRECT` and answered instantly. Otherwise, it uses keyword scoring and entity recognition (checking if the query mentions a known artist or genre from the graph's lookup tables) to route to SQL, Vector, or Hybrid.

The orchestrator's LangGraph workflow was updated with a `kg_agent` node. The routing decision now supports five paths: `kg_direct` (KG only), `sql`, `vector`, `hybrid` (KG + Vector), and `both_sql_vector` (SQL + Vector). For hybrid KG+Vector queries, the KG handles the structured part and the Vector agent handles the semantic part, with results merged via template (no LLM synthesis needed).

The graph is serialized to `data/music4all_kg.gpickle` and loaded on server startup. If the pickle file is missing, the graph is automatically rebuilt from SQLite. The pickle file is not committed to version control (added to `.gitignore`) since it can be regenerated with `python -m src.knowledge_graph.kg_builder`.

### 7. Qdrant API Upgrade

The installed Qdrant client (v1.16+) removed the `.search()` method in favour of `.query_points()`. Updated `src/vectordb/operations.py` accordingly. **`qdrant-client` is pinned to `1.16.2`** alongside other runtime/dev pins (`sqlalchemy`, `networkx`, `python-dotenv`, etc.) so teammates don’t hit silent client API breaks.

### 8. Codebase Cleanup and GitHub Push

All YouTube-era files, scripts, and references were removed. The project was pushed to a new branch (`phase1-text-search`) on a remote GitHub repository, keeping the existing `main` branch untouched.

---

## To Do / Known Issues

- **Router empty / UNKNOWN:** Orchestrator routes unrouted queries to synthesis with a user-facing fallback message; API avoids returning `None` for `answer`. (LLM-based `QueryRouter` fallback path is rarely used when KG router is active.)
- **OpenRouter / model choice:** Some free-tier model IDs hit **429** (rate limit) or **404** (model unavailable). Use a stable model id in `.env` (e.g. **`openai/gpt-4o-mini`**) for reliable SQL generation; restart the API after changes.
- **Conversation store:** In-memory only — **server restart clears all sessions**; not shared across multiple Uvicorn workers without an external store.
- **SQL “more” / pagination:** Continuation queries now route to **SQL**; exact **OFFSET** behaviour still depends on the LLM reading the rolling summary — refine prompts or add explicit API parameters if demos need deterministic paging.

---

## Open Technical Decisions

> **SQLite vs PostgreSQL:** The project specification (`project_plan.md`) targets PostgreSQL as the SQL backend. The current implementation uses SQLite for simplicity during solo development. Revisit this decision before integrating with other subgroups or deploying to a shared server — migration may be necessary at that point.

---

## Note on Instrumental Lyric Files

Approximately 9,419 lyric files contain only the text `"INSTRUMENTAL"` — a dataset placeholder for songs with no lyrics. These are excluded from the lyrics Qdrant collection because:
- All 9,419 would produce nearly identical vectors, causing false matches on any semantic query
- These songs remain discoverable through the `music4all_tags` collection via tags and genres

The `lang` column uses `"INTRUMENTAL"` (dataset typo, missing 'S') for these songs. Preserved as-is — SQL queries must use `'INTRUMENTAL'` to match them.
