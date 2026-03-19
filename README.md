# Music Maven

A full-stack AI-powered Music Information Retrieval (MIR) system built on the Music4All dataset. Uses a multi-agent architecture to answer natural language queries about music using SQL analytics, semantic search over lyrics and tags, and LLM-generated responses.

## Project Overview

- **Multi-Agent AI**: KG-aware query router dispatches to KG, SQL, Vector, or hybrid agents with zero LLM routing calls
- **Knowledge Graph**: In-memory NetworkX graph with pre-computed aggregations for instant factual answers (artist stats, genre distributions, leaderboards, comparisons)
- **Two Qdrant Collections**: Semantic search over lyrics (`music4all_lyrics`) and tags/style (`music4all_tags`)
- **FastAPI Backend**: REST API with LangGraph-based orchestration
- **React Frontend**: Clean UI for submitting queries and viewing results
- **Music4All Dataset**: ~109k songs with metadata, audio features, lyrics, genres, and tags

## Project Structure

```
music_mavern/
├── frontend/                  # React + Vite frontend
│   └── src/
│       ├── App.jsx
│       ├── components/        # QueryInput, ResultsDisplay, UI primitives
│       └── services/api.js    # Axios API client
├── src/                       # Python backend
│   ├── agents/
│   │   ├── orchestrator.py    # LangGraph multi-agent workflow (KG-enhanced)
│   │   ├── query_router.py    # KG-aware local classifier + LLM fallback
│   │   ├── sql_agent.py       # LangChain SQL agent
│   │   └── vector_agent.py    # Dual-collection semantic search
│   ├── api/
│   │   ├── main.py            # FastAPI app
│   │   └── models.py          # Pydantic request/response models
│   ├── config/settings.py     # All configuration (models, paths, collections)
│   ├── data/
│   │   ├── music4all_processor.py  # CSV → SQLite ingestion
│   │   └── embedding_pipeline.py  # SQLite → Qdrant embeddings
│   ├── knowledge_graph/
│   │   ├── kg_builder.py      # SQLite → NetworkX graph construction
│   │   └── kg_query_engine.py # Regex-template query resolution (zero LLM)
│   ├── embeddings/            # Embedding model wrappers
│   ├── search/                # Semantic search utilities
│   ├── vectordb/              # Qdrant client + operations
│   └── main.py                # CLI entry point
├── reports/                   # Project reports and notes
├── tests/                     # Test suite
├── run_api.py                 # API startup helper
├── docker-compose.yml         # Qdrant container
└── requirements.txt
```

## Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- Docker (for Qdrant)
- OpenRouter API key (set as `OPENAI_API_KEY` in `.env`)

### 1. Backend Setup

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openrouter_key_here
OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

### 3. Load the Dataset (run once)

Place the Music4All CSV files under `data/raw/dataset/` then:
```bash
python -m src.data.music4all_processor
```

### 4. Start Qdrant and Run the Embedding Pipeline (run once)

```bash
docker-compose up -d
python -m src.data.embedding_pipeline
```
This creates two Qdrant collections (`music4all_lyrics`, `music4all_tags`) from the SQLite database. Takes ~20-40 minutes on CPU.

### 5. Build the Knowledge Graph (run once)

```bash
python -m src.knowledge_graph.kg_builder
```
This reads the SQLite database and builds an in-memory NetworkX graph with Song, Artist, Genre, Language, Tag, and Tier nodes. The graph is serialized to `data/music4all_kg.gpickle` (~300MB) and loaded automatically on server startup. Use `--rebuild` to force regeneration.

### 6. Start the Backend

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 7. Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

### Access
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Query Types

The system automatically routes queries to the right agent(s). Factual queries are answered instantly from the Knowledge Graph with zero LLM calls; semantic and complex queries fall through to SQL or Vector agents.

| Type | Examples |
|---|---|
| **KG Direct** | "Top 10 artists", "How many songs are in Portuguese?", "Stats for Coldplay", "Compare Drake and Eminem", "Genre distribution", "Songs by Radiohead" |
| **SQL** | "Average tempo of hip-hop songs", "Songs with energy above 0.9 sorted by popularity" |
| **Vector** | "Songs with melancholic lyrics", "Music that feels like a road trip", "Songs similar to Bohemian Rhapsody" |
| **Hybrid** | "High-energy songs about heartbreak with popularity above 70", "Fast tempo rock songs that feel melancholic" |

## Dataset

[Music4All](https://github.com/music4all) — ~109k songs with:
- Spotify audio features: `popularity`, `danceability`, `energy`, `valence`, `tempo`, `mode`
- Metadata: `artist`, `song_name`, `album`, `lang`
- `genres`, `tags`, `lyrics` (where available)

## Technology Stack

| Layer | Tools |
|---|---|
| Frontend | React 18, Vite, TailwindCSS, Axios, Lucide React |
| Backend | FastAPI, LangChain, LangGraph, Pydantic |
| LLM | OpenRouter (configurable, used only for SQL agent + hybrid synthesis) |
| Knowledge Graph | NetworkX (in-memory directed graph with pre-computed aggregations) |
| Vector DB | Qdrant (HNSW index, cosine similarity) |
| SQL DB | SQLite |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (lyrics), `all-MiniLM-L6-v2` (tags) |
| Infrastructure | Docker, Uvicorn, Loguru |

## Reports

Project documentation lives in `reports/`:
- `data_quality_report.md` — Music4All dataset analysis
- `sqldb_report.md` — SQLite schema and statistics
- `embedding_options_report.md` — Embedding model comparisons
- `vector_embeddings_report.md` — Phase C implementation details
- `api_options.md` — LLM API options and recommendations
- `notes.md` — Progress tracker and open decisions
- `project_plan.md` — Full project specification

## License

MIT
