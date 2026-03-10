# Music Maven — Frontend

A modern React frontend for the Music Maven Music Information Retrieval system.

## Features

- Natural language search across 109,000 songs
- Colour-coded example queries (SQL, Vector, Hybrid)
- Song cards displaying audio features: tempo, energy, danceability, valence, popularity
- System info panel showing agent status, LLM model, embedding models, and database connections
- Real-time health monitoring

## Tech Stack

- **Framework**: React 18 with Vite
- **Styling**: TailwindCSS
- **Icons**: Lucide React
- **Animations**: Framer Motion
- **HTTP Client**: Axios
- **Notifications**: React Hot Toast

## Quick Start

### Prerequisites

- Node.js 18+ and npm
- Backend API running on `http://localhost:8000`
- Qdrant running on `localhost:6333` (via Docker)

### Installation

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── ui/                  # Reusable UI primitives (Card, Button, Badge, Input)
│   │   ├── QueryInput.jsx       # Search bar with categorised example queries
│   │   └── ResultsDisplay.jsx   # SongCard results and AI answer display
│   ├── services/
│   │   └── api.js               # Axios API client (3 min timeout)
│   ├── utils/
│   │   └── cn.js                # Tailwind class merge utility
│   ├── App.jsx                  # Main application layout
│   ├── main.jsx                 # Entry point
│   └── index.css                # Global styles
├── index.html
├── package.json
├── vite.config.js
├── tailwind.config.js
└── postcss.config.js
```

## Query Types

| Badge | Description | Example |
|---|---|---|
| SQL | Structured database queries | "Who are the top 10 most popular artists?" |
| Vector | Semantic / mood / lyric search | "Chill lo-fi hip hop" |
| Hybrid | SQL filters + vector search | "Popular sad rock songs in English" |

## API Endpoints Used

- `GET /health` — health check
- `POST /query` — process a natural language query
- `GET /system/info` — agent status and configuration
- `GET /examples` — load example queries

## Build for Production

```bash
npm run build
```

Output goes to `dist/`.

## Troubleshooting

**Backend connection failed** — ensure `uvicorn src.api.main:app --port 8000` is running.

**CORS error** — check that your frontend port (5173 or 5174) is in the `allow_origins` list in `src/api/main.py`.

**Slow responses** — SQL queries depend on the OpenRouter free tier LLM; response times vary with API load. Vector queries are fast as they run locally.
