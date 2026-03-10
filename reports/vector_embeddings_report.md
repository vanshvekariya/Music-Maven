# Vector Embeddings Report — Music4All Phase C

_Generated: Phase C completion_

---

## 1. Overview

Phase C adds semantic (vector) search on top of the existing SQL layer. The core
idea is that SQL is great for structured filtering ("songs with tempo > 140") but
terrible at subjective or language-level queries ("songs that feel melancholic at
3 AM"). Embeddings let us encode meaning as numbers so the system can match by
semantic similarity.

Two Qdrant collections were created:

| Collection | Content embedded | Model | Dimension |
|---|---|---|---|
| `music4all_lyrics` | Full lyric text per song | `paraphrase-multilingual-MiniLM-L12-v2` | 384 |
| `music4all_tags`   | artist + song_name + genres + tags | `all-MiniLM-L6-v2` | 384 |

---

## 2. What Is an Embedding?

An embedding is a dense vector (a list of numbers) that captures the *meaning* of
a piece of text. Two sentences that mean the same thing will produce similar vectors
even if they share no words.

**Example:**

| Text | Conceptual position |
|---|---|
| "heartbreak and longing" | near "sad love song" |
| "upbeat dance floor energy" | near "high danceability" |
| "guitar riff heavy metal" | near "rock guitar" |

The model learns these relationships from hundreds of millions of training
sentences. We reuse a pre-trained model — no training needed on our side.

---

## 3. Why Two Separate Collections?

### Lyrics collection (`music4all_lyrics`)

- Only ~40–60 % of songs have usable lyrics (remainder are instrumental).
- Lyrics express mood, story, emotion — very different vocabulary from metadata.
- Uses a **multilingual model** because the dataset contains songs in pt, es, en,
  ko, and many other languages. Mixing a monolingual model with multilingual text
  produces poor similarity scores.

### Tags collection (`music4all_tags`)

- Covers every song regardless of whether lyrics exist.
- Short text (artist + title + genres + tags) is better handled by a faster model.
- English-only tags make the lighter `all-MiniLM-L6-v2` model sufficient here.

> **Key concept — model selection matters.**
> `paraphrase-multilingual-MiniLM-L12-v2` knows 50+ languages but is 12 layers
> deep. `all-MiniLM-L6-v2` is only 6 layers, 2× faster, but English-only. Use the
> right tool for each content type.

---

## 4. Files Changed

### New files

| File | Purpose |
|---|---|
| `src/data/embedding_pipeline.py` | Reads SQLite → embeds → upserts to Qdrant |
| `src/agents/vector_agent.py` | Complete rewrite for Music4All two-collection search |

### Modified files

| File | Change |
|---|---|
| `src/config/settings.py` | Added `lyric_collection_name`, `tag_collection_name`, `lyric_embedding_model` |
| `src/vectordb/client.py` | `QdrantManager.__init__` now accepts optional `collection_name` and `vector_size` overrides |
| `src/vectordb/operations.py` | Added `index_documents_with_offset()` for batch-safe indexing |
| `src/main.py` | Changed default `enable_vector=True` |

---

## 5. Embedding Pipeline — How It Works

```
SQLite (songs table)
       │
       ▼
  Load ~109k rows
       │
       ├──► Pass 1: Lyrics
       │         Read lyrics/<song_id>.txt
       │         Skip if file missing OR length < 50 chars (instrumentals)
       │         Embed with multilingual model → 384-dim vector
       │         Upsert to Qdrant: music4all_lyrics
       │
       └──► Pass 2: Tags
                 Build string: artist + song_name + genres + tags
                 Embed with English model → 384-dim vector
                 Upsert to Qdrant: music4all_tags
```

### Batching

Embedding and indexing are done in batches of 256 songs. This is important for
two reasons:

1. **Memory** — 109k × 384 floats × 4 bytes ≈ 168 MB if loaded all at once.
   Batching keeps peak RAM under 1 MB per batch.
2. **Error recovery** — if a batch fails, only that batch needs to be retried.

### Global point IDs

Qdrant requires every point to have a unique integer ID within a collection.
The `index_documents_with_offset(offset=start)` method passes the batch's
starting index as the base, so IDs are unique across all batches without a
global counter.

---

## 6. Vector Agent — Query Routing

The `VectorAgent` automatically picks which collection(s) to search:

| Query contains | Collection searched |
|---|---|
| "lyric / mood / sad / heartbreak / feel / theme" | `music4all_lyrics` only |
| Everything else | `music4all_tags` only |
| Both signals present | Both → merged by cosine score |

### Score-based fusion

When both collections are searched, results are merged by highest cosine
similarity per song. A more sophisticated approach for production would be
**Reciprocal Rank Fusion (RRF)**: instead of raw scores, each result is
weighted by 1/(rank + 60), which is more robust when the two scores are on
slightly different scales. This is flagged as a future improvement.

---

## 7. Distance Metric — Cosine Similarity

Both collections use **cosine similarity** (configured in Qdrant as
`Distance.COSINE`). Why cosine rather than Euclidean distance?

- Sentence-Transformer models output `normalize_embeddings=True` by default —
  meaning vectors are unit-length.
- For unit vectors, cosine similarity and dot product are equivalent, and both
  are faster to compute than Euclidean distance.
- Cosine similarity is direction-only; it ignores vector magnitude, making it
  robust to documents of different lengths.

---

## 8. How to Run the Pipeline

Prerequisites: Qdrant running locally via Docker.

```bash
# 1. Start Qdrant
docker-compose up -d

# 2. Activate virtual environment
source venv/bin/activate

# 3. Run the embedding pipeline (first time)
python -m src.data.embedding_pipeline

# 4. To wipe and rebuild collections from scratch
python -m src.data.embedding_pipeline --recreate
```

Expected runtime on CPU: ~20–40 minutes for ~109k songs (both passes).

---

## 9. HNSW Index (Qdrant's ANN Algorithm)

Qdrant uses **HNSW (Hierarchical Navigable Small World)** graphs for
approximate nearest-neighbour (ANN) search. Understanding this is helpful for
later optimisation.

- Exact nearest-neighbour search over 109k × 384 vectors requires comparing
  the query vector to every stored vector — O(n) and very slow.
- HNSW builds a multi-layer graph where nodes are vectors and edges connect
  nearby nodes. Search starts at the top (sparse) layer and greedily follows
  edges toward the query vector, diving to deeper layers to refine.
- This gives ~99 % recall at 10–100× less compute than exact search.
- Qdrant tunes this automatically but exposes `m` (number of edges per node)
  and `ef_construct` (build-time accuracy) as knobs for Phase 3 tuning.

---

## 10. Future Improvements (Phase D+)

| Improvement | Benefit |
|---|---|
| Audio embeddings (CLAP / wav2vec2) | Query by sound, not just text |
| Reciprocal Rank Fusion across collections | Better score calibration |
| Payload indexing on `lang`, `artist` | Fast filtered vector search |
| Lyric chunking | Better retrieval for long songs (index verses separately) |
| Fine-tuning embedding model on music data | Better domain-specific similarity |

---

## 11. Open Decisions

- **Lyric chunking**: Currently one vector per song. Chunking by verse and
  aggregating could improve recall for long songs — left for Phase D.
- **Tag language**: Tags appear to be English only; if non-English tags
  appear after deeper inspection, the tag model should also be switched to
  the multilingual variant.
