# Embedding Options Report

**Project:** Music Maven  
**Date:** 2026-03-05  
**Scope:** Text embeddings (Phase 2), Audio + Multimodal embeddings (Phase 3)

---

## What We Are Embedding

Each song in Music4All has three text sources and one future audio source:

| Source | Content | Available Now |
|---|---|---|
| Lyrics | Full song text — mood, themes, narrative | Yes (`lyrics/<song_id>.txt`) |
| Tags | User-generated labels e.g. `"rock, classic, guitar"` | Yes (`id_tags.csv`) |
| Genres | Structured categories e.g. `"pop, electronic"` | Yes (`id_genres.csv`) |
| Audio clips | 30-second `.mp3` files | Phase 3 |

---

## Phase 2 — Text Embedding Options

### Option 1: `all-MiniLM-L6-v2` (currently configured)
- **Dimensions:** 384  
- **Model size:** 22 MB  
- **Language:** English only  
- **Speed:** Very fast (~14k sentences/sec on CPU)  
- **Good for:** Tags, genres, short text matching  
- **Weakness:** Poor on non-English lyrics. ~23% of the dataset is Portuguese, Spanish, Korean — this model will give poor semantic matches for those songs  
- **When to use:** Quick prototyping, tag/genre-only collections

---

### Option 2: `all-mpnet-base-v2`
- **Dimensions:** 768  
- **Model size:** 420 MB  
- **Language:** English only  
- **Speed:** ~2,800 sentences/sec on CPU  
- **Good for:** Higher quality English semantic search than MiniLM  
- **Weakness:** Still English-biased, larger memory footprint, slower  
- **When to use:** English-only lyric search where quality matters more than speed

---

### Option 3: `paraphrase-multilingual-MiniLM-L12-v2` ⭐ Recommended for lyrics
- **Dimensions:** 384  
- **Model size:** 118 MB  
- **Language:** 50+ languages including English, Portuguese, Spanish, Korean  
- **Speed:** ~7,500 sentences/sec on CPU  
- **Good for:** Lyric search across all languages in the dataset. "melancholy" and "melancolia" (Spanish) land near each other in vector space  
- **Weakness:** Slightly lower quality than `all-mpnet-base-v2` on English-only text  
- **When to use:** Primary lyrics collection — handles the full multilingual dataset  
- **Note:** Same 384 dimensions as the currently configured model — no schema changes needed in Qdrant or `settings.py`, just a model name swap

---

### Option 4: `paraphrase-multilingual-mpnet-base-v2`
- **Dimensions:** 768  
- **Model size:** 278 MB  
- **Language:** 50+ languages  
- **Speed:** ~2,500 sentences/sec on CPU  
- **Good for:** Best quality multilingual embeddings for lyrics  
- **Weakness:** 768 dimensions doubles Qdrant storage and index memory vs. 384-dim models  
- **When to use:** If search quality on non-English lyrics is a priority and memory is not a constraint

---

### Option 5: `all-MiniLM-L6-v2` (for tags/genres only) ⭐ Recommended for tags
- Same as Option 1, but used in a **dedicated tags collection** rather than for lyrics  
- Tags and genres are English-only, short strings — MiniLM handles this perfectly  
- Keeps the lyrics collection and tags collection independent so they can be searched and weighted separately

---

## Recommended Phase 2 Architecture — Two Collections

Rather than one collection mixing lyrics and tags, use two separate Qdrant collections:

| Collection | Model | Dimensions | Content |
|---|---|---|---|
| `music4all_lyrics` | `paraphrase-multilingual-MiniLM-L12-v2` | 384 | Full lyric text per song |
| `music4all_tags` | `all-MiniLM-L6-v2` | 384 | `tags + genres + artist + song_name` concatenated |

**Why two collections?**
- Lyric queries ("find melancholic songs") should search `music4all_lyrics`
- Style queries ("indie folk acoustic") should search `music4all_tags`
- The vector agent can search both and merge/re-rank results by score
- Each collection is independently tunable — you can swap the lyrics model later without affecting tags

**Settings changes needed:**
```python
# settings.py additions
lyric_embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
tag_embedding_model: str = "all-MiniLM-L6-v2"
lyric_collection_name: str = "music4all_lyrics"
tag_collection_name: str = "music4all_tags"
vector_size: int = 384  # same for both models — no change needed
```

---

## Phase 3 — Audio Embedding Options

When the 30-second audio clips (`.mp3` files) become available, these models can embed raw audio into vectors, enabling queries like "find songs that sound like this" or "identify the mood from audio alone."

### Option A: `CLAP` (Contrastive Language-Audio Pretraining) ⭐ Top recommendation
- **Model:** `laion/clap-htsat-unfused` (HuggingFace)  
- **Dimensions:** 512  
- **What it does:** Embeds both **text and audio in the same vector space** — a text query like "upbeat dance music" and an audio clip of a dance track will have similar vectors. This is the audio equivalent of CLIP for images  
- **Why it's powerful:** Enables cross-modal search — query with text, retrieve by audio similarity, or vice versa  
- **Requirements:** 30-second clips (exactly what Music4All provides), GPU recommended for batch processing  
- **When to use:** Primary audio embedding model for Phase 3

---

### Option B: `OpenL3`
- **Model:** `openl3` (Python package)  
- **Dimensions:** 512 or 6144  
- **What it does:** Learns audio representations from video/audio co-occurrence. Good for general audio similarity  
- **Weakness:** Not text-aligned — cannot query with natural language, only audio-to-audio similarity  
- **When to use:** Supplementary audio similarity when you have a reference audio clip

---

### Option C: `VGGish` (Google)
- **Model:** `torch.hub` / `tensorflow_hub`  
- **Dimensions:** 128  
- **What it does:** CNN trained on YouTube-8M audio. Produces compact audio fingerprints  
- **Weakness:** Low-dimensional (128), older architecture, not text-aligned  
- **When to use:** Lightweight audio fingerprinting, not recommended as primary model

---

### Option D: `wav2vec 2.0` (Facebook/Meta)
- **Model:** `facebook/wav2vec2-base` (HuggingFace)  
- **Dimensions:** 768  
- **What it does:** Self-supervised speech/audio representation learning. Excellent for speech-heavy content  
- **Weakness:** Optimised for speech, not general music  
- **When to use:** If lyric-audio alignment is needed (matching sung lyrics to audio)

---

### Option E: `music2vec` / `mert-v1-95M`
- **Model:** `m-a-p/MERT-v1-95M` (HuggingFace)  
- **Dimensions:** 768  
- **What it does:** Music-specific self-supervised model trained entirely on music audio (not speech). State-of-the-art for music understanding tasks  
- **Weakness:** Newer model, less community tooling than CLAP  
- **When to use:** When pure music understanding quality matters most (genre classification, mood detection from audio)

---

## Phase 3 Recommended Architecture — Multimodal Fusion

When audio clips are available, add a third Qdrant collection:

| Collection | Model | Dimensions | Content |
|---|---|---|---|
| `music4all_lyrics` | `paraphrase-multilingual-MiniLM-L12-v2` | 384 | Lyrics text |
| `music4all_tags` | `all-MiniLM-L6-v2` | 384 | Tags + genres + metadata |
| `music4all_audio` | `laion/clap-htsat-unfused` (CLAP) | 512 | 30-sec audio clips |

**Multimodal fusion** means combining scores from all three collections to rank results:

```
final_score = w1 * lyric_score + w2 * tag_score + w3 * audio_score
```

The weights (`w1`, `w2`, `w3`) can be tuned per query type:
- "Find melancholic songs" → heavy lyric weight
- "Indie folk acoustic" → heavy tag weight  
- "Songs that sound like this clip" → heavy audio weight

> This is the core of multimodal MIR research — learning how to combine signals from different modalities to best represent musical meaning.

---

## Summary Table

| Model | Phase | Dimensions | Multilingual | Audio | Recommended |
|---|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 2 | 384 | No | No | Yes — tags collection |
| `paraphrase-multilingual-MiniLM-L12-v2` | 2 | 384 | Yes (50+) | No | Yes — lyrics collection |
| `all-mpnet-base-v2` | 2 | 768 | No | No | Optional upgrade |
| `paraphrase-multilingual-mpnet-base-v2` | 2 | 768 | Yes (50+) | No | If quality > memory |
| CLAP (`laion/clap-htsat-unfused`) | 3 | 512 | Yes (text+audio) | Yes | Yes — audio collection |
| `MERT-v1-95M` | 3 | 768 | No | Yes | Supplementary |
| `wav2vec2-base` | 3 | 768 | No | Yes (speech) | Not recommended |
| `VGGish` | 3 | 128 | No | Yes | Not recommended |
| `OpenL3` | 3 | 512 | No | Yes (no text) | Supplementary |
