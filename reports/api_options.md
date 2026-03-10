# LLM API Options Report

**Project:** Music Maven — MP2.G1  
**Date:** 2026-03-05  
**Scope:** LLM selection across all agents — SQL, Vector, Graph, Synthesizer, Router

---

## Current Configuration

- **Provider:** OpenRouter (`https://openrouter.ai/api/v1`)
- **Model:** `openai/gpt-oss-20b:free`
- **Used for:** All agents (router, SQL agent, synthesizer)
- **Status:** Working for SQL queries

---

## LLM Roles in the Full System

| Role | Layer | What it needs to do |
|---|---|---|
| Query Router | All layers | Classify intent: SQL / vector / graph / hybrid — runs on every single query |
| SQL Agent | Layer 1 | Generate SQL against the `songs` table, interpret tabular results |
| Vector Agent | Layer 1 | Format and summarise semantic search results from Qdrant |
| Graph Agent | Layer 2 | Generate **Cypher** queries for Neo4j, traverse artist–genre–album relationships |
| Synthesizer / Orchestrator | All layers | Combine results from multiple agents into one coherent answer |
| Audio Analysis (future) | Future | Interpret audio features, describe mood and recommend from signal data |

---

## Free Models Available on OpenRouter

### `mistralai/mistral-7b-instruct:free`
- **Size:** 7B parameters
- **Good for:** Fast classification, structured instruction following
- **Weakness:** Less capable on complex multi-step reasoning
- **Best role:** Query Router — simple classification task, must be low-latency

### `meta-llama/llama-3.1-8b-instruct:free`
- **Size:** 8B parameters
- **Good for:** General instruction following, summarisation
- **Weakness:** Smaller context window, weaker on SQL/Cypher generation
- **Best role:** Vector Agent — only needs to format/summarise search results, no query generation

### `meta-llama/llama-3.3-70b-instruct:free`
- **Size:** 70B parameters
- **Good for:** Strong SQL generation, multi-step reasoning, clean structured answers
- **Weakness:** Slower, rate limited on free tier
- **Best role:** SQL Agent and Synthesizer — quality matters most here

### `qwen/qwen-2.5-72b-instruct:free`
- **Size:** 72B parameters
- **Good for:** Excellent at structured/code tasks (SQL and Cypher), multilingual, strong reasoning
- **Weakness:** Rate limits on free tier
- **Best role:** Graph Agent — Cypher (Neo4j query language) is rare in training data and Qwen 2.5 handles it best among free models; also good for multilingual music data

### `google/gemma-2-9b-it:free`
- **Size:** 9B parameters
- **Good for:** Instruction following, general tasks
- **Weakness:** Less tested for structured query generation (SQL/Cypher)
- **Best role:** Fallback/supplementary

---

## Recommended Model Assignment (Full System)

| Component | Recommended Model | Reason |
|---|---|---|
| Query Router | `mistralai/mistral-7b-instruct:free` | Runs on every query — keep it fast and lightweight |
| SQL Agent | `meta-llama/llama-3.3-70b-instruct:free` | Best free model for SQL generation and structured answers |
| Vector Agent | `meta-llama/llama-3.1-8b-instruct:free` | Only formats results — no complex generation needed |
| Graph Agent | `qwen/qwen-2.5-72b-instruct:free` | Best free option for Cypher query generation over Neo4j |
| Synthesizer | `meta-llama/llama-3.3-70b-instruct:free` | Needs to reason across SQL + vector + graph results simultaneously |

---

## Single Model Recommendation (Simplicity)

If managing multiple model configs is too much overhead, use **`qwen/qwen-2.5-72b-instruct:free`** for everything. It performs well across SQL, Cypher, multilingual text, and structured reasoning. Trade-off: slower than using a lightweight model for routing.

---

## Why Cypher Matters

Cypher is Neo4j's graph query language (Phase D). It is significantly less common in LLM training data than SQL, which means smaller or older models often generate broken Cypher syntax. When the Graph Agent is built, model selection matters more than for the SQL agent — Qwen 2.5 72B and Llama 3.3 70B are the two free models most likely to handle it reliably.

---

## Model Cascading Pattern

The recommended assignment uses a pattern called **model cascading**:
- Small, fast model for routing (runs every query)
- Larger, slower model for generation (runs only when needed)

This minimises average latency while preserving answer quality where it matters. Standard optimisation in production LLM systems.

---

## Action Items (To Revisit)

- [ ] Update query router to `mistralai/mistral-7b-instruct:free` when ready to optimise latency
- [ ] Switch SQL agent to `meta-llama/llama-3.3-70b-instruct:free` when testing quality
- [ ] Assign `qwen/qwen-2.5-72b-instruct:free` to the Graph Agent when Phase D begins
- [ ] Evaluate whether OpenRouter free tier rate limits become a bottleneck under load
- [ ] Investigate music-specific fine-tuned models (e.g. MusicLM, MusicGen) for audio analysis phase
