# Music Maven — Phase E Milestone Report (Draft)

**Course:** CSC475 / CSC575 — Music Information Retrieval  
**Project:** MP2.G1 — Knowledge Graph & Multi-Agent Architecture  
**Dataset:** Music4All (~109k songs)  
**Date:** April 2026  
**Status:** Draft for team review before final submission

---

## 1. Executive summary

Music Maven is a multi-agent music information retrieval system over the Music4All dataset. Users ask natural-language questions; a local **KG-aware router** sends each query to one or more of: a **Knowledge Graph** (instant factual answers, no LLM), a **SQL agent** (LLM-generated analytics over SQLite), and a **vector agent** (local embeddings + Qdrant over lyrics and tag/metadata collections). A **FastAPI** backend and **React** frontend provide the demo surface. Phase E completes integration polish: **language-filtered vector search**, **subgroup plugin hooks** for other teams’ services, and this **milestone report draft**.

---

## 2. System architecture (current)

| Layer | Component | Role |
|-------|-----------|------|
| Data | SQLite `songs`, optional `listening_history` | Structured metadata and MIR features |
| Data | Qdrant `music4all_lyrics`, `music4all_tags` | Semantic search (multilingual lyrics + English-heavy tags) |
| Data | NetworkX KG (`data/music4all_kg.gpickle`) | Precomputed leaderboards, distributions, entity links |
| Routing | `KGQueryRouter` | Regex + KG `try_answer` + keyword scoring → `kg_direct`, `sql`, `vector`, `hybrid` |
| Orchestration | LangGraph (`MultiAgentOrchestrator`) | Route → KG / SQL / Vector → synthesize |
| LLM | OpenRouter (configurable model) | SQL agent, hybrid SQL+vector synthesis; not used on KG-direct or pure vector formatting |
| UI | React + Vite + Tailwind | Query box, optional language filter, results as markdown + song cards |

**Why this split:** Factual questions that match KG templates avoid cost and latency of the LLM. Semantic questions use **local** sentence-transformers so embeddings do not require an API call. The LLM is reserved for dynamic SQL and for merging SQL + vector outputs when both run.

---

## 3. Phase E deliverables

### 3.1 Language filter for vector search

**Problem (from project notes):** Queries such as “French sad songs” treated “French” as semantic text, not as metadata.

**Solution:**  
- API accepts optional `lang_filter` (ISO-style code, e.g. `en`, `pt`, `fr`), passed into the orchestrator state and applied as a Qdrant payload filter on field `lang` for vector searches.  
- Optional **heuristic**: if the client does not send `lang_filter`, the vector path may infer a language from common English names in the query (e.g. “Portuguese” → `pt`) and apply the same filter.  
- **Limitation:** Heuristic can misfire on ambiguous text; explicit `lang_filter` from the UI is preferred for demos.

**Teaching note:** This is standard **metadata filtering** on top of **vector similarity**: similarity narrows “what it’s about,” `lang` narrows “which language bucket,” improving precision for multilingual catalogs.

### 3.2 Subgroup (plugin) integration

**Problem:** Other subgroups (tempo, artist classifier, lyrics classification/search) ship on separate branches or services with no single agreed contract.

**Solution:**  
- **Registry:** `GET /plugins/status` lists named modules (`tempo`, `artist_classifier`, `lyrics`) and whether each is configured.  
- **Forward:** `POST /plugins/{module}/forward` accepts a JSON body and, if that module’s base URL is set in environment variables, **proxies** the request to the subgroup service and returns its response.  
- **Configuration:** Optional `.env` keys (see §6). Teams keep ownership of their HTTP API; Music Maven does not embed their code until URLs are agreed.

**Teaching note:** This is a **stable integration boundary** (adapter pattern): the core app stays unchanged when a subgroup swaps implementation as long as the HTTP shape is compatible.

### 3.3 Reliability and UX fixes (recent)

- Unrouted queries (`UNKNOWN` / empty agent list) now reach **synthesis** with a helpful fallback message instead of an empty answer.  
- KG template ordering and sanity checks reduce wrong answers when the user asks for **aggregates** (e.g. average tempo) but the text also contains “songs by …”.  
- SQL agent system prompt **column list** aligned with the real SQLite `songs` schema to reduce “no such column” errors.  
- API returns a structured error response for unexpected failures instead of always raising HTTP 500 for handler bugs.

---

## 4. Evaluation (suggested for final report)

| Criterion | How to demonstrate |
|-----------|-------------------|
| KG path | Example: “Who are the top 10 most popular artists?” — fast, no LLM. |
| SQL path | Example: “Which album has the highest average danceability?” — requires valid API key / model. |
| Vector path | Example: “Songs about heartbreak” — local embeddings + Qdrant. |
| Language filter | Same query with `lang_filter: "pt"` vs none — compare result languages. |
| Plugins | `GET /plugins/status` with/without env URLs; `POST .../forward` when a teammate’s service is up. |
| Failure modes | Empty or nonsense query → fallback text, not crash. |

---

## 5. Risks and open items

- **Free LLM rate limits** (OpenRouter) affect SQL and hybrid synthesis; paid tier or different model recommended for demos.  
- **Memory:** Loading KG + two embedding models is heavy; exit code 137 on small VMs indicates OOM — use one machine with sufficient RAM or disable reload in production.  
- **Plugin contract:** Subgroups should document their expected JSON body and response; adjust `forward` path or env URL (including path suffix) per team.  
- **PostgreSQL vs SQLite** (see `reports/notes.md`) remains an open decision before shared deployment.

---

## 6. Configuration reference (Phase E)

**Vector / query (API JSON):**

```json
{
  "query": "sad love songs",
  "max_results": 15,
  "lang_filter": "en"
}
```

**Optional subgroup URLs (`.env`):**

```text
PLUGIN_TEMPO_URL=
PLUGIN_ARTIST_CLASSIFIER_URL=
PLUGIN_LYRICS_URL=
```

When unset, plugin endpoints report `configured: false` and forward returns 503 with a clear message.

---

## 7. Conclusion (draft)

Phase E closes the loop between **architecture** (KG + SQL + vector + hybrid), **usability** (language-scoped search, safer routing), and **team integration** (plugin registry and forward). Final submission should add measured latency/cost notes, screenshots of the UI, and any results from joint testing with subgroup services.

---

*End of draft — edit for voice, add author names, figures, and exact course submission formatting before hand-in.*
