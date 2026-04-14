# Report helper — demo session evidence & notes

**Purpose:** Raw UI outcomes and short analysis for milestone / final write-ups.  
**Product:** Music Maven (Music4All, multi-agent: KG + SQL + vector).

---

## Session A — “Most common genres” + follow-up

### Turn 1 — KG-direct

- **Query:** What are the most common genres?
- **Route:** `kg_direct`
- **Latency / confidence (UI):** ~0.01s, 95%
- **Answer (abridged):** Top 10 genres by **song count** (dataset-wide, from KG aggregations):
  - rock — 25,731  
  - pop — 22,013  
  - electronic — 12,769  
  - alternative rock — 8,103  
  - indie rock — 7,943  
  - metal — 6,459  
  - folk — 6,106  
  - singer-songwriter — 5,793  
  - classic rock — 5,688  
  - soul — 5,458  

### Turn 2 — SQL follow-up

- **Query:** more please  
- **Route:** `sql` (continuation routing)  
- **Latency / confidence (UI):** ~5.17s, 88%  
- **Answer (abridged):** “More common genres” with **different counts** and labels, e.g. Pop 6,092, Rock 1,887, Rap1,431, … “Indie Rock, Rock” as a combined label.

### Notes for the report

- **KG** answers “most common genres” using **pre-aggregated graph stats** — fast, consistent definition (songs per genre node / leaderboard).
- **SQL follow-up** is **not guaranteed** to reproduce the same metric: the LLM may interpret “more” as a different query (e.g. parsing `genres` text field, substring counts, or LIMIT/OFFSET without matching KG semantics). **Counts and labels diverge** from Turn 1 — good **limitations / future work** bullet: align SQL continuation with KG definition or pass structured “next page” hints.
- **Teaching line:** Multi-agent demos should either **re-query KG with offset** for “more” or **freeze the exact SQL** from a template so follow-ups stay comparable.

---

## Session B — “Top 10 most popular artists” + follow-up

### Turn 1 — KG-direct

- **Query:** Who are the top 10 most popular artists?
- **Route:** `kg_direct`
- **Latency / confidence (UI):** ~0.01s, 95%
- **Answer (abridged):** Top 10 by **average popularity** (as surfaced by KG), many with **1 song** — e.g. Pedro Capó 88/100, Calvin Harris & Rag'n'Bone Man 83/100, …

### Turn 2 — SQL follow-up

- **Query:** 10 more please  
- **Route:** `sql`  
- **Latency / confidence (UI):** ~11s, 88%  
- **Answer (abridged):** “10 more popular artists” with avg popularity ~78 — Blueface, Dua Lipa & BLACKPINK, etc.

### Notes for the report

- **KG “top artists by popularity”** here reflects **artist-level aggregates in the graph** (e.g. avg popularity × song count); **single-track artists** can dominate the “top by avg popularity” list — valid dataset behaviour, worth **one sentence in the report** so reviewers aren’t surprised.
- **SQL “10 more”** should ideally mean **ranks 11–20** under the **same ranking rule** as KG; in practice the LLM infers from conversation and may approximate. **Report angle:** demonstrates **session memory + continuation routing**, but also **metric alignment** as an improvement area.
- **Teaching line:** For papers/demos, cite **KG for reproducible leaderboards**; use **SQL** for **ad hoc** questions where the user accepts LLM-generated query semantics.

---

## Cross-cutting UI / system tags (both sessions)

- **Memory:** Rolling summary (LLM) — post-reply summarization for follow-ups; not the cost of the KG instant path itself.
- **Agents shown:** `kg_direct` then `sql` — matches **zero-LLM classification** for structured questions, then **SQL** for short continuations (“more please”, “10 more please”).

---

## Suggested report bullets (copy-ready)

1. **Strength:** KG returns **deterministic, fast** answers for template questions (genres, top artists) with **no LLM cost** on the answer path.  
2. **Demonstrated:** **Multi-turn** flow routes follow-ups to **SQL** with **session memory** enabled.  
3. **Limitation:** **Follow-up answers may not be strictly comparable** to the initial KG answer (different SQL semantics, pagination not locked to KG ordering).  
4. **Future work:** **Shared ranking spec** for “more” (OFFSET from cached KG slice, or fixed SQL view matching KG definitions).

---

*Captured from live UI sessions for report drafting. Numbers and routes reflect that run; restart/API version may vary slightly.*
