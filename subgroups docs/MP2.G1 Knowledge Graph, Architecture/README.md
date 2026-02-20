# Music Maven: MP2.G1 – Knowledge Graph & Architecture

**Course:** CSC475 / CSC575 – Music Information Retrieval  
**Date:** February 2026

---

## Abstract

Existing music information retrieval systems often separate structured metadata queries from semantic content search. Music Maven MP2.G1 aims to bridge this gap with a multi-agent retrieval and analytics system built on the Music4All dataset. We use a query routing architecture that dispatches natural language queries to specialized SQL, vector, and graph search agents. The base system supports metadata analytics and text/lyrics retrieval, with a roadmap for knowledge graph integration, session-aware conversations, and integration with other subgroup modules (tempo, recommender, artist classifier, lyrics, genre). This specification outlines the system design, implementation plan, and team roles.

---

## 1. Introduction

The goal of this project is to create a music-specific chatbot that can play and analyze music as part of the conversation. We aim to use open-source and commercial LLMs to create structured queries, support various ways of asking the same question, and combine analyzed audio information with knowledge from the LLM to answer queries in contextually useful ways.

MP2.G1 is responsible for the central architecture that other subgroups (recommender, tempo analysis, artist classifier, lyrics search, genre classification) will integrate with. Our contributions include:

- A modular plugin-based architecture so subgroups can register modules without touching the core orchestrator
- Multi-source retrieval: vector (Qdrant), SQL (PostgreSQL), and graph (Neo4j)
- An LLM-based query router that classifies intent and routes to the right retrieval path
- A RAG pipeline that grounds responses in music-domain knowledge

---

## 2. Related Work

Music information retrieval (MIR) has evolved significantly over the past two decades, with foundational surveys [2, 4] outlining techniques for music search and analysis. Beyond simple metadata lookup, semantic annotation and content-based retrieval [3] enable richer querying over musical attributes. Our system builds on these foundations by combining structured database queries with semantic search over lyrics and metadata.

Efficient semantic search at scale relies on learned text representations. Sentence-BERT [9] provides sentence-level embeddings that have become standard for dense retrieval. Vector databases such as Qdrant [11] leverage HNSW indexing [10] for approximate nearest neighbor search, enabling sub-linear lookup over large embedding spaces. We adopt this pipeline for lyrics and song metadata to support open-ended natural language queries.

Knowledge graphs offer a complementary representation for relationship-aware retrieval. The Music Ontology [12] provides a formal vocabulary for describing musical entities and their relations. Oramas et al. [13] demonstrate how graph-structured data supports queries over artist–genre and album–song relationships. Our architecture extends this line of work by constructing a knowledge graph from Music4All metadata and inferring additional relationships from shared attributes across artists, genres, and tags.

Recent advances in large language models (LLMs) [14, 15] and retrieval-augmented generation (RAG) [19] enable natural language interfaces that ground responses in retrieved documents. General-purpose LLMs lack domain-specific music knowledge; RAG addresses this by combining retrieval with generation. We extend the RAG paradigm by routing queries across multiple retrieval backends—vector, SQL, and graph—rather than a single source.

Content-based and similarity-based music recommendation [16, 17] inform our retrieval design, particularly for artist and genre-based queries. Multimodal learning frameworks [18] provide conceptual grounding for combining text, audio, and metadata signals, which we plan to explore in later phases of the project.

---

## 3. Data

We use the Music4All dataset [1], which contains:

- **109,269 songs** with unique identifiers and 30-second audio clips
- **16,269 unique artists** across 38,363 albums
- **Lyrics** with language detection (46 languages)
- **Spotify audio features:** tempo, key, mode, danceability, energy, valence
- **Popularity scores** (0–100)
- **Genre and tag annotations:** 853 genre tags, 19,541 user tags
- **15,602 user listening histories** for recommendation research

We also use a lyrics dataset from Kaggle to extend semantic search over song text. The combination of metadata, lyrics, and audio makes Music4All well-suited for multimodal retrieval.

---

## 4. System Design

### 4.1 Architecture Overview

Music Maven MP2.G1 uses a multi-agent architecture:

1. **Query Router:** LLM-based classifier that determines intent (SQL, vector, graph, or hybrid)
2. **SQL Agent:** Analytical queries over structured metadata in PostgreSQL
3. **Vector Agent:** Semantic search over text embeddings in Qdrant
4. **Graph Agent:** Relationship queries (artist–genre, album–song) in Neo4j
5. **Orchestrator:** Coordinates agents and synthesizes responses (LangGraph)
6. **API Layer:** FastAPI backend
7. **Frontend:** React chat interface

### 4.2 Supported Query Types

**Basic (Layer 1)**  
Show me songs by artist X; find songs with tempo above 120 BPM; what genres are most popular; list high-energy dance tracks; search lyrics for specific words; top 10 most popular songs; songs released between years; find songs tagged as "chill."

**Expected (Layer 2)**  
Play a song similar to [title]; which artists collaborated with X; show songs in the same album; find upbeat songs in minor key; recommend songs by artists in same genre; relationship-based questions (which genres influenced this artist?).

**Advanced (Layer 3)**  
Based on previous queries, find more like that; analyze this song's mood and recommend similar; session-aware conversations; analyzing voice notes and other input audio.

---

## 5. Implementation Plan

We use a three-layer iterative approach.

### Layer 1: Basic System

Delivers: SQL analytics over metadata; text embeddings for lyrics/tags/genres (sentence-transformers); vector search via Qdrant with metadata filtering; REST API and basic UI.

**Tech stack:** Python, FastAPI, PostgreSQL, Qdrant, sentence-transformers (all-MiniLM-L6-v2), React.

### Layer 2: Expected Enhancements

Builds on Layer 1: knowledge graph construction (artist–album–genre–tag relationships) in Neo4j; hybrid retrieval combining vector and SQL; query routing with relationship awareness; conversation memory for multi-turn interactions; FastAPI endpoints for subgroups to register modules.

### Layer 3: Advanced Features

Production-ready: inferred relationships (artist similarity from shared genres); session memory for context-aware responses; caching (Redis or in-memory); performance tuning; Docker deployment; comprehensive testing.

---

## 6. Phases and Milestones

| Phase | Deliverables |
|-------|--------------|
| A | Requirements finalized, data schema mapped, query list confirmed |
| B | Metadata ingestion complete, SQL analytics functional |
| C | Text embeddings indexed, vector search end-to-end |
| D | Knowledge graph schema and ingestion, graph query agent |
| E | UI/API integration, hybrid retrieval, report draft complete |
| Future | Audio embeddings, session memory, subgroup module integration |

---

## 7. Planned Evaluation

*To be determined.* We plan to evaluate through offline retrieval metrics (precision@k, recall@k) on labeled subsets, qualitative case studies with representative queries, and response latency benchmarks. Detailed methodology will be specified in subsequent iterations.

---

## 8. Individual Contributions

### Vansh Vekaria

**Objective 1: Modular Architecture & Orchestration**
- PI.1 (Basic): Design and document plugin-based architecture with BaseModule interface and module registry
- PI.2 (Basic): Set up development environment with PostgreSQL, Qdrant, and Neo4j
- PI.3 (Expected): Implement configuration management (Pydantic, .env) and secure secrets handling
- PI.4 (Expected): Implement query router that classifies and routes to vector/SQL/hybrid/graph
- PI.5 (Advanced): Build multi-agent orchestrator with LangGraph

**Objective 2: LLM Integration & RAG Pipeline**
- PI.1 (Basic): Set up LLM integration with Ollama or Groq; create LLM manager with error handling
- PI.2 (Basic): Implement vector retrieval and semantic similarity search via Qdrant
- PI.3 (Expected): Build RAG pipeline (retrieval + prompts + generation) for music queries
- PI.4 (Expected): Create ChatbotAgent module with query processing and response formatting
- PI.5 (Advanced): Generate embeddings for songs/lyrics; set up Qdrant collections

**Objective 3: Multi-Source Retrieval & Conversation Memory**
- PI.1 (Basic): Implement SQL-based metadata retrieval with natural language to SQL
- PI.2 (Basic): Build hybrid retrieval combining vector search and SQL with ranking
- PI.3 (Expected): Implement conversation memory and context window management
- PI.4 (Expected): Add conversation summarization for long chats
- PI.5 (Advanced): Optimize retrieval with result deduplication and fusion

---

### Aishwarya Pujitha

**Objective 1: Data Pipeline & Database Design**
- PI.1 (Basic): Load Music4All and Kaggle lyrics into structured format; parse and validate data
- PI.2 (Basic): Design and document PostgreSQL schema for songs, artists, albums, genres, tags, lyrics
- PI.3 (Expected): Implement batch ingestion with cleaning, normalization, and multi-value field handling
- PI.4 (Expected): Create indexes and constraints; verify referential integrity
- PI.5 (Advanced): Add data validation layer and automated schema consistency checks

**Objective 2: Knowledge Graph Implementation**
- PI.1 (Basic): Define Neo4j node types (Song, Artist, Genre, Album, Tag) and relationship types (PERFORMED, HAS_GENRE, etc.)
- PI.2 (Basic): Ingest entities and relationships from Music4All into Neo4j
- PI.3 (Expected): Build graph query agent (Cypher interface) and integrate with RAG
- PI.4 (Expected): Implement entity resolution and deduplication for artist/genre/tag names
- PI.5 (Advanced): Extract inferred relationships (e.g., artist similarity from shared genres/tags)

**Objective 3: Frontend & API Integration**
- PI.1 (Basic): Set up React + Vite chat interface with message list, input box, loading states
- PI.2 (Basic): Implement FastAPI `/chat` and `/health` endpoints with request validation
- PI.3 (Expected): Connect frontend to backend; implement streaming and error handling
- PI.4 (Expected): Add example queries, markdown formatting, conversation history
- PI.5 (Advanced): Create Swagger/OpenAPI documentation and integration guide

---

## 9. Conclusion

Music Maven MP2.G1 aims to provide a unified, conversational interface for music retrieval and analytics. The base system will support SQL-based metadata queries and semantic text search, with a clear path toward knowledge graph integration, session-aware conversations, and integration with tempo, recommender, and other subgroup modules. We are building iteratively to ensure a functional system at each phase.

---

## References

[1] I. A. P. Santana et al., "Music4All: A New Music Database and Its Applications," in Proc. IWSSIP, 2020.

[2] J. S. Downie, "Music Information Retrieval," Annual Review of Information Science and Technology, 2003.

[3] D. Turnbull et al., "Semantic Annotation and Retrieval of Music and Sound Effects," IEEE TASLP, 2008.

[4] M. Schedl et al., "Music Information Retrieval: Recent Developments and Applications," Foundations and Trends in IR, 2014.

[5] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," EMNLP, 2019.

[6] Y. A. Malkov and D. A. Yashunin, "Efficient and Robust Approximate Nearest Neighbor Search Using HNSW Graphs," IEEE TPAMI, 2020.

[7] Qdrant Team, "Qdrant Vector Database Documentation," https://qdrant.tech/documentation/, 2024.

[8] Y. Raimond et al., "The Music Ontology," ISMIR, 2007.

[9] S. Oramas et al., "A Knowledge Graph for Music Retrieval," ISMIR, 2017.

[10] T. Brown et al., "Language Models are Few-Shot Learners," NeurIPS, 2020.

[11] H. Touvron et al., "LLaMA: Open and Efficient Foundation Language Models," arXiv, 2023.

[12] A. van den Oord et al., "Deep Content-Based Music Recommendation," NeurIPS, 2013.

[13] P. Knees and M. Schedl, "A Survey of Music Similarity and Recommendation from Music Context Data," ACM CSUR, 2016.

[14] T. Baltrusaitis et al., "Multimodal Machine Learning: A Survey and Taxonomy," IEEE TPAMI, 2019.

[15] P. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," NeurIPS, 2020.
