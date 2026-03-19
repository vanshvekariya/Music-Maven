## Aishwarya Pujitha

I'm focusing on the data pipeline, knowledge graph, and frontend/API. Here are my objectives and how I plan to measure progress.

### Objective 1: Data Pipeline & Database Design

- PI.1 (Basic): Load Music4All and Kaggle lyrics, parse into structured format, and validate data integrity
- PI.2 (Basic): Design and document the PostgreSQL schema (songs, artists, albums, genres, tags, lyrics)
- PI.3 (Expected): Implement batch ingestion with cleaning, normalization, and handling for multi-value fields like genres and tags
- PI.4 (Expected): Add indexes and constraints for performance; verify referential integrity
- PI.5 (Advanced): Build a data validation layer with automated checks for schema consistency

### Objective 2: Embedding Pipeline, Vector Database & LLM/RAG

- PI.1 (Basic): Set up Sentence-Transformers (all-MiniLM-L6-v2) and generate embeddings for song metadata
- PI.2 (Basic): Create Qdrant collections with payload schema and batch upload embeddings
- PI.3 (Basic): Set up LLM integration with OpenRouter free tier
- PI.4 (Basic): Build basic RAG pipeline for music queries
- PI.5 (Expected): Generate lyrics embeddings and load into separate Qdrant collection
- PI.6 (Expected): Verify vector search functionality and optimize retrieval quality
- PI.7 (Advanced): Implement quantization and tune HNSW parameters for large collections

### Objective 3: Frontend & API Integration

- PI.1 (Basic): Set up the React + Vite chat UI with message list, input box, and loading states
- PI.2 (Basic): Implement FastAPI `/chat` and `/health` endpoints with proper request validation
- PI.3 (Expected): Connect frontend to backend with streaming support and error handling
- PI.4 (Expected): Add UX improvements—example queries, markdown formatting, conversation history
- PI.5 (Advanced): Write Swagger/OpenAPI docs and an integration guide for the rest of the team
