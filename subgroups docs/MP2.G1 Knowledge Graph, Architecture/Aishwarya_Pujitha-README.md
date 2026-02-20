## Aishwarya Pujitha

I'm focusing on the data pipeline, knowledge graph, and frontend/API. Here are my objectives and how I plan to measure progress.

### Objective 1: Data Pipeline & Database Design

- PI.1 (Basic): Load Music4All and Kaggle lyrics, parse into structured format, and validate data integrity
- PI.2 (Basic): Design and document the PostgreSQL schema (songs, artists, albums, genres, tags, lyrics)
- PI.3 (Expected): Implement batch ingestion with cleaning, normalization, and handling for multi-value fields like genres and tags
- PI.4 (Expected): Add indexes and constraints for performance; verify referential integrity
- PI.5 (Advanced): Build a data validation layer with automated checks for schema consistency

### Objective 2: Knowledge Graph Implementation

- PI.1 (Basic): Define Neo4j node types (Song, Artist, Genre, Album, Tag) and relationship types (PERFORMED, HAS_GENRE, etc.)
- PI.2 (Basic): Ingest entities and relationships from Music4All into Neo4j with basic properties
- PI.3 (Expected): Build the graph query agent (Cypher interface) and wire it into the RAG pipeline
- PI.4 (Expected): Handle entity resolution and deduplication for artist/genre/tag names
- PI.5 (Advanced): Infer additional relationships (e.g., artist similarity from shared genres/tags)

### Objective 3: Frontend & API Integration

- PI.1 (Basic): Set up the React + Vite chat UI with message list, input box, and loading states
- PI.2 (Basic): Implement FastAPI `/chat` and `/health` endpoints with proper request validation
- PI.3 (Expected): Connect frontend to backend with streaming support and error handling
- PI.4 (Expected): Add UX improvements—example queries, markdown formatting, conversation history
- PI.5 (Advanced): Write Swagger/OpenAPI docs and an integration guide for the rest of the team
