"""SQL Agent for structured database queries using LangChain"""

import os
from typing import Dict, Any, Optional
from pathlib import Path

from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from loguru import logger

from .base_agent import BaseAgent
from ..config.settings import get_settings


# LangChain passes every sql_db_query result string back into the LLM; a broad SELECT
# can return hundreds of thousands of rows and exceed the model context (see OpenRouter
# "maximum context length" errors). Cap each tool result so the agent can recover.
class BoundedSQLDatabase(SQLDatabase):
    """SQLDatabase that truncates oversized query result strings."""

    MAX_QUERY_RESULT_CHARS = 14_000

    def run_no_throw(
        self,
        command: str,
        fetch="all",
        include_columns: bool = False,
        *,
        parameters=None,
        execution_options=None,
    ):
        out = super().run_no_throw(
            command,
            fetch=fetch,
            include_columns=include_columns,
            parameters=parameters,
            execution_options=execution_options,
        )
        if isinstance(out, str) and len(out) > self.MAX_QUERY_RESULT_CHARS:
            return (
                out[: self.MAX_QUERY_RESULT_CHARS]
                + "\n\n[TRUNCATED: result too large for the model. Rewrite SQL with a "
                "tighter WHERE and/or LIMIT (e.g. LIMIT 100).]"
            )
        return out


class SQLAgent(BaseAgent):
    """
    SQL Agent that converts natural language queries to SQL and executes them.
    Uses LangChain's SQL agent toolkit for intelligent query generation.
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize SQL Agent.
        
        Args:
            db_path: Path to SQLite database
            api_key: OpenAI/OpenRouter API key
            model: LLM model to use
        """
        super().__init__(name="SQLAgent")
        
        settings = get_settings()
        
        # Database configuration
        self.db_path = db_path or settings.sql_db_path
        self.table_name = settings.sql_table_name
        
        # LLM configuration
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.llm_model
        self.base_url = settings.openai_base_url
        
        if not self.api_key:
            raise ValueError(
                "API key is required. Set OPENAI_API_KEY in environment or .env file"
            )
        
        # Initialize components
        self._initialize_database()
        self._initialize_llm()
        self._initialize_agent()
        
        logger.info(f"SQL Agent initialized with database: {self.db_path}")
    
    def _initialize_database(self) -> None:
        """Initialize database connection"""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(
                f"Database file not found: {self.db_path}. "
                f"Please run the data processing pipeline first."
            )
        
        self.db = BoundedSQLDatabase.from_uri(f"sqlite:///{self.db_path}")
        logger.info(f"Connected to database: {self.db_path}")
    
    def _initialize_llm(self) -> None:
        """Initialize LLM for SQL generation"""
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=0,
            api_key=self.api_key,
            base_url=self.base_url
        )
        logger.info(f"Initialized LLM: {self.model}")
    
    def _initialize_agent(self) -> None:
        """Initialize LangChain SQL agent"""
        
        # Create a detailed prefix with exact column names
        prefix = f"""You are an expert music information retrieval analyst with access to a SQLite database containing the Music4All dataset.

Database: {self.db_path}
Table: {self.table_name}

IMPORTANT - EXACT COLUMN NAMES (use these exactly as shown):
┌──────────────┬──────────────────────────────────────────────────────────┐
│ Column Name  │ Description                                              │
├──────────────┼──────────────────────────────────────────────────────────┤
│ song_id      │ Unique song identifier (TEXT)                            │
│ artist       │ Artist name (TEXT)                                       │
│ song_name    │ Song title (TEXT)                                        │
│ album        │ Album name (TEXT)                                        │
│ lang         │ Language of lyrics, e.g. "en", "es" (TEXT)              │
│ spotify_id   │ Spotify track identifier (TEXT)                          │
│ popularity   │ Spotify popularity score 0-100 (REAL)                    │
│ release      │ Release year (INTEGER)                                   │
│ danceability │ Danceability score 0.0-1.0 (REAL)                       │
│ energy       │ Energy score 0.0-1.0 (REAL)                             │
│ key          │ Musical key 0-11 (REAL)                                  │
│ mode         │ 1=major, 0=minor (REAL)                                  │
│ valence      │ Musical positiveness 0.0-1.0 (REAL)                     │
│ tempo        │ Tempo in BPM (REAL)                                      │
│ duration_ms  │ Track duration in milliseconds (INTEGER)                 │
│ tags         │ Comma-separated user tags (TEXT)                         │
│ genres       │ Comma-separated genres (TEXT)                            │
│ has_lyrics   │ 1 if lyrics file exists, 0 otherwise (INTEGER)          │
└──────────────┴──────────────────────────────────────────────────────────┘

⚠️ CRITICAL COLUMN NAME MAPPINGS:
- For SPOTIFY ID: Use "spotify_id" NOT "spotifyid" or "spotify_track_id"
- For SONG TITLE: Use "song_name" NOT "title" or "name"
- For LANGUAGE: Use "lang" NOT "language"
- For MUSICAL KEY MODE: Use "mode" (1=major, 0=minor) NOT "key_mode"
- For RELEASE YEAR: Use "release" NOT "year" or "release_date"
- These columns do NOT exist: loudness, speechiness, acousticness, instrumentalness, liveness, time_signature, audio_path

QUERY SAFETY (non-negotiable):
- Broad SELECTs MUST include LIMIT (start with LIMIT 100; use LIMIT 20–50 for "top N" lists).
- Never scan the whole table without LIMIT unless the user explicitly asks for a full export.
- For unique song titles: use DISTINCT song_name or GROUP BY song_name (plus artist filter if needed), always with LIMIT.
- If tool output was truncated, rewrite SQL to be narrower (WHERE + LIMIT), do not repeat the huge query.

COMMON QUERY PATTERNS:
1. Top artists by popularity: SELECT artist, AVG(popularity) AS avg_pop FROM songs GROUP BY artist ORDER BY avg_pop DESC LIMIT N;
2. Songs by genre: SELECT song_name, artist, genres FROM songs WHERE genres LIKE '%rock%' ORDER BY popularity DESC LIMIT N;
3. High-energy danceable songs: SELECT song_name, artist, energy, danceability FROM songs WHERE energy > 0.8 AND danceability > 0.7 ORDER BY popularity DESC LIMIT N;
4. Songs by language: SELECT lang, COUNT(*) as count FROM songs GROUP BY lang ORDER BY count DESC;
5. Songs with lyrics in a tempo range: SELECT song_name, artist, tempo FROM songs WHERE has_lyrics = 1 AND tempo BETWEEN 120 AND 140;

Your goal is to:
1. Understand the user's question
2. Generate SQL queries using the EXACT column names above
3. Execute them against the database
4. Return results in clear, natural language with PROPER MARKDOWN FORMATTING

⚠️ CRITICAL: NEVER SHOW SQL CODE TO THE USER
- DO NOT include SQL queries in your response
- DO NOT show database queries or technical details
- DO NOT add tips about running SQL queries
- ONLY provide the answer in natural, conversational language

RESPONSE FORMATTING REQUIREMENTS:
- Use markdown bullet points for lists
- Use **bold** for song titles and artist names
- Format audio feature scores as percentages (e.g., energy: 85%, danceability: 72%)
- Each bullet point must contain ONLY values that were returned by the SQL query
- DO NOT add editorial descriptions, song summaries, or genre commentary

⚠️ STRICT OUTPUT RULE: Every piece of information in your response must come directly
from the SQL query result. If a column was not in the SELECT, do not mention it.

CORRECT example format (only SQL result fields, nothing else):
Here are the most danceable songs:

- **Run the World (Girls)** by **Beyoncé** – popularity: 85, energy: 92%, danceability: 88%, tempo: 128 BPM
- **Reggaetón Lento** by **CNCO** – popularity: 78, energy: 79%, danceability: 91%, tempo: 95 BPM

WRONG — never do this:
- **Run the World (Girls)** by **Beyoncé** (≈100 M views) – An empowering anthem with infectious energy.
  ↑ WRONG: "views" is not in the database. "An empowering anthem" is invented commentary.

WHAT NOT TO DO:
❌ DO NOT mention: views, streams, play counts, YouTube metrics, chart positions — none of this exists in the database
❌ DO NOT add: song descriptions, editorial commentary, or genre summaries
❌ DO NOT invent: any value not directly returned by the SQL query
❌ DO NOT show: SELECT statements or any SQL code
✅ DO: List only the exact field values the query returned, formatted cleanly

If you encounter a "no such column" error, check the column names table above and use the exact names shown.
"""
        
        self.agent_executor = create_sql_agent(
            llm=self.llm,
            db=self.db,
            agent_type="openai-tools",
            verbose=True,
            handle_parsing_errors=True,
            prefix=prefix,
            top_k=50,
            max_iterations=12,
            agent_executor_kwargs={
                "handle_parsing_errors": True
            }
        )
        logger.info("SQL Agent executor initialized")
    
    def process_query(
        self,
        query: str,
        conversation_context: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Process a natural language query using SQL.

        Args:
            query: Natural language query (used for routing elsewhere; keep as the current turn)
            conversation_context: Optional prior transcript for follow-up questions (SQL agent only)
            **kwargs: Additional parameters

        Returns:
            Formatted response with results
        """
        if not self.validate_query(query):
            return self.format_response(
                success=False,
                error="Invalid query: Query cannot be empty"
            )

        # Hard cap: rolling summary should be small, but never trust unbounded input.
        max_ctx = 3_500
        agent_input = query
        if conversation_context and conversation_context.strip():
            ctx = conversation_context.strip()
            if len(ctx) > max_ctx:
                ctx = ctx[:max_ctx] + "\n[...truncated...]"
            agent_input = (
                "Prior conversation (for context only; answer using the database):\n"
                f"{ctx}\n\n"
                f"Current user question:\n{query}"
            )

        try:
            logger.info(f"Processing SQL query: {query}")

            # Invoke the agent
            response = self.agent_executor.invoke({"input": agent_input})
            
            # Extract output
            output = response.get("output", "I'm sorry, I couldn't find an answer.")
            
            logger.info("SQL query processed successfully")
            
            return self.format_response(
                success=True,
                data={
                    'answer': output,
                    'query_type': 'sql',
                    'source': 'structured_database'
                },
                metadata={
                    'database': self.db_path,
                    'table': self.table_name,
                    'model': self.model
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing SQL query: {e}")
            return self.format_response(
                success=False,
                error=f"SQL Agent error: {str(e)}"
            )
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get SQL Agent capabilities.
        
        Returns:
            Dictionary describing capabilities
        """
        return {
            'name': self.name,
            'type': 'sql',
            'description': 'Handles structured queries requiring aggregations, filtering, and statistical analysis',
            'capabilities': [
                'Aggregation queries (COUNT, SUM, AVG, MAX, MIN)',
                'Filtering by artist, language, genre, popularity, tempo, energy',
                'Sorting and ranking (TOP N queries)',
                'Statistical analysis of audio features',
                'Complex joins and grouping',
                'Aggregations across genres or languages'
            ],
            'best_for': [
                'Which artist has the most songs?',
                'Top 10 songs by popularity',
                'Average tempo of hip-hop songs',
                'Songs with energy above 0.8',
                'Comparison between genres',
                'Statistical summaries of the Music4All dataset'
            ],
            'database': self.db_path,
            'table': self.table_name
        }
    
    def get_schema_info(self) -> str:
        """
        Get database schema information.
        
        Returns:
            Schema description string
        """
        try:
            return self.db.get_table_info()
        except Exception as e:
            logger.error(f"Error getting schema info: {e}")
            return "Schema information unavailable"
    
    def execute_raw_sql(self, sql_query: str) -> Dict[str, Any]:
        """
        Execute a raw SQL query (for advanced users).
        
        Args:
            sql_query: Raw SQL query string
            
        Returns:
            Query results
        """
        try:
            logger.info(f"Executing raw SQL: {sql_query}")
            result = self.db.run(sql_query)
            
            return self.format_response(
                success=True,
                data={'result': result},
                metadata={'query': sql_query}
            )
            
        except Exception as e:
            logger.error(f"Error executing raw SQL: {e}")
            return self.format_response(
                success=False,
                error=f"SQL execution error: {str(e)}"
            )
