"""Query Router for classifying and routing queries to appropriate agents"""

from typing import Dict, Any, Literal, Optional
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import PydanticOutputParser

from pydantic import BaseModel, Field
from loguru import logger

from ..config.settings import get_settings


class QueryType(str, Enum):
    """Enum for query types"""
    SQL = "sql"
    VECTOR = "vector"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


class QueryClassification(BaseModel):
    """Structured output for query classification"""
    query_type: QueryType = Field(
        description="Type of query: 'sql' for structured/analytical, 'vector' for semantic/content-based, 'hybrid' for both"
    )
    confidence: float = Field(
        description="Confidence score between 0 and 1",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        description="Brief explanation of why this classification was chosen"
    )
    suggested_agent: str = Field(
        description="Which agent(s) should handle this query"
    )


class QueryRouter:
    """
    Intelligent query router that classifies queries and routes them to appropriate agents.
    Uses LLM-based classification with structured output.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize Query Router.
        
        Args:
            api_key: OpenAI/OpenRouter API key
            model: LLM model to use
        """
        settings = get_settings()
        
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.llm_model
        self.base_url = settings.openai_base_url
        
        if not self.api_key:
            raise ValueError(
                "API key is required for Query Router. "
                "Set OPENAI_API_KEY in environment or .env file"
            )
        
        self._initialize_llm()
        self._initialize_chain()
        
        logger.info("Query Router initialized")
    
    def _initialize_llm(self) -> None:
        """Initialize LLM for classification"""
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=0,
            api_key=self.api_key,
            base_url=self.base_url
        )
        logger.info(f"Router LLM initialized: {self.model}")
    
    def _initialize_chain(self) -> None:
        """Initialize classification chain"""
        # Create output parser
        self.parser = PydanticOutputParser(pydantic_object=QueryClassification)
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intelligent query router for Music Maven, a music information retrieval system backed by the Music4All dataset.
You must classify user queries into one of these types:

1. **SQL** - For structured, analytical queries requiring:
   - Aggregations (count, average, max, min)
   - Filtering by specific attributes (artist, language, popularity, tempo, energy, danceability, genre)
   - Sorting and ranking (top N, most popular, fastest BPM)
   - Statistical analysis of audio features
   - Comparisons between artists, genres, or languages

   Examples:
   - "Which artist has the most songs in the dataset?"
   - "Top 10 songs by popularity"
   - "Average tempo of hip-hop songs"
   - "Songs with energy above 0.8 and danceability above 0.7"
   - "How many songs are in Portuguese?"

2. **VECTOR** - For semantic, content-based queries requiring:
   - Mood or theme similarity (sad, uplifting, angry, nostalgic)
   - Lyric content search ("songs about heartbreak", "songs about summer")
   - Style or vibe similarity ("songs that sound like X", "chill late-night music")
   - Natural language / exploratory searches

   Examples:
   - "Find songs with melancholic lyrics"
   - "Songs that feel like a road trip"
   - "Songs similar to Bohemian Rhapsody"
   - "Upbeat pop songs about love"
   - "Reggaeton with heavy bass"

3. **HYBRID** - For queries requiring BOTH SQL and VECTOR:
   - Queries combining structured filters AND semantic/mood criteria
   - Ranking/statistics (SQL) + lyric or style search (VECTOR)
   
   IMPORTANT: Look for queries that contain BOTH:
   - Numeric/filter keywords (popularity, tempo, energy, top, count) AND
   - Mood/lyric/style keywords (feel, about, similar, vibe, theme, lyrics)

   Examples:
   - "High-energy songs about heartbreak with popularity above 70"
   - "Top 10 most danceable songs and find songs similar to ABBA"
   - "Portuguese songs with sad lyrics"
   - "Fast tempo rock songs that feel melancholic"

4. **UNKNOWN** - For unclear or out-of-scope queries

CRITICAL RULES:
- If query contains TWO distinct parts (connected by AND/ALSO/PLUS), classify as HYBRID
- If query asks for both numeric filters AND mood/lyric/style content, classify as HYBRID
- Pure numeric/attribute queries → SQL. Pure mood/lyric/style queries → VECTOR
- When in doubt between VECTOR and HYBRID, check if there is a numeric filter component

Analyze the query carefully and provide a classification with reasoning.

{format_instructions}
"""),
            ("user", "Query: {query}")
        ])
        
        # Create chain
        self.chain = self.prompt | self.llm | self.parser
        
        logger.info("Classification chain initialized")
    
    def classify_query(self, query: str) -> QueryClassification:
        """
        Classify a query into appropriate type.
        
        Args:
            query: User query string
            
        Returns:
            QueryClassification object
        """
        try:
            logger.info(f"Classifying query: {query}")
            
            # Invoke chain
            result = self.chain.invoke({
                "query": query,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            logger.info(
                f"Query classified as: {result.query_type} "
                f"(confidence: {result.confidence:.2f})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error classifying query: {e}")
            # Return default classification
            return QueryClassification(
                query_type=QueryType.UNKNOWN,
                confidence=0.0,
                reasoning=f"Classification failed: {str(e)}",
                suggested_agent="none"
            )
    
    def route_query(self, query: str) -> Dict[str, Any]:
        """
        Classify and route a query.
        
        Args:
            query: User query string
            
        Returns:
            Routing information dictionary
        """
        classification = self.classify_query(query)
        
        # Determine which agents to use
        agents_to_use = []
        
        if classification.query_type == QueryType.SQL:
            agents_to_use = ["sql"]
        elif classification.query_type == QueryType.VECTOR:
            agents_to_use = ["vector"]
        elif classification.query_type == QueryType.HYBRID:
            agents_to_use = ["sql", "vector"]
        else:
            agents_to_use = []
        
        routing_info = {
            'query': query,
            'classification': {
                'type': classification.query_type.value,
                'confidence': classification.confidence,
                'reasoning': classification.reasoning,
                'suggested_agent': classification.suggested_agent
            },
            'agents': agents_to_use,
            'execution_strategy': self._determine_execution_strategy(
                classification.query_type
            )
        }
        
        logger.info(f"Query routed to agents: {agents_to_use}")
        
        return routing_info
    
    def _determine_execution_strategy(self, query_type: QueryType) -> str:
        """
        Determine execution strategy based on query type.
        
        Args:
            query_type: Classified query type
            
        Returns:
            Execution strategy string
        """
        strategies = {
            QueryType.SQL: "single_agent",
            QueryType.VECTOR: "single_agent",
            QueryType.HYBRID: "multi_agent_parallel",
            QueryType.UNKNOWN: "fallback"
        }
        
        return strategies.get(query_type, "fallback")
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """
        Get routing statistics (placeholder for future implementation).
        
        Returns:
            Statistics dictionary
        """
        return {
            'total_queries': 0,
            'sql_queries': 0,
            'vector_queries': 0,
            'hybrid_queries': 0,
            'unknown_queries': 0
        }


class SimpleQueryRouter:
    """
    Simplified rule-based query router (fallback when LLM is unavailable).
    Uses keyword matching for classification.
    """
    
    SQL_KEYWORDS = [
        'how many', 'count', 'total', 'average', 'sum', 'top', 'bottom',
        'most', 'least', 'highest', 'lowest', 'compare', 'comparison',
        'statistics', 'stat', 'number of', 'percentage', 'ratio'
    ]
    
    VECTOR_KEYWORDS = [
        'find', 'search', 'similar', 'like', 'about', 'related to',
        'videos on', 'content about', 'show me', 'recommend', 'suggestion'
    ]
    
    def __init__(self):
        """Initialize simple router"""
        logger.info("Simple Query Router initialized (rule-based)")
    
    def classify_query(self, query: str) -> QueryClassification:
        """
        Classify query using simple keyword matching.
        
        Args:
            query: User query
            
        Returns:
            QueryClassification
        """
        query_lower = query.lower()
        
        sql_score = sum(1 for kw in self.SQL_KEYWORDS if kw in query_lower)
        vector_score = sum(1 for kw in self.VECTOR_KEYWORDS if kw in query_lower)
        
        # Check for hybrid indicators
        hybrid_connectors = ['and also', 'also', 'and suggest', 'and find', 'plus', 'and show']
        has_hybrid_connector = any(conn in query_lower for conn in hybrid_connectors)
        
        # If both SQL and VECTOR keywords present, or has hybrid connectors
        if (sql_score > 0 and vector_score > 0) or (has_hybrid_connector and (sql_score > 0 or vector_score > 0)):
            query_type = QueryType.HYBRID
            confidence = 0.7
            reasoning = "Query contains both analytical and semantic keywords, or has compound structure"
        elif sql_score > vector_score:
            query_type = QueryType.SQL
            confidence = min(0.7, 0.5 + sql_score * 0.1)
            reasoning = "Query contains analytical keywords"
        elif vector_score > sql_score:
            query_type = QueryType.VECTOR
            confidence = min(0.7, 0.5 + vector_score * 0.1)
            reasoning = "Query contains semantic search keywords"
        else:
            query_type = QueryType.VECTOR  # Default to vector search
            confidence = 0.5
            reasoning = "No clear indicators, defaulting to semantic search"
        
        return QueryClassification(
            query_type=query_type,
            confidence=confidence,
            reasoning=reasoning,
            suggested_agent=query_type.value
        )
    
    def route_query(self, query: str) -> Dict[str, Any]:
        """
        Route query using simple classification.
        
        Args:
            query: User query
            
        Returns:
            Routing information
        """
        classification = self.classify_query(query)
        
        agents_to_use = []
        if classification.query_type == QueryType.SQL:
            agents_to_use = ["sql"]
        elif classification.query_type == QueryType.VECTOR:
            agents_to_use = ["vector"]
        elif classification.query_type == QueryType.HYBRID:
            agents_to_use = ["sql", "vector"]
        
        return {
            'query': query,
            'classification': {
                'type': classification.query_type.value,
                'confidence': classification.confidence,
                'reasoning': classification.reasoning,
                'suggested_agent': classification.suggested_agent
            },
            'agents': agents_to_use,
            'execution_strategy': 'single_agent' if len(agents_to_use) == 1 else 'multi_agent_parallel'
        }
