"""Pydantic models for API requests and responses"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    query: str = Field(..., description="Natural language query", min_length=1)
    max_results: Optional[int] = Field(10, description="Maximum vector hits (orchestrator passes to vector agent)", ge=1, le=100)
    use_kg: Optional[bool] = Field(
        True,
        description="Whether to allow Knowledge Graph direct routing (true by default)",
    )
    lang_filter: Optional[str] = Field(
        None,
        description="Optional ISO-style language code (e.g. en, pt, fr) for Qdrant payload filter on vector search",
    )
    session_id: Optional[str] = Field(
        None,
        description="Optional client session id for multi-turn memory (in-memory on server; omit to disable)",
        max_length=128,
    )
    memory_turns: Optional[int] = Field(
        5,
        description="Set to 0 to disable session memory. Any value > 0 enables rolling summarized memory (budget uses memory_max_chars).",
        ge=0,
        le=30,
    )
    memory_max_chars: Optional[int] = Field(
        2000,
        description="Max characters for stored rolling summary and injected context (~500 tokens); summarizer compresses each turn to stay within this.",
        ge=0,
        le=8000,
    )


class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    query: str
    answer: str
    success: bool
    metadata: Optional[Dict[str, Any]] = None
    results: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None


class SystemInfoResponse(BaseModel):
    """Response model for system info endpoint"""
    orchestrator: str
    agents: Dict[str, Dict[str, Any]]
    configuration: Dict[str, Any]


class HealthResponse(BaseModel):
    """Response model for health check endpoint"""
    status: str
    version: str
    agents_available: List[str]
    database_connected: bool


class ExampleQuery(BaseModel):
    """Example query model"""
    category: str
    query: str
    description: str
