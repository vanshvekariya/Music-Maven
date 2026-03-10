"""Pydantic models for API requests and responses"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    query: str = Field(..., description="Natural language query", min_length=1)
    max_results: Optional[int] = Field(10, description="Maximum number of results", ge=1, le=100)


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
