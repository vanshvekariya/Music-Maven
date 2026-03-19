"""FastAPI application for Music Maven"""

import time
from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from .models import (
    QueryRequest,
    QueryResponse,
    SystemInfoResponse,
    HealthResponse,
    ExampleQuery
)
from ..main import MusicMavenApp
from ..config.settings import get_settings


# Global app instance
app_instance = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    global app_instance
    
    logger.info("Starting Music Maven API...")
    try:
        app_instance = MusicMavenApp()
        logger.info("Application initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    
    yield
    
    logger.info("Shutting down Music Maven API...")


# Create FastAPI app
app = FastAPI(
    title="Music Maven API",
    description="Multi-Agent AI system for music information retrieval (Music4All dataset)",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000", "http://127.0.0.1:5173", "http://127.0.0.1:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Music Maven API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    try:
        info = app_instance.get_system_info()
        agents = list(info.get('agents', {}).keys())
        
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            agents_available=agents,
            database_connected=True
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            version="1.0.0",
            agents_available=[],
            database_connected=False
        )


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def process_query(request: QueryRequest):
    """
    Process a natural language query about the Music4All dataset.

    The system will automatically route your query to the appropriate agent(s):
    - SQL Agent: For analytical queries (stats, counts, audio features, aggregations)
    - Vector Agent: For semantic/lyric search (Phase 2 — not yet active)
    - Hybrid: For queries requiring both capabilities (Phase 2)
    """
    if not app_instance:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Application not initialized"
        )
    
    try:
        start_time = time.time()
        
        logger.info(f"Processing query: {request.query}")
        response = app_instance.query(request.query)
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            query=request.query,
            answer=response.get('answer') or 'No answer generated',
            success=response.get('success', True),
            metadata=response.get('metadata'),
            results=response.get('results'),
            error=response.get('error'),
            processing_time=processing_time
        )
    
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )


@app.get("/system/info", response_model=SystemInfoResponse, tags=["System"])
async def get_system_info():
    """Get information about the system and available agents"""
    if not app_instance:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Application not initialized"
        )
    
    try:
        info = app_instance.get_system_info()
        
        return SystemInfoResponse(
            orchestrator=info.get('orchestrator', 'Unknown'),
            agents=info.get('agents', {}),
            configuration={
                'llm_model': settings.llm_model,
                'sql_database': settings.sql_db_path,
                'vector_db': f"Qdrant ({settings.qdrant_host}:{settings.qdrant_port})",
                'embedding_model (tags)': settings.local_embedding_model,
                'embedding_model (lyrics)': settings.lyric_embedding_model
            }
        )
    
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system info: {str(e)}"
        )


@app.get("/examples", response_model=List[ExampleQuery], tags=["Examples"])
async def get_example_queries():
    """Get example queries to help users understand the system capabilities"""
    return [
        ExampleQuery(
            category="KG (instant)",
            query="Who are the top 10 most popular artists?",
            description="Instant answer from Knowledge Graph (0 LLM calls)"
        ),
        ExampleQuery(
            category="KG (instant)",
            query="How many songs are in English?",
            description="Language count from Knowledge Graph"
        ),
        ExampleQuery(
            category="KG (instant)",
            query="Compare Queen and The Beatles",
            description="Side-by-side artist comparison from Knowledge Graph"
        ),
        ExampleQuery(
            category="KG (instant)",
            query="What are the most common genres?",
            description="Genre distribution from Knowledge Graph"
        ),
        ExampleQuery(
            category="KG (instant)",
            query="Songs by Queen",
            description="Artist song list from Knowledge Graph"
        ),
        ExampleQuery(
            category="SQL",
            query="Find high-energy danceable songs with popularity above 70",
            description="Complex multi-filter query via SQL agent"
        ),
        ExampleQuery(
            category="Vector",
            query="Upbeat Brazilian pop songs",
            description="Semantic search over artist, genre and tags"
        ),
        ExampleQuery(
            category="Vector",
            query="Find songs with lyrics about heartbreak and longing",
            description="Semantic search over lyric text"
        ),
        ExampleQuery(
            category="Vector",
            query="Chill lo-fi hip hop",
            description="Style and genre similarity search"
        ),
        ExampleQuery(
            category="Hybrid",
            query="Popular sad rock songs in English",
            description="KG structured filter + semantic mood search"
        ),
    ]


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An unexpected error occurred",
            "error": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
