"""Knowledge Graph module for Music Maven fast-path query resolution."""

from .kg_builder import KnowledgeGraphBuilder
from .kg_query_engine import KGQueryEngine

__all__ = ["KnowledgeGraphBuilder", "KGQueryEngine"]
