"""Factory for creating embedding models"""

from loguru import logger

from .base import BaseEmbedding
from .local_embeddings import LocalEmbedding
from .openai_embeddings import OpenAIEmbedding
from src.config import get_settings


def get_embedding_model() -> BaseEmbedding:
    """
    Get the appropriate embedding model based on settings
    
    Returns:
        Embedding model instance
    """
    settings = get_settings()
    
    if settings.use_local_embeddings:
        logger.info("Using local embedding model (sentence-transformers)")
        return LocalEmbedding(model_name=settings.local_embedding_model)
    else:
        logger.info("Using OpenAI embedding model")
        return OpenAIEmbedding(model_name=settings.openai_embedding_model)
