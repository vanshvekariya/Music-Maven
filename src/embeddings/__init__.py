"""Embeddings module for generating vector representations"""

from .base import BaseEmbedding
from .local_embeddings import LocalEmbedding
from .openai_embeddings import OpenAIEmbedding
from .factory import get_embedding_model

__all__ = ["BaseEmbedding", "LocalEmbedding", "OpenAIEmbedding", "get_embedding_model"]
