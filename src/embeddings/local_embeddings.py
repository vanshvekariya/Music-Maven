"""Local embedding model using sentence-transformers"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger

from .base import BaseEmbedding
from src.config import get_settings


class LocalEmbedding(BaseEmbedding):
    """Local embedding model using sentence-transformers (no API required)"""
    
    def __init__(self, model_name: str = None):
        """
        Initialize local embedding model
        
        Args:
            model_name: Name of the sentence-transformers model
        """
        settings = get_settings()
        self.model_name = model_name or settings.local_embedding_model
        
        logger.info(f"Loading local embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        logger.info(f"Model loaded successfully. Dimension: {self.get_dimension()}")
    
    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Encode multiple texts into embeddings
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            Array of embeddings
        """
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode a single text into embedding
        
        Args:
            text: Text string
            
        Returns:
            Embedding vector
        """
        embedding = self.model.encode([text], convert_to_numpy=True)[0]
        return embedding
    
    def get_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors
        
        Returns:
            Embedding dimension
        """
        return self.model.get_sentence_embedding_dimension()
