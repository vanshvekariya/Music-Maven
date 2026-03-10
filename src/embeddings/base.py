"""Base interface for embedding models"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np


class BaseEmbedding(ABC):
    """Abstract base class for embedding models"""
    
    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts into embeddings
        
        Args:
            texts: List of text strings
            
        Returns:
            Array of embeddings with shape (len(texts), embedding_dim)
        """
        pass
    
    @abstractmethod
    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode a single text into embedding
        
        Args:
            text: Text string
            
        Returns:
            Embedding vector
        """
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors
        
        Returns:
            Embedding dimension
        """
        pass
