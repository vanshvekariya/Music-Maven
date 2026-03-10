"""OpenAI embedding model"""

from typing import List
import numpy as np
from openai import OpenAI
from loguru import logger

from .base import BaseEmbedding
from src.config import get_settings


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI embedding model (requires API key)"""
    
    # Dimension mapping for OpenAI models
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    
    def __init__(self, model_name: str = None, api_key: str = None):
        """
        Initialize OpenAI embedding model
        
        Args:
            model_name: Name of the OpenAI embedding model
            api_key: OpenAI API key
        """
        settings = get_settings()
        self.model_name = model_name or settings.openai_embedding_model
        
        api_key = api_key or settings.openai_api_key
        if not api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY in .env file")
        
        self.client = OpenAI(api_key=api_key)
        logger.info(f"Initialized OpenAI embedding model: {self.model_name}")
    
    def encode(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """
        Encode multiple texts into embeddings
        
        Args:
            texts: List of text strings
            batch_size: Batch size for API calls
            
        Returns:
            Array of embeddings
        """
        if not texts:
            return np.array([])
        
        all_embeddings = []
        
        # Process in batches to avoid rate limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model_name
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                logger.debug(f"Encoded batch {i // batch_size + 1}: {len(batch)} texts")
                
            except Exception as e:
                logger.error(f"Error encoding batch: {e}")
                raise
        
        return np.array(all_embeddings)
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode a single text into embedding
        
        Args:
            text: Text string
            
        Returns:
            Embedding vector
        """
        try:
            response = self.client.embeddings.create(
                input=[text],
                model=self.model_name
            )
            embedding = response.data[0].embedding
            return np.array(embedding)
        
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            raise
    
    def get_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors
        
        Returns:
            Embedding dimension
        """
        return self.MODEL_DIMENSIONS.get(self.model_name, 1536)
