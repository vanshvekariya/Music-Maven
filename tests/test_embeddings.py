"""Tests for embedding models"""

import pytest
import numpy as np
from src.embeddings import LocalEmbedding


class TestLocalEmbedding:
    """Test local embedding model"""
    
    @pytest.fixture
    def embedding_model(self):
        """Create embedding model fixture"""
        return LocalEmbedding(model_name="all-MiniLM-L6-v2")
    
    def test_encode_single(self, embedding_model):
        """Test encoding a single text"""
        text = "This is a test video about cats"
        embedding = embedding_model.encode_single(text)
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding.shape) == 1
        assert embedding.shape[0] == embedding_model.get_dimension()
    
    def test_encode_multiple(self, embedding_model):
        """Test encoding multiple texts"""
        texts = [
            "Gaming video tutorial",
            "Cooking recipe for pasta",
            "Music concert performance"
        ]
        embeddings = embedding_model.encode(texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] == embedding_model.get_dimension()
    
    def test_similarity(self, embedding_model):
        """Test that similar texts have similar embeddings"""
        text1 = "cat video funny"
        text2 = "funny cat videos"
        text3 = "cooking recipe pasta"
        
        emb1 = embedding_model.encode_single(text1)
        emb2 = embedding_model.encode_single(text2)
        emb3 = embedding_model.encode_single(text3)
        
        # Cosine similarity
        sim_12 = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        sim_13 = np.dot(emb1, emb3) / (np.linalg.norm(emb1) * np.linalg.norm(emb3))
        
        # Similar texts should have higher similarity
        assert sim_12 > sim_13
