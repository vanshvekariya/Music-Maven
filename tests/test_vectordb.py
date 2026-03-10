"""Tests for vector database operations"""

import pytest
import numpy as np
from src.vectordb import QdrantManager


class TestQdrantManager:
    """Test Qdrant manager"""
    
    @pytest.fixture
    def qdrant_manager(self):
        """Create Qdrant manager fixture"""
        return QdrantManager()
    
    def test_connection(self, qdrant_manager):
        """Test connection to Qdrant"""
        assert qdrant_manager.client is not None
    
    def test_create_collection(self, qdrant_manager):
        """Test creating a collection"""
        test_collection = "test_collection"
        
        # Clean up if exists
        if qdrant_manager.collection_exists(test_collection):
            qdrant_manager.delete_collection(test_collection)
        
        # Create collection
        result = qdrant_manager.create_collection(
            collection_name=test_collection,
            vector_size=384,
            recreate=True
        )
        
        assert result is True
        assert qdrant_manager.collection_exists(test_collection)
        
        # Clean up
        qdrant_manager.delete_collection(test_collection)
    
    def test_collection_info(self, qdrant_manager):
        """Test getting collection info"""
        test_collection = "test_collection"
        
        # Create test collection
        qdrant_manager.create_collection(
            collection_name=test_collection,
            vector_size=384,
            recreate=True
        )
        
        # Get info
        info = qdrant_manager.get_collection_info(test_collection)
        
        assert 'name' in info
        assert info['name'] == test_collection
        assert 'points_count' in info
        
        # Clean up
        qdrant_manager.delete_collection(test_collection)
