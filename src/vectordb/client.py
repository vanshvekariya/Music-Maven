"""Qdrant client wrapper and connection management"""

from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from loguru import logger

from src.config import get_settings


class QdrantManager:
    """Manage Qdrant client and collection operations"""
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None,
        vector_size: Optional[int] = None,
    ):
        """
        Initialize Qdrant client

        Args:
            host: Qdrant host (default from settings)
            port: Qdrant port (default from settings)
            collection_name: Override the collection name from settings
            vector_size: Override the vector size from settings
        """
        settings = get_settings()
        self.host = host or settings.qdrant_host
        self.port = port or settings.qdrant_port
        self.collection_name = collection_name or settings.qdrant_collection_name
        self.vector_size = vector_size or settings.vector_size
        
        logger.info(f"Connecting to Qdrant at {self.host}:{self.port}")
        self.client = QdrantClient(host=self.host, port=self.port)
        
        # Test connection
        try:
            collections = self.client.get_collections()
            logger.info(f"Successfully connected to Qdrant. Collections: {len(collections.collections)}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    def collection_exists(self, collection_name: Optional[str] = None) -> bool:
        """
        Check if a collection exists
        
        Args:
            collection_name: Name of collection (default: from settings)
            
        Returns:
            True if collection exists
        """
        collection_name = collection_name or self.collection_name
        
        try:
            collections = self.client.get_collections()
            return any(col.name == collection_name for col in collections.collections)
        except Exception as e:
            logger.error(f"Error checking collection existence: {e}")
            return False
    
    def create_collection(
        self,
        collection_name: Optional[str] = None,
        vector_size: Optional[int] = None,
        distance: Distance = Distance.COSINE,
        recreate: bool = False
    ) -> bool:
        """
        Create a new collection
        
        Args:
            collection_name: Name of collection (default: from settings)
            vector_size: Size of vectors (default: from settings)
            distance: Distance metric (COSINE, EUCLID, DOT)
            recreate: If True, delete existing collection first
            
        Returns:
            True if successful
        """
        collection_name = collection_name or self.collection_name
        vector_size = vector_size or self.vector_size
        
        try:
            # Check if collection exists
            if self.collection_exists(collection_name):
                if recreate:
                    logger.warning(f"Deleting existing collection: {collection_name}")
                    self.client.delete_collection(collection_name)
                else:
                    logger.info(f"Collection {collection_name} already exists")
                    return True
            
            # Create collection
            logger.info(f"Creating collection: {collection_name} (vector_size={vector_size})")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance
                )
            )
            logger.info(f"Collection {collection_name} created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def delete_collection(self, collection_name: Optional[str] = None) -> bool:
        """
        Delete a collection
        
        Args:
            collection_name: Name of collection (default: from settings)
            
        Returns:
            True if successful
        """
        collection_name = collection_name or self.collection_name
        
        try:
            if self.collection_exists(collection_name):
                logger.info(f"Deleting collection: {collection_name}")
                self.client.delete_collection(collection_name)
                logger.info(f"Collection {collection_name} deleted")
                return True
            else:
                logger.warning(f"Collection {collection_name} does not exist")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise
    
    def get_collection_info(self, collection_name: Optional[str] = None) -> dict:
        """
        Get information about a collection
        
        Args:
            collection_name: Name of collection (default: from settings)
            
        Returns:
            Collection information
        """
        collection_name = collection_name or self.collection_name
        
        try:
            info = self.client.get_collection(collection_name)
            return {
                'name': collection_name,
                'vectors_count': info.vectors_count,
                'points_count': info.points_count,
                'status': info.status,
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            raise
    
    def get_client(self) -> QdrantClient:
        """
        Get the underlying Qdrant client
        
        Returns:
            QdrantClient instance
        """
        return self.client
