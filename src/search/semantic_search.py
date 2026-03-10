"""Semantic search functionality"""

from typing import List, Dict, Any, Optional
from loguru import logger

from src.embeddings import get_embedding_model, BaseEmbedding
from src.vectordb import QdrantManager, VectorDBOperations


class SemanticSearch:
    """Semantic search over Music4All songs"""
    
    def __init__(
        self,
        embedding_model: Optional[BaseEmbedding] = None,
        qdrant_manager: Optional[QdrantManager] = None
    ):
        """
        Initialize semantic search
        
        Args:
            embedding_model: Embedding model (creates default if None)
            qdrant_manager: Qdrant manager (creates default if None)
        """
        logger.info("Initializing semantic search")
        
        # Initialize embedding model
        self.embedding_model = embedding_model or get_embedding_model()
        
        # Initialize vector DB
        self.qdrant_manager = qdrant_manager or QdrantManager()
        self.db_ops = VectorDBOperations(self.qdrant_manager)
        
        logger.info("Semantic search initialized")
    
    def search(
        self,
        query: str,
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for videos using natural language query
        
        Args:
            query: Natural language search query
            limit: Maximum number of results
            score_threshold: Minimum similarity score (0-1)
            filters: Optional metadata filters (category, country, min_views, etc.)
            
        Returns:
            List of matching videos with metadata
            
        Example:
            >>> search = SemanticSearch()
            >>> results = search.search("funny cat videos", limit=5)
            >>> for result in results:
            ...     print(f"{result['title']} - {result['score']:.3f}")
        """
        logger.info(f"Searching for: '{query}' (limit={limit})")
        
        # Generate query embedding
        query_vector = self.embedding_model.encode_single(query)
        
        # Search in vector DB
        results = self.db_ops.search(
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            filters=filters
        )
        
        logger.info(f"Found {len(results)} results")
        return results
    
    def search_by_category(
        self,
        query: str,
        category: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search within a specific category
        
        Args:
            query: Search query
            category: Category name (e.g., "Gaming", "Music")
            limit: Maximum number of results
            
        Returns:
            List of matching videos
        """
        return self.search(
            query=query,
            limit=limit,
            filters={'category': category}
        )
    
    def search_popular(
        self,
        query: str,
        min_views: int = 100000,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for popular videos (with minimum view count)
        
        Args:
            query: Search query
            min_views: Minimum number of views
            limit: Maximum number of results
            
        Returns:
            List of matching popular videos
        """
        return self.search(
            query=query,
            limit=limit,
            filters={'min_views': min_views}
        )
    
    def find_similar_videos(
        self,
        video_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find videos similar to a given video
        
        Args:
            video_id: ID of the reference video
            limit: Maximum number of results
            
        Returns:
            List of similar videos
        """
        # Get the reference video
        video = self.db_ops.get_document_by_id(video_id)
        
        if not video:
            logger.warning(f"Video {video_id} not found")
            return []
        
        # Use the video's text for similarity search
        video_text = video['metadata'].get('text', '')
        
        if not video_text:
            logger.warning(f"No text available for video {video_id}")
            return []
        
        # Search using the video's text
        results = self.search(query=video_text, limit=limit + 1)
        
        # Remove the reference video from results
        filtered_results = [r for r in results if r['id'] != video_id]
        
        return filtered_results[:limit]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the indexed data
        
        Returns:
            Dictionary with statistics
        """
        try:
            info = self.qdrant_manager.get_collection_info()
            return {
                'total_videos': info['points_count'],
                'collection_name': info['name'],
                'status': info['status'],
                'embedding_dimension': self.embedding_model.get_dimension()
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}
    
    def format_results(self, results: List[Dict[str, Any]], show_text: bool = False) -> str:
        """
        Format search results for display
        
        Args:
            results: Search results
            show_text: Whether to show the full searchable text
            
        Returns:
            Formatted string
        """
        if not results:
            return "No results found."
        
        output = []
        output.append(f"\n{'='*80}")
        output.append(f"Found {len(results)} results")
        output.append(f"{'='*80}\n")
        
        for i, result in enumerate(results, 1):
            output.append(f"{i}. {result['title']}")
            output.append(f"   Channel: {result['channel']}")
            output.append(f"   Category: {result['category']}")
            output.append(f"   Views: {result['views']:,} | Likes: {result['likes']:,}")
            output.append(f"   Similarity Score: {result['score']:.4f}")
            
            if result.get('tags'):
                tags_str = ', '.join(result['tags'][:5])
                output.append(f"   Tags: {tags_str}")
            
            if show_text:
                output.append(f"   Text: {result['text'][:200]}...")
            
            output.append("")
        
        return "\n".join(output)
