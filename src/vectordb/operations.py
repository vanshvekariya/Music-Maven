"""Vector database operations for indexing and searching"""

from typing import List, Dict, Any, Optional
import numpy as np
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
from loguru import logger
from tqdm import tqdm

from .client import QdrantManager
from src.config import get_settings


class VectorDBOperations:
    """High-level operations for vector database"""
    
    def __init__(self, qdrant_manager: Optional[QdrantManager] = None):
        """
        Initialize vector DB operations
        
        Args:
            qdrant_manager: QdrantManager instance (creates new if None)
        """
        self.manager = qdrant_manager or QdrantManager()
        self.client = self.manager.get_client()
        self.collection_name = self.manager.collection_name
        self.settings = get_settings()
    
    def index_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: np.ndarray,
        batch_size: Optional[int] = None
    ) -> int:
        """
        Index documents with their embeddings
        
        Args:
            documents: List of document dictionaries with 'id', 'text', and 'metadata'
            embeddings: Array of embeddings corresponding to documents
            batch_size: Batch size for uploading (default: from settings)
            
        Returns:
            Number of documents indexed
        """
        if len(documents) != len(embeddings):
            raise ValueError(f"Mismatch: {len(documents)} documents but {len(embeddings)} embeddings")
        
        batch_size = batch_size or self.settings.batch_size
        
        logger.info(f"Indexing {len(documents)} documents in batches of {batch_size}")
        
        # Prepare points
        points = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            point = PointStruct(
                id=i,  # Use sequential ID for Qdrant
                vector=embedding.tolist(),
                payload={
                    'song_id': doc['id'],
                    'text': doc['text'],
                    **doc['metadata']
                }
            )
            points.append(point)
        
        # Upload in batches
        total_indexed = 0
        for i in tqdm(range(0, len(points), batch_size), desc="Uploading batches"):
            batch = points[i:i + batch_size]
            
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                total_indexed += len(batch)
                
            except Exception as e:
                logger.error(f"Error uploading batch {i // batch_size}: {e}")
                raise
        
        logger.info(f"Successfully indexed {total_indexed} documents")
        return total_indexed
    
    def index_documents_with_offset(
        self,
        documents: List[Dict[str, Any]],
        embeddings: np.ndarray,
        offset: int = 0,
        batch_size: Optional[int] = None
    ) -> int:
        """
        Index documents using a global offset for point IDs.

        Args:
            documents: List of document dicts with 'id', 'text', 'metadata'
            embeddings: Embedding array matching documents
            offset: Starting integer for Qdrant point IDs (avoids collisions
                    when calling in a loop over large datasets)
            batch_size: Upload batch size

        Returns:
            Number of documents indexed
        """
        if len(documents) != len(embeddings):
            raise ValueError(f"Mismatch: {len(documents)} docs vs {len(embeddings)} embeddings")

        batch_size = batch_size or self.settings.batch_size

        points = [
            PointStruct(
                id=offset + i,
                vector=embedding.tolist(),
                payload={
                    'song_id': doc['id'],
                    'text':    doc['text'],
                    **doc['metadata'],
                }
            )
            for i, (doc, embedding) in enumerate(zip(documents, embeddings))
        ]

        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                )
            except Exception as e:
                logger.error(f"Error uploading batch at offset {offset + i}: {e}")
                raise

        return len(points)

    def search(
        self,
        query_vector: np.ndarray,
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors
        
        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            filters: Optional metadata filters
            
        Returns:
            List of search results with scores and metadata
        """
        try:
            # Build filter if provided
            qdrant_filter = None
            if filters:
                qdrant_filter = self._build_filter(filters)

            # Qdrant client v1.7+ replaced .search() with .query_points()
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector.tolist(),
                limit=limit,
                score_threshold=score_threshold,
                query_filter=qdrant_filter,
                with_payload=True,
            )
            results = response.points

            # Format results with Music4All metadata
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'id': result.payload.get('song_id'),
                    'score': result.score,
                    'song_name': result.payload.get('song_name', ''),
                    'artist': result.payload.get('artist', ''),
                    'album': result.payload.get('album', ''),
                    'lang': result.payload.get('lang', ''),
                    'spotify_id': result.payload.get('spotify_id', ''),
                    'popularity': result.payload.get('popularity', 0),
                    'danceability': result.payload.get('danceability', 0.0),
                    'energy': result.payload.get('energy', 0.0),
                    'valence': result.payload.get('valence', 0.0),
                    'tempo': result.payload.get('tempo', 0.0),
                    'mode': result.payload.get('mode', None),
                    'tags': result.payload.get('tags', []),
                    'genres': result.payload.get('genres', []),
                    'has_lyrics': result.payload.get('has_lyrics', False),
                    'audio_path': result.payload.get('audio_path', ''),
                    'text': result.payload.get('text', ''),
                    'metadata': result.payload
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise
    
    def _build_filter(self, filters: Dict[str, Any]) -> Filter:
        """
        Build Qdrant filter from dictionary for Music4All metadata.
        
        Args:
            filters: Filter dictionary with various filter options:
                - artist: Artist name (exact match)
                - lang: Language code e.g. "en" (exact match)
                - mode: 1=major, 0=minor (exact match)
                - has_lyrics: True/False
                - min_popularity: Minimum Spotify popularity (0-100)
                - max_popularity: Maximum Spotify popularity (0-100)
                - min_tempo: Minimum tempo in BPM
                - max_tempo: Maximum tempo in BPM
                - min_energy: Minimum energy score (0.0-1.0)
                - max_energy: Maximum energy score (0.0-1.0)
                - min_danceability: Minimum danceability score (0.0-1.0)
                - max_danceability: Maximum danceability score (0.0-1.0)
                - min_valence: Minimum valence score (0.0-1.0)
                - max_valence: Maximum valence score (0.0-1.0)
                - genres: List of genre strings (any match)
                - tags: List of tag strings (any match)
            
        Returns:
            Qdrant Filter object
        """
        conditions = []

        # Artist filter
        if 'artist' in filters:
            conditions.append(
                FieldCondition(
                    key='artist',
                    match=MatchValue(value=filters['artist'])
                )
            )

        # Language filter
        if 'lang' in filters:
            conditions.append(
                FieldCondition(
                    key='lang',
                    match=MatchValue(value=filters['lang'])
                )
            )

        # Mode filter (1=major, 0=minor)
        if 'mode' in filters:
            conditions.append(
                FieldCondition(
                    key='mode',
                    match=MatchValue(value=filters['mode'])
                )
            )

        # Has lyrics filter
        if 'has_lyrics' in filters:
            conditions.append(
                FieldCondition(
                    key='has_lyrics',
                    match=MatchValue(value=filters['has_lyrics'])
                )
            )

        # Popularity range
        if 'min_popularity' in filters:
            conditions.append(
                FieldCondition(
                    key='popularity',
                    range={'gte': filters['min_popularity']}
                )
            )
        if 'max_popularity' in filters:
            conditions.append(
                FieldCondition(
                    key='popularity',
                    range={'lte': filters['max_popularity']}
                )
            )

        # Tempo range
        if 'min_tempo' in filters:
            conditions.append(
                FieldCondition(
                    key='tempo',
                    range={'gte': filters['min_tempo']}
                )
            )
        if 'max_tempo' in filters:
            conditions.append(
                FieldCondition(
                    key='tempo',
                    range={'lte': filters['max_tempo']}
                )
            )

        # Energy range
        if 'min_energy' in filters:
            conditions.append(
                FieldCondition(
                    key='energy',
                    range={'gte': filters['min_energy']}
                )
            )
        if 'max_energy' in filters:
            conditions.append(
                FieldCondition(
                    key='energy',
                    range={'lte': filters['max_energy']}
                )
            )

        # Danceability range
        if 'min_danceability' in filters:
            conditions.append(
                FieldCondition(
                    key='danceability',
                    range={'gte': filters['min_danceability']}
                )
            )
        if 'max_danceability' in filters:
            conditions.append(
                FieldCondition(
                    key='danceability',
                    range={'lte': filters['max_danceability']}
                )
            )

        # Valence range
        if 'min_valence' in filters:
            conditions.append(
                FieldCondition(
                    key='valence',
                    range={'gte': filters['min_valence']}
                )
            )
        if 'max_valence' in filters:
            conditions.append(
                FieldCondition(
                    key='valence',
                    range={'lte': filters['max_valence']}
                )
            )

        # Genres filter (match any genre in the list)
        if 'genres' in filters and filters['genres']:
            for genre in filters['genres']:
                conditions.append(
                    FieldCondition(
                        key='genres',
                        match=MatchValue(value=genre)
                    )
                )

        # Tags filter (match any tag in the list)
        if 'tags' in filters and filters['tags']:
            for tag in filters['tags']:
                conditions.append(
                    FieldCondition(
                        key='tags',
                        match=MatchValue(value=tag)
                    )
                )

        if conditions:
            return Filter(must=conditions)

        return None
    
    def get_document_by_id(self, song_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by its song ID.
        
        Args:
            song_id: Music4All song identifier
            
        Returns:
            Document data or None if not found
        """
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key='song_id',
                            match=MatchValue(value=song_id)
                        )
                    ]
                ),
                limit=1
            )

            if results[0]:
                point = results[0][0]
                return {
                    'id': point.payload.get('song_id'),
                    'song_name': point.payload.get('song_name', ''),
                    'artist': point.payload.get('artist', ''),
                    'metadata': point.payload
                }

            return None

        except Exception as e:
            logger.error(f"Error retrieving document: {e}")
            return None
    
    def count_documents(self) -> int:
        """
        Count total documents in collection
        
        Returns:
            Number of documents
        """
        try:
            info = self.manager.get_collection_info()
            return info['points_count']
        except Exception as e:
            logger.error(f"Error counting documents: {e}")
            return 0
    
    def get_filter_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about available filter values in the collection.
        Useful for understanding what filters can be applied.
        
        Returns:
            Dictionary with filter statistics
        """
        try:
            # Scroll through all documents to gather statistics
            all_points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000  # Adjust based on collection size
            )
            
            languages = set()
            artists = set()
            genres = set()

            for point in all_points:
                payload = point.payload
                if 'lang' in payload:
                    languages.add(payload['lang'])
                if 'artist' in payload:
                    artists.add(payload['artist'])
                if 'genres' in payload:
                    for g in (payload['genres'] or []):
                        genres.add(g)

            return {
                'total_documents': len(all_points),
                'available_languages': sorted(list(languages)),
                'total_artists': len(artists),
                'sample_artists': sorted(list(artists))[:20],
                'available_genres': sorted(list(genres)),
            }
            
        except Exception as e:
            logger.error(f"Error getting filter statistics: {e}")
            return {}
