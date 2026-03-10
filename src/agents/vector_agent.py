"""
Vector Agent for Music Maven.

Handles semantic search over the Music4All dataset using two Qdrant collections:
  - music4all_lyrics : full lyric text, multilingual model
  - music4all_tags   : artist + song_name + genres + tags, English model

Query types handled:
  - Mood / theme queries  → searches lyrics collection
  - Style / genre queries → searches tags collection
  - Similar song queries  → searches both, merges by score
"""

from typing import Dict, Any, List, Optional

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from .base_agent import BaseAgent
from src.config.settings import get_settings
from src.vectordb.client import QdrantManager
from src.vectordb.operations import VectorDBOperations


class VectorAgent(BaseAgent):
    """
    Semantic search agent over Music4All lyric and tag collections.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        super().__init__(name="VectorAgent")
        self.agent_type = "vector"
        self.settings = get_settings()

        logger.info("Loading embedding models…")
        self.lyric_model = SentenceTransformer(self.settings.lyric_embedding_model)
        self.tag_model   = SentenceTransformer(self.settings.local_embedding_model)
        logger.info("Embedding models loaded.")

        # Two VectorDBOperations instances — one per collection
        lyric_manager = QdrantManager(
            collection_name=self.settings.lyric_collection_name,
            vector_size=384,
        )
        tag_manager = QdrantManager(
            collection_name=self.settings.tag_collection_name,
            vector_size=384,
        )
        self.lyric_ops = VectorDBOperations(lyric_manager)
        self.tag_ops   = VectorDBOperations(tag_manager)

        logger.info("Vector Agent initialised (lyrics + tags collections).")

    # ----------------------------------------------------------------------- #
    # BaseAgent interface
    # ----------------------------------------------------------------------- #

    def process_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Route the query to the right collection(s) and return results.

        Strategy:
          - Default: search tags (fast, works for most music queries)
          - If query contains lyric-mood keywords: also search lyrics, merge results
        """
        if not self.validate_query(query):
            return self.format_response(
                success=False,
                error="Query cannot be empty"
            )

        try:
            limit = kwargs.get("limit", 10)
            filters = kwargs.get("filters", {})

            lyric_keywords = {
                "lyric", "lyrics", "song about", "songs about",
                "mood", "melanchol", "sad", "happy", "uplifting",
                "heartbreak", "love song", "angry", "nostalgic",
                "theme", "meaning", "feel", "feeling",
            }
            use_lyrics = any(kw in query.lower() for kw in lyric_keywords)

            if use_lyrics:
                results = self._search_both(query, limit=limit, filters=filters)
                source = "lyrics + tags"
            else:
                results = self._search_tags(query, limit=limit, filters=filters)
                source = "tags"

            if not results:
                return self.format_response(
                    success=True,
                    data={"answer": "No matching songs found. Try rephrasing your query.", "results": []},
                    metadata={"source": source, "count": 0},
                )

            answer = self._format_answer(query, results)
            return self.format_response(
                success=True,
                data={"answer": answer, "results": results},
                metadata={"source": source, "count": len(results)},
            )

        except Exception as e:
            logger.error(f"Vector agent error: {e}")
            return self.format_response(success=False, error=str(e))

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.agent_type,
            "description": "Semantic search over Music4All lyrics and tags",
            "collections": [
                self.settings.lyric_collection_name,
                self.settings.tag_collection_name,
            ],
            "best_for": [
                "Find songs with a similar mood or feel",
                "Songs lyrically about a specific theme",
                "Style or genre similarity queries",
                "Find songs similar to a given title",
            ],
        }

    # ----------------------------------------------------------------------- #
    # Search helpers
    # ----------------------------------------------------------------------- #

    def _search_tags(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Embed query with tag model and search music4all_tags."""
        vec = self._embed(query, self.tag_model)
        return self.tag_ops.search(vec, limit=limit, filters=filters or {})

    def _search_lyrics(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Embed query with lyric model and search music4all_lyrics."""
        vec = self._embed(query, self.lyric_model)
        return self.lyric_ops.search(vec, limit=limit, filters=filters or {})

    def _search_both(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search both collections and merge by score.

        Deduplicate on song_id, keeping the highest score seen for each song.
        This is a simple score-based fusion — a production system would use
        Reciprocal Rank Fusion (RRF) for better cross-collection ranking.
        """
        lyric_results = self._search_lyrics(query, limit=limit, filters=filters)
        tag_results   = self._search_tags(query, limit=limit, filters=filters)

        # Merge: keep best score per song_id
        seen: Dict[str, Dict] = {}
        for result in lyric_results + tag_results:
            sid = result.get("id")
            if sid is None:
                continue
            if sid not in seen or result["score"] > seen[sid]["score"]:
                seen[sid] = result

        merged = sorted(seen.values(), key=lambda r: r["score"], reverse=True)
        return merged[:limit]

    # ----------------------------------------------------------------------- #
    # Public search methods (called by orchestrator / other agents)
    # ----------------------------------------------------------------------- #

    def search_by_mood(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Search lyrics collection for mood / theme queries."""
        try:
            results = self._search_lyrics(query, limit=limit, filters=filters)
            return self.format_response(
                success=True,
                data={"answer": self._format_answer(query, results), "results": results},
                metadata={"source": "lyrics", "count": len(results)},
            )
        except Exception as e:
            logger.error(f"search_by_mood error: {e}")
            return self.format_response(success=False, error=str(e))

    def search_by_style(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Search tags collection for style / genre queries."""
        try:
            results = self._search_tags(query, limit=limit, filters=filters)
            return self.format_response(
                success=True,
                data={"answer": self._format_answer(query, results), "results": results},
                metadata={"source": "tags", "count": len(results)},
            )
        except Exception as e:
            logger.error(f"search_by_style error: {e}")
            return self.format_response(success=False, error=str(e))

    def find_similar_songs(
        self,
        song_id: str,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """
        Find songs similar to a given song_id.
        Retrieves the song's tag vector and searches the tags collection.
        """
        try:
            ref = self.tag_ops.get_document_by_id(song_id)
            if ref is None:
                return self.format_response(
                    success=False,
                    error=f"Song '{song_id}' not found in tags collection",
                )

            tag_text = " ".join(filter(None, [
                ref.get("metadata", {}).get("artist", ""),
                ref.get("song_name", ""),
                ref.get("metadata", {}).get("genres", ""),
                ref.get("metadata", {}).get("tags", ""),
            ]))
            vec = self._embed(tag_text, self.tag_model)
            results = self.tag_ops.search(vec, limit=limit + 1)

            results = [r for r in results if r.get("id") != song_id][:limit]

            return self.format_response(
                success=True,
                data={
                    "answer": self._format_answer(f"songs similar to {ref.get('song_name', song_id)}", results),
                    "results": results,
                },
                metadata={"source": "tags", "reference_song_id": song_id},
            )
        except Exception as e:
            logger.error(f"find_similar_songs error: {e}")
            return self.format_response(success=False, error=str(e))

    # ----------------------------------------------------------------------- #
    # Utilities
    # ----------------------------------------------------------------------- #

    @staticmethod
    def _embed(text: str, model: SentenceTransformer) -> np.ndarray:
        """Encode a single query string into a normalised vector."""
        return model.encode(text, normalize_embeddings=True)

    @staticmethod
    def _format_answer(query: str, results: List[Dict[str, Any]]) -> str:
        """Format search results as a markdown answer string."""
        if not results:
            return "No matching songs found."

        lines = [f"Here are songs matching **{query}**:\n"]
        for r in results:
            name   = r.get("song_name") or r.get("metadata", {}).get("song_name", "Unknown")
            artist = r.get("artist")    or r.get("metadata", {}).get("artist", "Unknown")
            score  = r.get("score", 0)
            genres = r.get("genres")    or r.get("metadata", {}).get("genres", "")
            line   = f"- **{name}** by **{artist}** – similarity: {score:.0%}"
            if genres:
                line += f", genres: {genres}"
            lines.append(line)

        return "\n".join(lines)
