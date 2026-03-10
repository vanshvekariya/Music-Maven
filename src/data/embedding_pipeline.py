"""
Music4All Embedding Pipeline

Builds two Qdrant collections from the Music4All dataset:

  music4all_lyrics  — one vector per song, embedding the full lyric text.
                      Model: paraphrase-multilingual-MiniLM-L12-v2 (384-dim)
                      Skips instrumental placeholders (<50 chars).

  music4all_tags    — one vector per song, embedding a concatenation of
                      artist + song_name + genres + tags.
                      Model: all-MiniLM-L6-v2 (384-dim)

Run from the project root:
    python -m src.data.embedding_pipeline
"""

import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config.settings import get_settings
from src.vectordb.client import QdrantManager
from src.vectordb.operations import VectorDBOperations


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

INSTRUMENTAL_THRESHOLD = 50   # lyric files shorter than this are skipped
BATCH_SIZE = 256               # songs per embedding batch — tuned for CPU RAM


# --------------------------------------------------------------------------- #
# Main pipeline class
# --------------------------------------------------------------------------- #

class EmbeddingPipeline:
    """
    Reads data from SQLite, embeds it, and upserts into Qdrant.

    Two passes:
      1. Lyrics  → music4all_lyrics  (multilingual model)
      2. Tags    → music4all_tags    (English model)
    """

    def __init__(self, recreate_collections: bool = False):
        self.settings   = get_settings()
        self.db_path    = self.settings.sql_db_path
        self.lyrics_dir = Path(self.settings.raw_data_dir) / "lyrics"
        self.recreate   = recreate_collections

        logger.info("Initialising embedding models…")
        # Lyric model — multilingual so pt / es / ko lyrics embed meaningfully
        self.lyric_model = SentenceTransformer(self.settings.lyric_embedding_model)
        # Tag model — lightweight English model, fast for short text
        self.tag_model   = SentenceTransformer(self.settings.local_embedding_model)
        logger.info("Models loaded.")

        # Two separate Qdrant managers — one per collection
        self.lyric_manager = QdrantManager(
            collection_name=self.settings.lyric_collection_name,
            vector_size=384,
        )
        self.tag_manager = QdrantManager(
            collection_name=self.settings.tag_collection_name,
            vector_size=384,
        )

        self.lyric_ops = VectorDBOperations(self.lyric_manager)
        self.tag_ops   = VectorDBOperations(self.tag_manager)

    # ----------------------------------------------------------------------- #
    # Public entry point
    # ----------------------------------------------------------------------- #

    def run(self) -> None:
        """Run both embedding passes."""
        logger.info("=" * 60)
        logger.info("Music4All Embedding Pipeline")
        logger.info(f"SQLite  : {self.db_path}")
        logger.info(f"Lyrics  : {self.lyrics_dir}")
        logger.info("=" * 60)

        self._ensure_collections()
        songs = self._load_songs()

        logger.info(f"Loaded {len(songs):,} songs from SQLite")

        self._run_lyrics_pass(songs)
        self._run_tags_pass(songs)

        logger.info("Pipeline complete.")

    # ----------------------------------------------------------------------- #
    # Collection setup
    # ----------------------------------------------------------------------- #

    def _ensure_collections(self) -> None:
        """Create Qdrant collections if they don't exist."""
        self.lyric_manager.create_collection(recreate=self.recreate)
        self.tag_manager.create_collection(recreate=self.recreate)

    # ----------------------------------------------------------------------- #
    # Data loading
    # ----------------------------------------------------------------------- #

    def _load_songs(self) -> list[dict]:
        """Load all songs from SQLite."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT song_id, artist, song_name, album, lang, "
            "popularity, danceability, energy, valence, tempo, "
            "mode, has_lyrics, tags, genres "
            "FROM songs"
        )
        songs = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return songs

    def _read_lyric(self, song_id: str) -> Optional[str]:
        """
        Read lyric file for a song.
        Returns None if the file is missing or is an instrumental placeholder.
        """
        path = self.lyrics_dir / f"{song_id}.txt"
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if len(text) < INSTRUMENTAL_THRESHOLD:
            return None          # placeholder like "INSTRUMENTAL"
        return text

    # ----------------------------------------------------------------------- #
    # Pass 1 — lyrics
    # ----------------------------------------------------------------------- #

    def _run_lyrics_pass(self, songs: list[dict]) -> None:
        """Embed lyrics and upsert into music4all_lyrics."""
        logger.info("--- Pass 1: Lyrics ---")

        docs, texts = [], []
        for song in songs:
            lyric = self._read_lyric(song["song_id"])
            if lyric is None:
                continue
            docs.append(song)
            texts.append(lyric)

        logger.info(f"Songs with usable lyrics: {len(docs):,} / {len(songs):,}")
        self._embed_and_index(docs, texts, self.lyric_model, self.lyric_ops, label="lyrics")

    # ----------------------------------------------------------------------- #
    # Pass 2 — tags
    # ----------------------------------------------------------------------- #

    def _run_tags_pass(self, songs: list[dict]) -> None:
        """Embed tag documents and upsert into music4all_tags."""
        logger.info("--- Pass 2: Tags ---")

        docs, texts = [], []
        for song in songs:
            tag_text = self._build_tag_text(song)
            docs.append(song)
            texts.append(tag_text)

        logger.info(f"Building tag embeddings for {len(docs):,} songs")
        self._embed_and_index(docs, texts, self.tag_model, self.tag_ops, label="tags")

    @staticmethod
    def _build_tag_text(song: dict) -> str:
        """
        Concatenate artist + song_name + genres + tags into one string.
        Putting artist/title first gives the model better anchoring —
        a query for 'Beyoncé upbeat' will match even if tags don't say 'Beyoncé'.
        """
        parts = [
            song.get("artist", ""),
            song.get("song_name", ""),
            song.get("genres", ""),
            song.get("tags", ""),
        ]
        return " ".join(p for p in parts if p).strip()

    # ----------------------------------------------------------------------- #
    # Shared embedding + indexing
    # ----------------------------------------------------------------------- #

    def _embed_and_index(
        self,
        docs: list[dict],
        texts: list[str],
        model: SentenceTransformer,
        ops: VectorDBOperations,
        label: str,
    ) -> None:
        """
        Embed texts in batches and upsert into Qdrant.

        Batch processing is critical here: loading 109k embeddings into memory
        at once would require ~160 MB for 384-dim float32 vectors.
        Processing in batches of 256 keeps peak RAM under 1 MB per batch.
        """
        total = len(docs)
        indexed = 0

        for start in tqdm(range(0, total, BATCH_SIZE), desc=f"Embedding {label}"):
            batch_docs  = docs[start : start + BATCH_SIZE]
            batch_texts = texts[start : start + BATCH_SIZE]

            # Encode — returns numpy array of shape (batch_size, 384)
            embeddings: np.ndarray = model.encode(
                batch_texts,
                batch_size=64,          # inner encode batch
                show_progress_bar=False,
                normalize_embeddings=True,   # unit vectors → cosine = dot product
            )

            # Wrap into the format VectorDBOperations.index_documents expects
            wrapped_docs = [
                {
                    "id":       d["song_id"],
                    "text":     t,
                    "metadata": {k: v for k, v in d.items() if k != "song_id"},
                }
                for d, t in zip(batch_docs, batch_texts)
            ]

            # Use a global offset so Qdrant point IDs don't collide across batches
            ops.index_documents_with_offset(wrapped_docs, embeddings, offset=start)
            indexed += len(batch_docs)

        logger.info(f"{label} pass complete — {indexed:,} vectors indexed")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Music4All Embedding Pipeline")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate Qdrant collections before indexing",
    )
    args = parser.parse_args()

    pipeline = EmbeddingPipeline(recreate_collections=args.recreate)
    pipeline.run()
