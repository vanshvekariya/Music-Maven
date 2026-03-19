"""
Music4All data processor.

Loads and merges the 5 Music4All CSV files, reads lyrics from the
lyrics/ subfolder, and writes two SQLite tables:
  - songs             : merged metadata for all 109k songs
  - listening_history : user listening history (user, song, timestamp)

No audio processing — audio clips are not available yet.
"""

import sqlite3
from pathlib import Path

import pandas as pd
from loguru import logger


class Music4AllProcessor:
    """
    Loads Music4All CSVs, merges them on song_id, and writes SQLite tables.

    Expected layout inside `data_dir`:
        id_information.csv
        id_metadata.csv
        id_lang.csv
        id_tags.csv
        id_genres.csv
        listening_history.csv
        lyrics/<song_id>.txt   (optional — used to mark has_lyrics)
    """

    # Column names as they appear in each CSV (tab-separated)
    _INFO_COLS    = ["id", "artist", "song", "album_name"]
    _META_COLS    = ["id", "spotify_id", "popularity", "release",
                     "danceability", "energy", "key", "mode",
                     "valence", "tempo", "duration_ms"]
    _LANG_COLS    = ["id", "lang"]
    _TAGS_COLS    = ["id", "tags"]
    _GENRES_COLS  = ["id", "genres"]
    _HIST_COLS    = ["user", "song", "timestamp"]

    def __init__(
        self,
        data_dir: str = "data/raw/dataset",
        db_path: str = "music4all.db",
    ):
        self.data_dir  = Path(data_dir)
        self.lyrics_dir = self.data_dir / "lyrics"
        self.db_path   = db_path

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def process(self) -> pd.DataFrame:
        """
        Full pipeline:
          1. Load & merge the 5 metadata CSVs
          2. Mark which songs have lyrics
          3. Write `songs` table to SQLite
          4. Write `listening_history` table to SQLite

        Returns the merged songs DataFrame.
        """
        logger.info("=" * 60)
        logger.info("Music4All ingestion pipeline starting")
        logger.info(f"Data directory : {self.data_dir}")
        logger.info(f"SQLite database: {self.db_path}")
        logger.info("=" * 60)

        songs_df = self._load_and_merge()
        songs_df = self._add_lyrics_flag(songs_df)
        self._write_songs_table(songs_df)
        self._write_listening_history()

        logger.info("Pipeline complete.")
        return songs_df

    # ------------------------------------------------------------------
    # Step 1 — load & merge
    # ------------------------------------------------------------------

    def _read_csv(self, filename: str) -> pd.DataFrame:
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
        df = pd.read_csv(path, sep="\t")
        df.columns = [c.strip() for c in df.columns]
        logger.info(f"Loaded {filename}: {len(df):,} rows, columns: {list(df.columns)}")
        return df

    def _load_and_merge(self) -> pd.DataFrame:
        """Load the 5 metadata CSVs and merge them on `id` → renamed `song_id`."""

        info    = self._read_csv("id_information.csv").rename(columns={"id": "song_id", "song": "song_name", "album_name": "album"})
        meta    = self._read_csv("id_metadata.csv").rename(columns={"id": "song_id"})
        lang    = self._read_csv("id_lang.csv").rename(columns={"id": "song_id"})
        tags    = self._read_csv("id_tags.csv").rename(columns={"id": "song_id"})
        genres  = self._read_csv("id_genres.csv").rename(columns={"id": "song_id"})

        df = (
            info
            .merge(meta,   on="song_id", how="left")
            .merge(lang,   on="song_id", how="left")
            .merge(tags,   on="song_id", how="left")
            .merge(genres, on="song_id", how="left")
        )

        # Fill missing text fields with empty string
        for col in ["artist", "song_name", "album", "lang", "tags", "genres", "spotify_id"]:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)

        # Fill missing numeric fields with 0
        num_cols = ["popularity", "release", "danceability", "energy",
                    "key", "mode", "valence", "tempo", "duration_ms"]
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        logger.info(f"Merged dataset: {len(df):,} songs, {len(df.columns)} columns")
        return df

    # ------------------------------------------------------------------
    # Step 2 — lyrics flag
    # ------------------------------------------------------------------

    def _add_lyrics_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a boolean `has_lyrics` column based on lyrics folder contents."""
        if self.lyrics_dir.exists():
            lyric_ids = {p.stem for p in self.lyrics_dir.glob("*.txt")}
            df["has_lyrics"] = df["song_id"].isin(lyric_ids)
            count = df["has_lyrics"].sum()
            logger.info(f"Lyrics folder found — {count:,} / {len(df):,} songs have lyrics")
        else:
            df["has_lyrics"] = False
            logger.warning(f"Lyrics folder not found at {self.lyrics_dir} — has_lyrics set to False")
        return df

    # ------------------------------------------------------------------
    # Step 3 — write songs table
    # ------------------------------------------------------------------

    def _write_songs_table(self, df: pd.DataFrame) -> None:
        """Write the merged songs DataFrame to the `songs` SQLite table."""
        logger.info(f"Writing `songs` table → {self.db_path}")

        # Select and order final columns
        cols = [
            "song_id", "artist", "song_name", "album",
            "lang", "spotify_id", "popularity", "release",
            "danceability", "energy", "key", "mode",
            "valence", "tempo", "duration_ms",
            "tags", "genres", "has_lyrics",
        ]
        # Keep only columns that actually exist in the DataFrame
        cols = [c for c in cols if c in df.columns]
        songs_df = df[cols].copy()

        with sqlite3.connect(self.db_path) as conn:
            songs_df.to_sql("songs", conn, if_exists="replace", index=False)

            # Useful indexes for fast querying
            conn.execute("CREATE INDEX IF NOT EXISTS idx_artist     ON songs(artist)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_popularity ON songs(popularity)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tempo      ON songs(tempo)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_lang       ON songs(lang)")

        logger.info(f"`songs` table: {len(songs_df):,} rows written")

    # ------------------------------------------------------------------
    # Step 4 — write listening history table
    # ------------------------------------------------------------------

    def _write_listening_history(self) -> None:
        """Write listening_history.csv to the `listening_history` SQLite table."""
        path = self.data_dir / "listening_history.csv"
        if not path.exists():
            logger.warning(f"listening_history.csv not found at {path} — skipping")
            return

        hist_df = pd.read_csv(path, sep="\t")
        hist_df.columns = [c.strip() for c in hist_df.columns]
        logger.info(f"Loaded listening_history.csv: {len(hist_df):,} rows")

        with sqlite3.connect(self.db_path) as conn:
            hist_df.to_sql("listening_history", conn, if_exists="replace", index=False)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_hist_user ON listening_history(user)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_hist_song ON listening_history(song)")

        logger.info(f"`listening_history` table: {len(hist_df):,} rows written")


if __name__ == "__main__":
    processor = Music4AllProcessor()
    processor.process()
