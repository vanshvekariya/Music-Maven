"""
Knowledge Graph Builder for Music Maven.

Reads the Music4All SQLite database and constructs an in-memory NetworkX
graph with Song, Artist, Genre, Language, Tag, and Tier nodes plus all
relationship edges.  Pre-computed aggregations are stored as graph-level
attributes so the KG query engine can answer common questions in O(1).

Usage:
    python -m src.knowledge_graph.kg_builder          # build & serialize
    python -m src.knowledge_graph.kg_builder --rebuild # force rebuild
"""

import pickle
import sqlite3
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Any, Optional

import networkx as nx
from loguru import logger

from src.config.settings import get_settings


# ── Tier definitions ─────────────────────────────────────────────────────────

POPULARITY_TIERS = [
    ("0-19", 0, 19),
    ("20-39", 20, 39),
    ("40-59", 40, 59),
    ("60-79", 60, 79),
    ("80-100", 80, 100),
]

ENERGY_TIERS = [
    ("very_low", 0.0, 0.2),
    ("low", 0.2, 0.4),
    ("medium", 0.4, 0.6),
    ("high", 0.6, 0.8),
    ("very_high", 0.8, 1.0),
]

DANCEABILITY_TIERS = ENERGY_TIERS  # same 0-1 ranges

TEMPO_TIERS = [
    ("slow", 0, 90),
    ("moderate", 90, 120),
    ("fast", 120, 150),
    ("very_fast", 150, 300),
]

MIN_TAG_FREQUENCY = 5  # only keep tags that appear on at least this many songs


# ── Builder ──────────────────────────────────────────────────────────────────

class KnowledgeGraphBuilder:
    """Builds and persists the Music4All knowledge graph."""

    def __init__(self, db_path: Optional[str] = None, pickle_path: Optional[str] = None):
        settings = get_settings()
        self.db_path = db_path or settings.sql_db_path
        self.pickle_path = Path(pickle_path or settings.kg_pickle_path)
        self.G: Optional[nx.DiGraph] = None

    # ── public API ───────────────────────────────────────────────────────

    def build(self) -> nx.DiGraph:
        """Build the full knowledge graph from SQLite."""
        logger.info(f"Building knowledge graph from {self.db_path}")
        self.G = nx.DiGraph()

        songs = self._load_songs()
        logger.info(f"Loaded {len(songs):,} songs")

        self._add_tier_nodes()
        self._add_song_nodes(songs)
        self._add_artist_nodes(songs)
        self._add_genre_nodes(songs)
        self._add_language_nodes(songs)
        self._add_tag_nodes(songs)
        self._add_genre_cooccurrence(songs)
        self._precompute_aggregations(songs)

        logger.info(
            f"KG built: {self.G.number_of_nodes():,} nodes, "
            f"{self.G.number_of_edges():,} edges"
        )
        return self.G

    def save(self) -> None:
        """Serialize graph to disk."""
        if self.G is None:
            raise RuntimeError("Graph not built yet")
        self.pickle_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.pickle_path, "wb") as f:
            pickle.dump(self.G, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"KG saved to {self.pickle_path}")

    @classmethod
    def load(cls, pickle_path: Optional[str] = None) -> nx.DiGraph:
        """Load a previously serialized graph."""
        settings = get_settings()
        path = Path(pickle_path or settings.kg_pickle_path)
        if not path.exists():
            raise FileNotFoundError(f"KG pickle not found at {path}")
        with open(path, "rb") as f:
            G = pickle.load(f)
        logger.info(f"KG loaded from {path} ({G.number_of_nodes():,} nodes)")
        return G

    @classmethod
    def load_or_build(cls, db_path: Optional[str] = None, pickle_path: Optional[str] = None) -> nx.DiGraph:
        """Load from pickle if available, otherwise build and save."""
        settings = get_settings()
        pkl = Path(pickle_path or settings.kg_pickle_path)
        if pkl.exists():
            return cls.load(str(pkl))
        builder = cls(db_path=db_path, pickle_path=str(pkl))
        builder.build()
        builder.save()
        return builder.G

    # ── data loading ─────────────────────────────────────────────────────

    def _load_songs(self) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT song_id, artist, song_name, album, lang, "
            "popularity, danceability, energy, valence, tempo, "
            "mode, duration_ms, has_lyrics, tags, genres "
            "FROM songs"
        )
        songs = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return songs

    # ── tier nodes ───────────────────────────────────────────────────────

    def _add_tier_nodes(self) -> None:
        for name, lo, hi in POPULARITY_TIERS:
            self.G.add_node(f"pop_tier:{name}", node_type="popularity_tier", label=name, lo=lo, hi=hi)
        for name, lo, hi in ENERGY_TIERS:
            self.G.add_node(f"energy_tier:{name}", node_type="energy_tier", label=name, lo=lo, hi=hi)
        for name, lo, hi in DANCEABILITY_TIERS:
            self.G.add_node(f"dance_tier:{name}", node_type="danceability_tier", label=name, lo=lo, hi=hi)
        for name, lo, hi in TEMPO_TIERS:
            self.G.add_node(f"tempo_tier:{name}", node_type="tempo_tier", label=name, lo=lo, hi=hi)

    # ── song nodes + tier edges ──────────────────────────────────────────

    def _add_song_nodes(self, songs: List[Dict]) -> None:
        for s in songs:
            sid = f"song:{s['song_id']}"
            self.G.add_node(
                sid,
                node_type="song",
                song_id=s["song_id"],
                song_name=s.get("song_name", ""),
                album=s.get("album", ""),
                lang=s.get("lang", ""),
                popularity=s.get("popularity", 0),
                danceability=s.get("danceability", 0.0),
                energy=s.get("energy", 0.0),
                valence=s.get("valence", 0.0),
                tempo=s.get("tempo", 0.0),
                mode=s.get("mode"),
                has_lyrics=s.get("has_lyrics", 0),
            )
            # tier edges
            pop = s.get("popularity", 0) or 0
            for name, lo, hi in POPULARITY_TIERS:
                if lo <= pop <= hi:
                    self.G.add_edge(sid, f"pop_tier:{name}", rel="IN_POPULARITY_TIER")
                    break

            energy = s.get("energy", 0.0) or 0.0
            for name, lo, hi in ENERGY_TIERS:
                if lo <= energy <= hi:
                    self.G.add_edge(sid, f"energy_tier:{name}", rel="IN_ENERGY_TIER")
                    break

            dance = s.get("danceability", 0.0) or 0.0
            for name, lo, hi in DANCEABILITY_TIERS:
                if lo <= dance <= hi:
                    self.G.add_edge(sid, f"dance_tier:{name}", rel="IN_DANCEABILITY_TIER")
                    break

            tempo = s.get("tempo", 0.0) or 0.0
            for name, lo, hi in TEMPO_TIERS:
                if lo <= tempo < hi or (name == "very_fast" and tempo >= hi):
                    self.G.add_edge(sid, f"tempo_tier:{name}", rel="IN_TEMPO_TIER")
                    break

    # ── artist nodes ─────────────────────────────────────────────────────

    def _add_artist_nodes(self, songs: List[Dict]) -> None:
        artist_songs: Dict[str, List[Dict]] = defaultdict(list)
        for s in songs:
            artist_songs[s["artist"]].append(s)

        for artist, a_songs in artist_songs.items():
            aid = f"artist:{artist}"
            n = len(a_songs)
            genres_counter: Counter = Counter()
            langs: set = set()
            for s in a_songs:
                if s.get("genres"):
                    for g in s["genres"].split(","):
                        g = g.strip()
                        if g:
                            genres_counter[g] += 1
                if s.get("lang"):
                    langs.add(s["lang"])

            self.G.add_node(
                aid,
                node_type="artist",
                name=artist,
                song_count=n,
                avg_popularity=sum(s.get("popularity", 0) or 0 for s in a_songs) / n,
                avg_energy=sum(s.get("energy", 0) or 0 for s in a_songs) / n,
                avg_danceability=sum(s.get("danceability", 0) or 0 for s in a_songs) / n,
                avg_valence=sum(s.get("valence", 0) or 0 for s in a_songs) / n,
                avg_tempo=sum(s.get("tempo", 0) or 0 for s in a_songs) / n,
                top_genres=[g for g, _ in genres_counter.most_common(5)],
                languages=sorted(langs),
            )

            # song → artist edges
            for s in a_songs:
                self.G.add_edge(f"song:{s['song_id']}", aid, rel="PERFORMED_BY")

            # artist → genre edges
            for genre, weight in genres_counter.items():
                gid = f"genre:{genre}"
                self.G.add_edge(aid, gid, rel="WORKS_IN_GENRE", weight=weight)

            # artist → language edges
            lang_counter: Counter = Counter(s.get("lang", "") for s in a_songs)
            for lang, weight in lang_counter.items():
                if lang:
                    self.G.add_edge(aid, f"lang:{lang}", rel="SINGS_IN", weight=weight)

    # ── genre nodes ──────────────────────────────────────────────────────

    def _add_genre_nodes(self, songs: List[Dict]) -> None:
        genre_songs: Dict[str, List[Dict]] = defaultdict(list)
        for s in songs:
            if s.get("genres"):
                for g in s["genres"].split(","):
                    g = g.strip()
                    if g:
                        genre_songs[g].append(s)

        for genre, g_songs in genre_songs.items():
            gid = f"genre:{genre}"
            n = len(g_songs)
            attrs = dict(
                node_type="genre",
                name=genre,
                song_count=n,
                avg_popularity=sum(s.get("popularity", 0) or 0 for s in g_songs) / n,
                avg_energy=sum(s.get("energy", 0) or 0 for s in g_songs) / n,
                avg_danceability=sum(s.get("danceability", 0) or 0 for s in g_songs) / n,
                avg_tempo=sum(s.get("tempo", 0) or 0 for s in g_songs) / n,
            )
            if gid in self.G:
                self.G.nodes[gid].update(attrs)
            else:
                self.G.add_node(gid, **attrs)

            for s in g_songs:
                self.G.add_edge(f"song:{s['song_id']}", gid, rel="HAS_GENRE")

    # ── language nodes ───────────────────────────────────────────────────

    def _add_language_nodes(self, songs: List[Dict]) -> None:
        lang_counter: Counter = Counter()
        for s in songs:
            if s.get("lang"):
                lang_counter[s["lang"]] += 1

        for lang, count in lang_counter.items():
            lid = f"lang:{lang}"
            self.G.add_node(lid, node_type="language", code=lang, song_count=count)
            for s in songs:
                if s.get("lang") == lang:
                    self.G.add_edge(f"song:{s['song_id']}", lid, rel="IN_LANGUAGE")

    # ── tag nodes ────────────────────────────────────────────────────────

    def _add_tag_nodes(self, songs: List[Dict]) -> None:
        tag_counter: Counter = Counter()
        tag_songs: Dict[str, List[str]] = defaultdict(list)
        for s in songs:
            if s.get("tags"):
                for t in s["tags"].split(","):
                    t = t.strip()
                    if t:
                        tag_counter[t] += 1
                        tag_songs[t].append(s["song_id"])

        for tag, count in tag_counter.items():
            if count < MIN_TAG_FREQUENCY:
                continue
            tid = f"tag:{tag}"
            self.G.add_node(tid, node_type="tag", name=tag, song_count=count)
            for song_id in tag_songs[tag]:
                self.G.add_edge(f"song:{song_id}", tid, rel="HAS_TAG")

    # ── genre co-occurrence ──────────────────────────────────────────────

    def _add_genre_cooccurrence(self, songs: List[Dict]) -> None:
        pair_counter: Counter = Counter()
        for s in songs:
            if not s.get("genres"):
                continue
            genres = [g.strip() for g in s["genres"].split(",") if g.strip()]
            for a, b in combinations(sorted(set(genres)), 2):
                pair_counter[(a, b)] += 1

        for (a, b), weight in pair_counter.items():
            if weight < 3:
                continue
            ga, gb = f"genre:{a}", f"genre:{b}"
            self.G.add_edge(ga, gb, rel="RELATED_TO", weight=weight)
            self.G.add_edge(gb, ga, rel="RELATED_TO", weight=weight)

    # ── pre-computed aggregations ────────────────────────────────────────

    def _precompute_aggregations(self, songs: List[Dict]) -> None:
        """Store sorted leaderboards and global stats on graph.graph."""

        # --- artist leaderboards ---
        artist_nodes = [
            (nid, d)
            for nid, d in self.G.nodes(data=True)
            if d.get("node_type") == "artist"
        ]

        self.G.graph["top_artists_by_popularity"] = sorted(
            [(d["name"], d["avg_popularity"], d["song_count"]) for _, d in artist_nodes],
            key=lambda x: x[1],
            reverse=True,
        )[:50]

        self.G.graph["top_artists_by_song_count"] = sorted(
            [(d["name"], d["song_count"], d["avg_popularity"]) for _, d in artist_nodes],
            key=lambda x: x[1],
            reverse=True,
        )[:50]

        # --- song leaderboards ---
        song_nodes = [
            (nid, d)
            for nid, d in self.G.nodes(data=True)
            if d.get("node_type") == "song"
        ]

        self.G.graph["top_songs_by_popularity"] = sorted(
            [
                (d["song_name"], d.get("popularity", 0), d["song_id"])
                for _, d in song_nodes
            ],
            key=lambda x: x[1],
            reverse=True,
        )[:50]

        # --- genre leaderboards ---
        genre_nodes = [
            (nid, d)
            for nid, d in self.G.nodes(data=True)
            if d.get("node_type") == "genre"
        ]

        self.G.graph["top_genres_by_song_count"] = sorted(
            [(d["name"], d["song_count"]) for _, d in genre_nodes],
            key=lambda x: x[1],
            reverse=True,
        )[:50]

        # --- language distribution ---
        lang_nodes = [
            (nid, d)
            for nid, d in self.G.nodes(data=True)
            if d.get("node_type") == "language"
        ]
        self.G.graph["language_distribution"] = sorted(
            [(d["code"], d["song_count"]) for _, d in lang_nodes],
            key=lambda x: x[1],
            reverse=True,
        )

        # --- global stats ---
        total = len(songs)
        self.G.graph["global_stats"] = {
            "total_songs": total,
            "total_artists": len(artist_nodes),
            "total_genres": len(genre_nodes),
            "total_languages": len(lang_nodes),
            "avg_popularity": sum(s.get("popularity", 0) or 0 for s in songs) / total,
            "avg_tempo": sum(s.get("tempo", 0) or 0 for s in songs) / total,
            "avg_energy": sum(s.get("energy", 0) or 0 for s in songs) / total,
            "avg_danceability": sum(s.get("danceability", 0) or 0 for s in songs) / total,
            "avg_valence": sum(s.get("valence", 0) or 0 for s in songs) / total,
        }

        # --- artist name lookup (lowercase → original) for fast entity recognition ---
        self.G.graph["artist_lookup"] = {
            d["name"].lower(): d["name"]
            for _, d in artist_nodes
        }

        # --- genre name lookup ---
        self.G.graph["genre_lookup"] = {
            d["name"].lower(): d["name"]
            for _, d in genre_nodes
        }

        # --- language lookup ---
        lang_name_map = {
            "en": "English", "pt": "Portuguese", "es": "Spanish",
            "ko": "Korean", "fr": "French", "ja": "Japanese",
            "de": "German", "pl": "Polish", "it": "Italian",
            "sv": "Swedish", "ru": "Russian", "id": "Indonesian",
            "tr": "Turkish", "fi": "Finnish", "nl": "Dutch",
            "zh": "Chinese", "ar": "Arabic", "hi": "Hindi",
            "th": "Thai", "vi": "Vietnamese", "INTRUMENTAL": "Instrumental",
        }
        self.G.graph["language_name_map"] = lang_name_map
        self.G.graph["language_code_lookup"] = {
            name.lower(): code for code, name in lang_name_map.items()
        }
        for code in [d["code"] for _, d in lang_nodes]:
            self.G.graph["language_code_lookup"][code.lower()] = code

        logger.info("Pre-computed aggregations stored on graph")


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build Music4All Knowledge Graph")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild even if pickle exists")
    args = parser.parse_args()

    settings = get_settings()
    pkl = Path(settings.kg_pickle_path)

    if args.rebuild or not pkl.exists():
        builder = KnowledgeGraphBuilder()
        builder.build()
        builder.save()
    else:
        print(f"Pickle already exists at {pkl}. Use --rebuild to recreate.")
