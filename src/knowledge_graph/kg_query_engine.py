"""
KG Query Engine for Music Maven.

Answers factual / analytical queries directly from the pre-built
NetworkX knowledge graph with zero LLM calls.  Each "template" is a
regex pattern coupled with a resolver that traverses the graph and
returns a formatted markdown answer.
"""

import re
from typing import Dict, Any, List, Optional, Tuple

import networkx as nx
from loguru import logger


# ── helpers ──────────────────────────────────────────────────────────────────

_NUM = r"(\d+)"
_FEATURE_NAMES = {
    "popularity": "popularity",
    "popular": "popularity",
    "energy": "energy",
    "energetic": "energy",
    "danceability": "danceability",
    "danceable": "danceability",
    "valence": "valence",
    "happy": "valence",
    "positive": "valence",
    "tempo": "tempo",
    "bpm": "tempo",
    "fast": "tempo",
    "slow": "tempo",
}

_COMPARATORS = {
    "above": "gt",
    "over": "gt",
    "greater than": "gt",
    "more than": "gt",
    "higher than": "gt",
    ">": "gt",
    "below": "lt",
    "under": "lt",
    "less than": "lt",
    "lower than": "lt",
    "<": "lt",
    "between": "between",
}


def _pct(v: float, is_01: bool = True) -> str:
    if is_01 and 0 <= v <= 1:
        return f"{v * 100:.0f}%"
    return f"{v:.1f}"


def _fmt_pop(v: float) -> str:
    return f"{v:.0f}/100"


class KGQueryEngine:
    """Resolve structured queries against the knowledge graph."""

    def __init__(self, graph: nx.DiGraph):
        self.G = graph
        self._artist_lookup: Dict[str, str] = graph.graph.get("artist_lookup", {})
        self._genre_lookup: Dict[str, str] = graph.graph.get("genre_lookup", {})
        self._lang_code_lookup: Dict[str, str] = graph.graph.get("language_code_lookup", {})
        self._lang_name_map: Dict[str, str] = graph.graph.get("language_name_map", {})

        self._templates: List[Tuple[re.Pattern, str, callable]] = self._build_templates()

    # ── public API ───────────────────────────────────────────────────────

    def try_answer(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to answer *query* from the KG.

        Returns a dict ``{"answer": str, "template": str}`` on success,
        or ``None`` if no template matched.
        """
        q = query.strip()
        q_lower = q.lower()

        for pattern, template_name, resolver in self._templates:
            m = pattern.search(q_lower)
            if m:
                try:
                    answer = resolver(q, q_lower, m)
                    if answer:
                        logger.info(f"KG answered via template '{template_name}'")
                        return {"answer": answer, "template": template_name}
                except Exception as e:
                    logger.warning(f"KG template '{template_name}' failed: {e}")
        return None

    # ── template registry ────────────────────────────────────────────────

    def _build_templates(self):
        """Return list of (compiled_pattern, name, resolver_fn)."""
        return [
            # --- top N artists ---
            (
                re.compile(
                    r"(?:top|best|most popular)\s*(\d+)?\s*(?:most\s+popular\s+)?artists?"
                    r"(?:\s+by\s+(\w+))?",
                ),
                "top_n_artists",
                self._resolve_top_artists,
            ),
            # "who are the most popular artists" / "most popular artists"
            (
                re.compile(r"(?:who\s+are\s+)?(?:the\s+)?most\s+popular\s+artists?"),
                "top_n_artists_alt",
                self._resolve_top_artists,
            ),
            # --- top N songs ---
            (
                re.compile(
                    r"(?:top|best|most popular)\s*(\d+)?\s*(?:most\s+popular\s+)?songs?"
                    r"(?:\s+by\s+(\w+))?",
                ),
                "top_n_songs",
                self._resolve_top_songs,
            ),
            # --- top genres ---
            (
                re.compile(r"(?:top|most common|most popular|best)\s*(\d+)?\s*genres?"),
                "top_n_genres",
                self._resolve_top_genres,
            ),
            # "what are the most common genres"
            (
                re.compile(r"(?:what\s+are\s+)?(?:the\s+)?most\s+common\s+genres?"),
                "top_n_genres_alt",
                self._resolve_top_genres,
            ),
            # --- how many songs in <language> (must be before generic count) ---
            (
                re.compile(r"(?:how\s+many|number\s+of|count(?:\s+of)?)\s+songs?\s+(?:in|are\s+in)\s+(\w+)"),
                "songs_in_language",
                self._resolve_songs_in_language,
            ),
            # --- how many / count ---
            (
                re.compile(
                    r"(?:how\s+many|total(?:\s+number\s+of)?|count(?:\s+of)?)\s+"
                    r"(songs?|artists?|genres?|languages?)"
                ),
                "count_entities",
                self._resolve_count,
            ),
            # --- language distribution ---
            (
                re.compile(r"(?:language|lang)\s*distribution|songs?\s+(?:per|by|in\s+each)\s+language"),
                "language_distribution",
                self._resolve_language_distribution,
            ),
            # --- genre distribution ---
            (
                re.compile(r"genre\s*distribution|songs?\s+(?:per|by|in\s+each)\s+genre"),
                "genre_distribution",
                self._resolve_genre_distribution,
            ),
            # --- artist info ---
            (
                re.compile(r"(?:tell\s+me\s+about|info(?:rmation)?\s+(?:about|on|for)|stats?\s+(?:for|of|about))\s+(.+)"),
                "artist_info",
                self._resolve_artist_info,
            ),
            # --- average <feature> of <genre/artist> (must be before songs_by_artist) ---
            (
                re.compile(r"(?:average|avg|mean)\s+(\w+)\s+(?:of|for|in)\s+(.+?)(?:\s+songs?)?$"),
                "average_feature",
                self._resolve_average_feature,
            ),
            # --- songs by artist (skip if query asks for aggregates) ---
            (
                re.compile(r"(?!.*\b(?:average|avg|mean|total|count|sum|how many|number of)\b)songs?\s+(?:by|from|of)\s+(.+?)(?:\s+sorted|\s+ordered|\s+by\s+popularity)?$"),
                "songs_by_artist",
                self._resolve_songs_by_artist,
            ),
            # --- which artists have the most songs ---
            (
                re.compile(r"which\s+artists?\s+(?:has|have)\s+(?:the\s+)?most\s+songs?"),
                "artists_by_song_count",
                self._resolve_artists_by_song_count,
            ),
            # --- songs with <feature> above/below N ---
            (
                re.compile(
                    r"songs?\s+with\s+(\w+)\s+(above|below|over|under|greater\s+than|less\s+than|higher\s+than|lower\s+than)\s+(\d+(?:\.\d+)?)"
                ),
                "songs_with_filter",
                self._resolve_songs_with_filter,
            ),
            # --- songs with <feature> between X and Y ---
            (
                re.compile(
                    r"songs?\s+with\s+(\w+)\s+between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)"
                ),
                "songs_with_range",
                self._resolve_songs_with_range,
            ),
            # --- compare artist A and artist B ---
            (
                re.compile(r"compare\s+(.+?)\s+(?:and|vs\.?|versus)\s+(.+)"),
                "compare_artists",
                self._resolve_compare_artists,
            ),
            # --- what genres does <artist> sing ---
            (
                re.compile(r"what\s+genres?\s+does\s+(.+?)\s+(?:sing|play|make|do|perform|have)"),
                "artist_genres",
                self._resolve_artist_genres,
            ),
            # --- genres related to X ---
            (
                re.compile(r"(?:genres?\s+)?(?:related|similar)\s+(?:to|genres?)\s+(.+)"),
                "related_genres",
                self._resolve_related_genres,
            ),
            # --- popularity distribution ---
            (
                re.compile(r"popularity\s*distribution"),
                "popularity_distribution",
                self._resolve_popularity_distribution,
            ),
            # --- global stats ---
            (
                re.compile(r"(?:global|overall|dataset|general)\s*(?:stats?|statistics|summary|overview)"),
                "global_stats",
                self._resolve_global_stats,
            ),
        ]

    # ── resolvers ────────────────────────────────────────────────────────

    def _resolve_top_artists(self, q: str, ql: str, m: re.Match) -> str:
        n = int(m.group(1)) if m.lastindex and m.group(1) else 10
        sort_by = (m.group(2) or "popularity").lower() if m.lastindex and m.lastindex >= 2 and m.group(2) else "popularity"

        if sort_by in ("songs", "song_count", "count"):
            data = self.G.graph["top_artists_by_song_count"][:n]
            lines = [f"Here are the top {n} artists by song count:\n"]
            for i, (name, cnt, avg_pop) in enumerate(data, 1):
                lines.append(f"{i}. **{name}** -- {cnt} songs, avg popularity: {_fmt_pop(avg_pop)}")
        else:
            data = self.G.graph["top_artists_by_popularity"][:n]
            lines = [f"Here are the top {n} artists by popularity:\n"]
            for i, (name, avg_pop, cnt) in enumerate(data, 1):
                lines.append(f"{i}. **{name}** -- popularity: {_fmt_pop(avg_pop)}, {cnt} song(s)")
        return "\n".join(lines)

    def _resolve_top_songs(self, q: str, ql: str, m: re.Match) -> str:
        n = int(m.group(1)) if m.lastindex and m.group(1) else 10
        data = self.G.graph["top_songs_by_popularity"][:n]
        lines = [f"Here are the top {n} songs by popularity:\n"]
        for i, (name, pop, sid) in enumerate(data, 1):
            node = self.G.nodes.get(f"song:{sid}", {})
            artist = ""
            for _, tgt, d in self.G.edges(f"song:{sid}", data=True):
                if d.get("rel") == "PERFORMED_BY":
                    artist = self.G.nodes[tgt].get("name", "")
                    break
            artist_str = f" by **{artist}**" if artist else ""
            lines.append(f"{i}. **{name}**{artist_str} -- popularity: {_fmt_pop(pop)}")
        return "\n".join(lines)

    def _resolve_top_genres(self, q: str, ql: str, m: re.Match) -> str:
        n = int(m.group(1)) if m.lastindex and m.group(1) else 10
        data = self.G.graph["top_genres_by_song_count"][:n]
        lines = [f"Here are the top {n} genres by song count:\n"]
        for i, (name, cnt) in enumerate(data, 1):
            lines.append(f"{i}. **{name}** -- {cnt:,} songs")
        return "\n".join(lines)

    def _resolve_count(self, q: str, ql: str, m: re.Match) -> str:
        entity = m.group(1).lower().rstrip("s")
        stats = self.G.graph["global_stats"]
        key_map = {
            "song": "total_songs",
            "artist": "total_artists",
            "genre": "total_genres",
            "language": "total_languages",
        }
        key = key_map.get(entity)
        if not key:
            return None
        val = stats[key]
        return f"There are **{val:,}** {entity}s in the Music4All dataset."

    def _resolve_language_distribution(self, q: str, ql: str, m: re.Match) -> str:
        data = self.G.graph["language_distribution"]
        lines = ["## Language Distribution\n"]
        for code, count in data:
            name = self._lang_name_map.get(code, code)
            lines.append(f"- **{name}** ({code}): {count:,} songs")
        return "\n".join(lines)

    def _resolve_songs_in_language(self, q: str, ql: str, m: re.Match) -> str:
        lang_input = m.group(1).strip().lower()
        code = self._lang_code_lookup.get(lang_input, lang_input)
        lid = f"lang:{code}"
        if lid not in self.G:
            return None
        count = self.G.nodes[lid].get("song_count", 0)
        name = self._lang_name_map.get(code, code)
        return f"There are **{count:,}** songs in **{name}** ({code}) in the dataset."

    def _resolve_genre_distribution(self, q: str, ql: str, m: re.Match) -> str:
        data = self.G.graph["top_genres_by_song_count"][:20]
        lines = ["## Genre Distribution (top 20)\n"]
        for name, count in data:
            lines.append(f"- **{name}**: {count:,} songs")
        return "\n".join(lines)

    def _resolve_artist_info(self, q: str, ql: str, m: re.Match) -> str:
        raw = m.group(1).strip().rstrip("?. ")
        artist = self._find_artist(raw)
        if not artist:
            return None
        aid = f"artist:{artist}"
        d = self.G.nodes[aid]
        lines = [
            f"## Artist: {artist}\n",
            f"- **Songs**: {d['song_count']}",
            f"- **Avg popularity**: {_fmt_pop(d['avg_popularity'])}",
            f"- **Avg energy**: {_pct(d['avg_energy'])}",
            f"- **Avg danceability**: {_pct(d['avg_danceability'])}",
            f"- **Avg valence**: {_pct(d['avg_valence'])}",
            f"- **Avg tempo**: {d['avg_tempo']:.0f} BPM",
            f"- **Top genres**: {', '.join(d['top_genres'])}",
            f"- **Languages**: {', '.join(d['languages'])}",
        ]
        return "\n".join(lines)

    def _resolve_songs_by_artist(self, q: str, ql: str, m: re.Match) -> str:
        raw = m.group(1).strip().rstrip("?. ")
        artist = self._find_artist(raw)
        if not artist:
            return None
        aid = f"artist:{artist}"
        song_ids = [
            src for src, tgt, d in self.G.in_edges(aid, data=True)
            if d.get("rel") == "PERFORMED_BY"
        ]
        songs = []
        for sid in song_ids:
            nd = self.G.nodes.get(sid, {})
            if nd.get("node_type") == "song":
                songs.append(nd)

        songs.sort(key=lambda s: s.get("popularity", 0), reverse=True)
        top = songs[:15]

        lines = [f"Here are the top songs by **{artist}** (by popularity):\n"]
        for i, s in enumerate(top, 1):
            lines.append(
                f"{i}. **{s['song_name']}** -- popularity: {_fmt_pop(s.get('popularity', 0))}, "
                f"energy: {_pct(s.get('energy', 0))}, tempo: {s.get('tempo', 0):.0f} BPM"
            )
        if len(songs) > 15:
            lines.append(f"\n*...and {len(songs) - 15} more songs*")
        return "\n".join(lines)

    def _resolve_artists_by_song_count(self, q: str, ql: str, m: re.Match) -> str:
        data = self.G.graph["top_artists_by_song_count"][:10]
        lines = ["Here are the artists with the most songs:\n"]
        for i, (name, cnt, avg_pop) in enumerate(data, 1):
            lines.append(f"{i}. **{name}** -- {cnt} songs, avg popularity: {_fmt_pop(avg_pop)}")
        return "\n".join(lines)

    def _resolve_average_feature(self, q: str, ql: str, m: re.Match) -> str:
        feature_raw = m.group(1).strip().lower()
        target_raw = m.group(2).strip().rstrip("?. ")

        feature = _FEATURE_NAMES.get(feature_raw)
        if not feature:
            return None

        # check artist
        artist = self._find_artist(target_raw)
        if artist:
            aid = f"artist:{artist}"
            d = self.G.nodes[aid]
            val = d.get(f"avg_{feature}", d.get(feature))
            if val is None:
                return None
            fmt = _pct(val) if feature not in ("popularity", "tempo") else (
                _fmt_pop(val) if feature == "popularity" else f"{val:.0f} BPM"
            )
            return f"The average {feature} of **{artist}** is **{fmt}**."

        # check genre
        genre = self._find_genre(target_raw)
        if genre:
            gid = f"genre:{genre}"
            d = self.G.nodes[gid]
            val = d.get(f"avg_{feature}")
            if val is None:
                return None
            fmt = _pct(val) if feature not in ("popularity", "tempo") else (
                _fmt_pop(val) if feature == "popularity" else f"{val:.0f} BPM"
            )
            return f"The average {feature} of **{genre}** songs is **{fmt}** (across {d['song_count']:,} songs)."

        return None

    def _resolve_songs_with_filter(self, q: str, ql: str, m: re.Match) -> str:
        feature_raw = m.group(1).lower()
        comparator_raw = m.group(2).lower()
        threshold = float(m.group(3))

        feature = _FEATURE_NAMES.get(feature_raw)
        if not feature:
            return None
        comp = _COMPARATORS.get(comparator_raw, "gt")

        songs = self._filter_songs(feature, comp, threshold)
        songs.sort(key=lambda s: s.get("popularity", 0), reverse=True)
        top = songs[:15]

        if not top:
            return f"No songs found with {feature} {'above' if comp == 'gt' else 'below'} {threshold}."

        label = "above" if comp == "gt" else "below"
        lines = [f"Here are songs with {feature} {label} {threshold} (sorted by popularity):\n"]
        for i, s in enumerate(top, 1):
            lines.append(
                f"{i}. **{s['song_name']}** by **{s.get('_artist', 'Unknown')}** -- "
                f"{feature}: {s.get(feature, 0):.2f}, popularity: {_fmt_pop(s.get('popularity', 0))}"
            )
        if len(songs) > 15:
            lines.append(f"\n*{len(songs):,} total songs match this filter*")
        return "\n".join(lines)

    def _resolve_songs_with_range(self, q: str, ql: str, m: re.Match) -> str:
        feature_raw = m.group(1).lower()
        lo = float(m.group(2))
        hi = float(m.group(3))

        feature = _FEATURE_NAMES.get(feature_raw)
        if not feature:
            return None

        songs = self._filter_songs_range(feature, lo, hi)
        songs.sort(key=lambda s: s.get("popularity", 0), reverse=True)
        top = songs[:15]

        if not top:
            return f"No songs found with {feature} between {lo} and {hi}."

        lines = [f"Here are songs with {feature} between {lo} and {hi} (sorted by popularity):\n"]
        for i, s in enumerate(top, 1):
            lines.append(
                f"{i}. **{s['song_name']}** by **{s.get('_artist', 'Unknown')}** -- "
                f"{feature}: {s.get(feature, 0):.2f}, popularity: {_fmt_pop(s.get('popularity', 0))}"
            )
        if len(songs) > 15:
            lines.append(f"\n*{len(songs):,} total songs match this filter*")
        return "\n".join(lines)

    def _resolve_compare_artists(self, q: str, ql: str, m: re.Match) -> str:
        raw_a = m.group(1).strip().rstrip("?. ")
        raw_b = m.group(2).strip().rstrip("?. ")

        artist_a = self._find_artist(raw_a)
        artist_b = self._find_artist(raw_b)
        if not artist_a or not artist_b:
            return None

        da = self.G.nodes[f"artist:{artist_a}"]
        db = self.G.nodes[f"artist:{artist_b}"]

        lines = [
            f"## Comparison: {artist_a} vs {artist_b}\n",
            f"| Metric | **{artist_a}** | **{artist_b}** |",
            f"|--------|{'---' * 5}|{'---' * 5}|",
            f"| Songs | {da['song_count']} | {db['song_count']} |",
            f"| Avg Popularity | {_fmt_pop(da['avg_popularity'])} | {_fmt_pop(db['avg_popularity'])} |",
            f"| Avg Energy | {_pct(da['avg_energy'])} | {_pct(db['avg_energy'])} |",
            f"| Avg Danceability | {_pct(da['avg_danceability'])} | {_pct(db['avg_danceability'])} |",
            f"| Avg Valence | {_pct(da['avg_valence'])} | {_pct(db['avg_valence'])} |",
            f"| Avg Tempo | {da['avg_tempo']:.0f} BPM | {db['avg_tempo']:.0f} BPM |",
            f"| Top Genres | {', '.join(da['top_genres'][:3])} | {', '.join(db['top_genres'][:3])} |",
        ]
        return "\n".join(lines)

    def _resolve_artist_genres(self, q: str, ql: str, m: re.Match) -> str:
        raw = m.group(1).strip().rstrip("?. ")
        artist = self._find_artist(raw)
        if not artist:
            return None
        aid = f"artist:{artist}"
        genre_edges = [
            (tgt, d.get("weight", 0))
            for _, tgt, d in self.G.edges(aid, data=True)
            if d.get("rel") == "WORKS_IN_GENRE"
        ]
        genre_edges.sort(key=lambda x: x[1], reverse=True)

        lines = [f"**{artist}** works in these genres:\n"]
        for gid, weight in genre_edges[:15]:
            gname = self.G.nodes[gid].get("name", gid)
            lines.append(f"- **{gname}** ({weight} songs)")
        return "\n".join(lines)

    def _resolve_related_genres(self, q: str, ql: str, m: re.Match) -> str:
        raw = m.group(1).strip().rstrip("?. ")
        genre = self._find_genre(raw)
        if not genre:
            return None
        gid = f"genre:{genre}"
        related = [
            (tgt, d.get("weight", 0))
            for _, tgt, d in self.G.edges(gid, data=True)
            if d.get("rel") == "RELATED_TO"
        ]
        related.sort(key=lambda x: x[1], reverse=True)

        if not related:
            return f"No related genres found for **{genre}**."

        lines = [f"Genres related to **{genre}** (by co-occurrence):\n"]
        for rid, weight in related[:15]:
            rname = self.G.nodes[rid].get("name", rid)
            lines.append(f"- **{rname}** (co-occurs in {weight} songs)")
        return "\n".join(lines)

    def _resolve_popularity_distribution(self, q: str, ql: str, m: re.Match) -> str:
        tiers = [
            (nid, d)
            for nid, d in self.G.nodes(data=True)
            if d.get("node_type") == "popularity_tier"
        ]
        lines = ["## Popularity Distribution\n"]
        for nid, d in sorted(tiers, key=lambda x: x[1].get("lo", 0)):
            count = self.G.in_degree(nid)
            lines.append(f"- **{d['label']}**: {count:,} songs")
        return "\n".join(lines)

    def _resolve_global_stats(self, q: str, ql: str, m: re.Match) -> str:
        s = self.G.graph["global_stats"]
        lines = [
            "## Music4All Dataset Overview\n",
            f"- **Total songs**: {s['total_songs']:,}",
            f"- **Total artists**: {s['total_artists']:,}",
            f"- **Total genres**: {s['total_genres']:,}",
            f"- **Total languages**: {s['total_languages']}",
            f"- **Avg popularity**: {_fmt_pop(s['avg_popularity'])}",
            f"- **Avg energy**: {_pct(s['avg_energy'])}",
            f"- **Avg danceability**: {_pct(s['avg_danceability'])}",
            f"- **Avg valence**: {_pct(s['avg_valence'])}",
            f"- **Avg tempo**: {s['avg_tempo']:.0f} BPM",
        ]
        return "\n".join(lines)

    # ── entity recognition helpers ───────────────────────────────────────

    def _find_artist(self, text: str) -> Optional[str]:
        """Fuzzy-match text against known artist names."""
        t = text.lower().strip()
        if t in self._artist_lookup:
            return self._artist_lookup[t]
        # try removing common suffixes
        for suffix in ("'s", "'s", " songs", " music", " tracks"):
            cleaned = t.rstrip(suffix) if t.endswith(suffix) else t
            if cleaned in self._artist_lookup:
                return self._artist_lookup[cleaned]
        return None

    def _find_genre(self, text: str) -> Optional[str]:
        t = text.lower().strip()
        if t in self._genre_lookup:
            return self._genre_lookup[t]
        # try with hyphens
        t_hyp = t.replace(" ", "-")
        if t_hyp in self._genre_lookup:
            return self._genre_lookup[t_hyp]
        return None

    # ── filtering helpers ────────────────────────────────────────────────

    def _filter_songs(self, feature: str, comp: str, threshold: float) -> List[Dict]:
        results = []
        for nid, d in self.G.nodes(data=True):
            if d.get("node_type") != "song":
                continue
            val = d.get(feature)
            if val is None:
                continue
            if comp == "gt" and val > threshold:
                results.append(self._enrich_song(nid, d))
            elif comp == "lt" and val < threshold:
                results.append(self._enrich_song(nid, d))
        return results

    def _filter_songs_range(self, feature: str, lo: float, hi: float) -> List[Dict]:
        results = []
        for nid, d in self.G.nodes(data=True):
            if d.get("node_type") != "song":
                continue
            val = d.get(feature)
            if val is None:
                continue
            if lo <= val <= hi:
                results.append(self._enrich_song(nid, d))
        return results

    def _enrich_song(self, nid: str, d: dict) -> dict:
        """Add artist name to a song dict for display."""
        enriched = dict(d)
        for _, tgt, ed in self.G.edges(nid, data=True):
            if ed.get("rel") == "PERFORMED_BY":
                enriched["_artist"] = self.G.nodes[tgt].get("name", "Unknown")
                break
        return enriched
