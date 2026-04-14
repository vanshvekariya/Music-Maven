"""In-memory session summaries for multi-turn chat (single API process).

Stores one rolling text summary per session (LLM-compressed), not full
transcripts—keeps injected context small and avoids repeating long answers.
"""

from __future__ import annotations

from threading import Lock
from typing import Dict, Optional

from loguru import logger

_MAX_SESSIONS = 5000


class ConversationStore:
    """Thread-safe session_id → rolling summary string."""

    def __init__(self) -> None:
        self._summaries: Dict[str, str] = {}
        self._lock = Lock()

    def get_summary(self, session_id: str) -> Optional[str]:
        if not session_id or not session_id.strip():
            return None
        sid = session_id.strip()
        with self._lock:
            s = self._summaries.get(sid)
        return s if s else None

    def set_summary(self, session_id: str, text: str, max_stored: int) -> None:
        if not session_id or not session_id.strip():
            return
        sid = session_id.strip()
        t = (text or "").strip()
        if max_stored > 0 and len(t) > max_stored:
            t = t[: max_stored - 3] + "..."
        if not t:
            return
        with self._lock:
            if sid not in self._summaries and len(self._summaries) >= _MAX_SESSIONS:
                drop = next(iter(self._summaries))
                del self._summaries[drop]
                logger.warning(
                    f"ConversationStore: evicted oldest session {drop!r} (limit {_MAX_SESSIONS})"
                )
            self._summaries[sid] = t

    def clear(self, session_id: str) -> None:
        sid = (session_id or "").strip()
        if not sid:
            return
        with self._lock:
            self._summaries.pop(sid, None)


_store: Optional[ConversationStore] = None


def get_conversation_store() -> ConversationStore:
    global _store
    if _store is None:
        _store = ConversationStore()
    return _store
