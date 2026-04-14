"""Rolling conversation summary for multi-turn context (token-efficient).

After each reply, the LLM compresses prior summary + latest exchange into a
short brief: what was asked, what was shown, and what not to repeat—so
follow-ups like \"show me more\" can be grounded without re-injecting full
transcripts.
"""

from __future__ import annotations

from typing import Optional

from langchain_openai import ChatOpenAI
from loguru import logger

from ..config.settings import get_settings


def _fallback_summary(
    previous_summary: Optional[str],
    user_message: str,
    assistant_message: str,
    max_chars: int,
) -> str:
    """If the LLM fails, keep a crude compressed trace (still bounded)."""
    u = user_message.strip()[:500]
    a = assistant_message.strip()[:1200]
    block = f"User: {u}\nAssistant (excerpt): {a}"
    prev = (previous_summary or "").strip()
    if prev:
        merged = f"{prev}\n---\n{block}"
    else:
        merged = block
    if len(merged) > max_chars:
        return merged[: max_chars - 3] + "..."
    return merged


def update_rolling_summary(
    previous_summary: Optional[str],
    user_message: str,
    assistant_message: str,
    max_output_chars: int,
) -> tuple[str, str]:
    """
    Returns (new_summary, mode) where mode is 'llm' or 'fallback'.
    """
    settings = get_settings()
    um = user_message.strip()[:2500]
    am = assistant_message.strip()
    if len(am) > 4000:
        am = am[:4000] + "\n[...truncated before summarization...]"
    prev = (previous_summary or "").strip()
    if len(prev) > 3000:
        prev = prev[-3000:]

    prompt = f"""You maintain a compact ROLLING SUMMARY for a Music4All music-assistant chat.

Prior summary (may be empty):
{prev if prev else "(none)"}

Latest exchange:
User: {um}
Assistant: {am}

Write the UPDATED rolling summary for the next turn. It must capture:
- What the user asked for (filters, limits like top N, language, genre, mood)
- What the assistant actually returned (use counts and short pointers; do NOT paste long song lists—say e.g. "returned 10 songs" or "top 5 artists by popularity")
- What was already shown so follow-ups ("more", "same for Spanish", "exclude X") avoid repeating identical results unless the user explicitly wants a repeat

Rules:
- Plain text, short bullets or tight paragraphs
- Max {max_output_chars} characters
- No markdown headings, no SQL
- Output ONLY the new summary text"""

    try:
        llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        raw = llm.invoke(prompt)
        content = getattr(raw, "content", None) or str(raw)
        content = (content or "").strip()
        if not content:
            raise ValueError("empty summarizer output")
        if len(content) > max_output_chars:
            content = content[: max_output_chars - 3] + "..."
        return content, "llm"
    except Exception as e:
        logger.warning(f"Rolling summary LLM failed, using fallback: {e}")
        fb = _fallback_summary(previous_summary, user_message, assistant_message, max_output_chars)
        return fb, "fallback"
