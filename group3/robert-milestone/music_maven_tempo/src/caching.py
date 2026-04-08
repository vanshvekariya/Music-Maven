from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .audio_io import metadata_signature, safe_track_id


class TrackCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_path(self, audio_path: Path) -> Path:
        return self.cache_dir / f"{safe_track_id(audio_path)}.json"

    def load(self, audio_path: Path) -> Optional[Dict[str, Any]]:
        path = self.cache_path(audio_path)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        current_signature = metadata_signature(audio_path)
        if payload.get("metadata_signature") != current_signature:
            return None
        return payload.get("result")

    def save(self, audio_path: Path, result: Dict[str, Any]) -> None:
        path = self.cache_path(audio_path)
        payload = {
            "metadata_signature": metadata_signature(audio_path),
            "result": result,
        }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
