from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import librosa
import numpy as np


def file_metadata(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "mtime": float(stat.st_mtime),
        "size": int(stat.st_size),
    }


def metadata_signature(path: Path) -> str:
    meta = file_metadata(path)
    raw = f"{meta['path']}|{meta['mtime']}|{meta['size']}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def safe_track_id(path: Path) -> str:
    return hashlib.md5(str(path.resolve()).encode("utf-8")).hexdigest()


def infer_genre(path: Path, dataset_root: Path | None = None) -> str:
    if dataset_root is not None:
        try:
            rel = path.resolve().relative_to(dataset_root.resolve())
            if len(rel.parts) >= 2:
                return rel.parts[0]
        except Exception:
            pass
    parent = path.parent.name.strip()
    return parent or "unknown"


def load_audio(path: Path, sample_rate: int = 22050, mono: bool = True) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(str(path), sr=sample_rate, mono=mono)
    if y.size == 0:
        raise ValueError(f"Empty audio buffer for {path}")
    if not np.isfinite(y).all():
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y.astype(np.float32), int(sr)


def duration_seconds(y: np.ndarray, sr: int) -> float:
    return float(len(y) / float(sr))
