"""Data processing module"""

__all__ = ["Music4AllProcessor", "EmbeddingPipeline"]


def __getattr__(name: str):
    if name == "Music4AllProcessor":
        from .music4all_processor import Music4AllProcessor
        return Music4AllProcessor
    if name == "EmbeddingPipeline":
        from .embedding_pipeline import EmbeddingPipeline
        return EmbeddingPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
