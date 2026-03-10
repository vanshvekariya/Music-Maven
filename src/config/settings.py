"""Application settings and configuration management"""

from pathlib import Path
from typing import Optional
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Qdrant Configuration
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "music4all"  # legacy single-collection name
    lyric_collection_name: str = "music4all_lyrics"
    tag_collection_name: str = "music4all_tags"

    # SQL Database Configuration
    sql_db_path: str = "music4all.db"
    sql_table_name: str = "songs"
    
    # OpenAI/LLM Configuration
    openai_api_key: Optional[str] = None
    openai_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "openai/gpt-oss-20b:free"
    llm_temperature: float = 0.0
    openai_embedding_model: str = "text-embedding-3-small"
    
    # Local Embeddings Configuration
    use_local_embeddings: bool = True
    local_embedding_model: str = "all-MiniLM-L6-v2"          # tags collection
    lyric_embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"  # lyrics collection
    
    # Data Configuration
    data_dir: Path = Path("./data")
    raw_data_dir: Path = Path("./data/raw/dataset")
    processed_data_dir: Path = Path("./data/processed")
    
    # Application Configuration
    log_level: str = "INFO"
    batch_size: int = 100
    
    # Vector Configuration
    vector_size: int = 384  # for all-MiniLM-L6-v2
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
