"""Configuration settings for Chiller RAG system."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    host: str = os.getenv("CHILLER_RAG_HOST", "127.0.0.1")
    port: int = int(os.getenv("CHILLER_RAG_PORT", "8001"))
    data_dir: Path = Path(os.getenv("CHILLER_RAG_DATA_DIR", "./data"))
    chunk_size: int = int(os.getenv("CHILLER_RAG_CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHILLER_RAG_CHUNK_OVERLAP", "80"))
    similarity_threshold: float = float(os.getenv("CHILLER_RAG_SIMILARITY_THRESHOLD", "0.20"))
    top_k: int = int(os.getenv("CHILLER_RAG_TOP_K", "4"))
    embedding_provider: str = os.getenv("CHILLER_RAG_EMBEDDING_PROVIDER", "local")
    llm_provider: str = os.getenv("CHILLER_RAG_LLM_PROVIDER", "grounded_rule")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")


settings = Settings()
