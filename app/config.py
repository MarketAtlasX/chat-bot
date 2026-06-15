from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    llm_provider: str = "ollama"
    llm_model: str = "qwen2.5:7b"
    ollama_base_url: str = "http://localhost:11434"

    postgres_dsn: str = "postgresql+asyncpg://marketatlas:marketatlas@localhost:5432/marketatlas"

    neo4j_uri: Optional[str] = "bolt://localhost:7687"
    neo4j_user: Optional[str] = "neo4j"
    neo4j_password: Optional[str] = "test"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "marketatlas_events"

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
