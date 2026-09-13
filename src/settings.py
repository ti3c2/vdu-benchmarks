import logging
import sys
from functools import lru_cache
from pathlib import Path

from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    path_root: Path = Path(__file__).parents[1]
    path_data: Path = path_root / "data"
    path_data_raw: Path = path_data / "raw"
    path_data_processed: Path = path_data / "processed"
    path_data_interim: Path = path_data / "interim"

    sql_database_url: str = "postgresql+asyncpg://vdu:vdu@localhost:5432/vdu"
    sql_pool_size: int = 10
    sql_max_overflow: int = 20
    sql_pool_timeout: int = 30
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "vdu-admin"
    minio_secret_key: str = "vdu-local-secret"
    minio_secure: bool = False
    minio_bucket: str = "vdu-benchmarks"

    log_level: int = logging.INFO

    model_config = SettingsConfigDict(
        env_file=path_root / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings():
    settings = Settings()
    logger.remove()
    logger.add(sys.stderr, level=settings.log_level)
    return settings
