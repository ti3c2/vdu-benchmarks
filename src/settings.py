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

    openai_vlm_preprocess_api_key: str = ""
    openai_vlm_preprocess_api_base: str = "https://api.openai.com/v1"
    openai_vlm_preprocess_model: str = "deepseek-ai/DeepSeek-OCR-2"

    openai_emb_api_key: str = ""
    openai_emb_api_base: str = "https://api.openai.com/v1"
    openai_emb_model: str = "text-embedding-3-small"
    openai_emb_batch_size: int = 256
    openai_emb_num_workers: int = 3

    openai_max_retries: int = 3
    openai_timeout: int = 600 * 1000
    log_chat_completion_input: bool = False
    openai_chat_completion_max_concurrency: int = 64

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
