"""Preflight checks for OpenAI-compatible model endpoints."""

import logging
from collections.abc import Callable
from typing import Any

from openai import AsyncOpenAI
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_fixed

from src.config import EndpointProfile

logger = logging.getLogger(__name__)


async def verify_model_endpoint(
    profile: EndpointProfile,
    *,
    purpose: str,
    client_factory: Callable[..., Any] = AsyncOpenAI,
) -> None:
    """Fetch /models and require the configured model before starting work."""
    client = client_factory(
        base_url=profile.base_url,
        api_key=profile.resolve_api_key(),
        timeout=profile.timeout_seconds,
        max_retries=0,
    )

    @retry(
        stop=stop_after_attempt(profile.max_retries),
        wait=wait_fixed(profile.retry_wait_seconds),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def list_models():
        return await client.models.list()

    try:
        try:
            response = await list_models()
        except Exception as exc:
            raise ValueError(
                f"{purpose} endpoint {profile.base_url} did not return /models"
            ) from exc
        models = sorted(
            {
                model_id
                for model in getattr(response, "data", [])
                if isinstance(model_id := getattr(model, "id", None), str)
            }
        )
        if profile.model not in models:
            preview = ", ".join(models[:20])
            suffix = f": {preview}" if preview else ""
            if len(models) > 20:
                suffix += f", ... ({len(models)} total)"
            raise ValueError(
                f"{purpose} endpoint {profile.base_url} does not list required model "
                f"{profile.model!r}{suffix}"
            )
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await result
