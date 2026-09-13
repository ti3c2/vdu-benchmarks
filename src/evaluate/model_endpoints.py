"""Preflight checks for OpenAI-compatible model endpoints."""

import logging
import math
from collections.abc import Callable
from typing import Any

from openai import AsyncOpenAI
from openai.types import CreateEmbeddingResponse
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_fixed

from src.config import DenseConfig, EndpointProfile

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


async def verify_embedding_endpoint(
    config: DenseConfig,
    *,
    purpose: str = "embedding",
    client_factory: Callable[..., Any] = AsyncOpenAI,
) -> None:
    """Send a tiny /embeddings request because providers may omit embeddings from /models."""
    profile = config.endpoint
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
    async def request_embedding():
        sample = "preflight embedding check"
        if config.adapter == "vllm":
            body = dict(profile.extra_body)
            body.update(
                model=profile.model,
                encoding_format="float",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": config.query_instruction + sample}
                        ],
                    }
                ],
            )
            if config.dimensions is not None:
                body["dimensions"] = config.dimensions
            return await client.post(
                "/embeddings", cast_to=CreateEmbeddingResponse, body=body
            )
        kwargs = {
            "model": profile.model,
            "input": [config.query_instruction + sample],
            "encoding_format": "float",
        }
        if config.dimensions is not None:
            kwargs["dimensions"] = config.dimensions
        if profile.extra_body:
            kwargs["extra_body"] = profile.extra_body
        return await client.embeddings.create(**kwargs)

    try:
        try:
            response = await request_embedding()
        except Exception as exc:
            raise ValueError(
                f"{purpose} endpoint {profile.base_url} did not return a test "
                f"embedding for model {profile.model!r}"
            ) from exc
        data = getattr(response, "data", None) or []
        embedding = getattr(data[0], "embedding", None) if data else None
        if not embedding or any(
            not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in embedding
        ):
            raise ValueError(
                f"{purpose} endpoint {profile.base_url} returned an invalid test "
                f"embedding for model {profile.model!r}"
            )
        if config.dimensions is not None and len(embedding) != config.dimensions:
            raise ValueError(
                f"{purpose} endpoint {profile.base_url} returned "
                f"{len(embedding)} dimensions for model {profile.model!r}; expected "
                f"{config.dimensions}"
            )
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await result
