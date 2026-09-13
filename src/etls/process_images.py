"""Persist page OCR independently so expensive preprocessing can be reused."""

import asyncio
import base64
import logging
import time
from functools import partial
from typing import Any, Awaitable, Callable
from uuid import UUID

from openai import AsyncOpenAI
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_fixed

from src.config import PreprocessConfig
from src.stor_obj import ObjectStore
from src.stor_rel import crud
from src.stor_rel.schema import Asset, Corpus, Dataset, PageRepresentation, RunItem

logger = logging.getLogger(__name__)


async def process_image(
    image_bytes: bytes,
    mime_type: str,
    config: PreprocessConfig,
    *,
    client: AsyncOpenAI | None = None,
) -> dict[str, Any]:
    """Run one OCR request, with explicit MIME and independent retry delay."""
    profile = config.endpoint
    owns_client = client is None
    client = client or AsyncOpenAI(
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
    async def request():
        return await client.chat.completions.create(
            model=profile.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
                            },
                        },
                        {"type": "text", "text": config.prompt},
                    ],
                }
            ],
            extra_body=profile.extra_body,
            timeout=profile.timeout_seconds,
        )

    started = time.perf_counter()
    try:
        response = await request()
        text = response.choices[0].message.content
        if not isinstance(text, str) or not text.strip():
            raise ValueError("OCR endpoint returned no page text")
        return {
            "text": text,
            "metadata": {
                "response_id": response.id,
                "model": response.model,
                "finish_reason": response.choices[0].finish_reason,
                "usage": response.usage.model_dump(mode="json")
                if response.usage
                else None,
                "latency_seconds": time.perf_counter() - started,
            },
        }
    finally:
        if owns_client:
            await client.close()


async def preprocess_pages(
    dataset_id: UUID,
    config: PreprocessConfig,
    resume_run_id: UUID | None = None,
    *,
    object_store: ObjectStore | None = None,
    ocr_processor: Callable[..., Awaitable[dict[str, Any]]] | None = None,
) -> UUID:
    dataset = await crud.get_record(Dataset, dataset_id)
    if dataset is None or dataset.status != "completed":
        raise ValueError("OCR requires a completed dataset ingestion")
    pages = sorted(
        await crud.find_records(Corpus, dataset_id=dataset_id),
        key=lambda page: page.original_id,
    )
    if config.page_limit is not None:
        pages = pages[: config.page_limit]
    run = await crud.start_run(
        dataset_id,
        "preprocess",
        config.model_dump(mode="json"),
        selection=[str(page.id) for page in pages],
        resume_run_id=resume_run_id,
    )
    if run.status == "completed":
        return run.id
    store = object_store or ObjectStore()
    client = None
    if ocr_processor is None:
        profile = config.endpoint
        client = AsyncOpenAI(
            base_url=profile.base_url,
            api_key=profile.resolve_api_key(),
            timeout=profile.timeout_seconds,
            max_retries=0,
        )
        ocr_processor = partial(process_image, client=client)
    items = {
        item.corpus_id: item for item in await crud.find_records(RunItem, run_id=run.id)
    }
    semaphore = asyncio.Semaphore(config.endpoint.concurrency)

    async def process_page(page):
        previous = items.get(page.id)
        if previous and previous.status == "completed":
            return True
        async with semaphore:
            item = await crud.upsert_record(
                RunItem,
                {"run_id": run.id, "corpus_id": page.id},
                {
                    "dataset_id": dataset_id,
                    "status": "running",
                    "attempts": (previous.attempts if previous else 0) + 1,
                    "error": None,
                },
            )
            try:
                asset = await crud.get_record(Asset, page.asset_id)
                if asset is None:
                    raise ValueError(f"Page {page.id} has no original image asset")
                image_bytes = await store.read_asset(asset)
                result = await ocr_processor(image_bytes, asset.mime_type, config)
                text = result["text"]
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("OCR endpoint returned no page text")
                await crud.upsert_record(
                    PageRepresentation,
                    {
                        "run_id": run.id,
                        "corpus_id": page.id,
                        "kind": "ocr",
                    },
                    {
                        "dataset_id": dataset_id,
                        "text": text,
                        "metadata_json": result.get("metadata", {}),
                    },
                )
                await crud.update_record(
                    RunItem, item.id, status="completed", error=None
                )
                return True
            except Exception as error:
                await crud.update_record(
                    RunItem, item.id, status="failed", error=str(error)
                )
                logger.warning("OCR failed for corpus %s: %s", page.id, error)
                return False

    try:
        outcomes = await asyncio.gather(*(process_page(page) for page in pages))
        completed = sum(outcomes)
        await crud.finish_run(
            run.id,
            status="completed"
            if completed == len(pages)
            else "partial"
            if completed
            else "failed",
            expected_count=len(pages),
            completed_count=completed,
            failed_count=len(pages) - completed,
        )
    except BaseException:
        await crud.finish_run(run.id, status="failed")
        raise
    finally:
        if client is not None:
            await client.close()
    return run.id
