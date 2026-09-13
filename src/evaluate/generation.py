"""Generate answers from persisted, complete page representations."""

import asyncio
import base64
import logging
from collections.abc import Awaitable, Callable
from time import perf_counter
from uuid import UUID

from openai import AsyncOpenAI
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_fixed

from src.config import GenerationConfig
from src.stor_obj import ObjectStore
from src.stor_rel.crud import (
    find_records,
    finish_run,
    get_record,
    start_run,
    update_record,
    upsert_record,
    validate_run,
)
from src.stor_rel.schema import (
    Asset,
    Generation,
    GenerationContext,
    GenerationRun,
    PageRepresentation,
    Query,
    Retrieval,
    RetrievalHit,
    RunItem,
)
from src.utils.progress import progress_bar

logger = logging.getLogger(__name__)
ContextRenderer = Callable[[PageRepresentation, ObjectStore], Awaitable[list[dict]]]
CONTEXT_RENDERERS: dict[str, ContextRenderer] = {}


def register_context_renderer(kind: str, renderer: ContextRenderer):
    """Register a renderer for a future whole-page representation."""
    if kind in CONTEXT_RENDERERS or kind in {"ocr", "original_image", "image"}:
        raise ValueError(f"Context renderer {kind!r} is already registered")
    CONTEXT_RENDERERS[kind] = renderer


async def render_page(
    representation: PageRepresentation, store: ObjectStore
) -> list[dict]:
    if representation.kind in CONTEXT_RENDERERS:
        return await CONTEXT_RENDERERS[representation.kind](representation, store)
    if representation.text is not None:
        return [{"type": "text", "text": representation.text}]
    if representation.asset_id:
        asset = await get_record(Asset, representation.asset_id)
        if not asset.mime_type.startswith("image/"):
            raise ValueError(f"No renderer for representation {representation.kind!r}")
        data = await store.read_asset(asset)
        return [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{asset.mime_type};base64,{base64.b64encode(data).decode()}"
                },
            }
        ]
    raise ValueError(f"Representation {representation.id} has no content")


async def run_generation(
    retrieval_run_id: UUID,
    config: GenerationConfig,
    resume_run_id: UUID | None = None,
    *,
    object_store: ObjectStore | None = None,
    completion=None,
) -> UUID:
    source_run = await validate_run(retrieval_run_id, kind="retrieval")
    if config.context.page_top_k > source_run.config.get("page_top_k", 20):
        raise ValueError("Generation page_top_k exceeds saved retrieval depth")
    inputs = {"retrieval": retrieval_run_id}
    if config.context.representation_run_id:
        await validate_run(
            config.context.representation_run_id, dataset_id=source_run.dataset_id
        )
        inputs["representations"] = config.context.representation_run_id
    kinds = [
        "original_image" if k == "image" else k for k in config.context.representations
    ]
    if len(set(kinds)) != len(kinds):
        raise ValueError("Context representations must be distinct")
    if (
        any(k != "original_image" for k in kinds)
        and not config.context.representation_run_id
    ):
        raise ValueError("Non-image context requires an explicit representation_run_id")
    retrievals = await find_records(Retrieval, run_id=retrieval_run_id)
    run = await start_run(
        source_run.dataset_id,
        "generation",
        config.model_dump(mode="json"),
        inputs=inputs,
        selection=sorted(str(r.query_id) for r in retrievals),
        resume_run_id=resume_run_id,
    )
    if run.status == "completed":
        return run.id
    await upsert_record(
        GenerationRun,
        {"id": run.id},
        {"dataset_id": run.dataset_id, "retrieval_run_id": retrieval_run_id},
    )
    store = object_store or ObjectStore()
    client = None
    if completion is None:
        endpoint = config.endpoint
        client = AsyncOpenAI(
            base_url=endpoint.base_url,
            api_key=endpoint.resolve_api_key(),
            timeout=endpoint.timeout_seconds,
            max_retries=0,
        )

        @retry(
            stop=stop_after_attempt(endpoint.max_retries),
            wait=wait_fixed(endpoint.retry_wait_seconds),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        async def completion(messages):
            return await client.chat.completions.create(
                model=endpoint.model,
                messages=messages,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                extra_body=endpoint.extra_body or None,
            )

    semaphore = asyncio.Semaphore(config.endpoint.concurrency)

    async def process(retrieval):
        async with semaphore:
            prior = await find_records(
                Generation, run_id=run.id, query_id=retrieval.query_id
            )
            if prior and prior[0].status == "completed":
                await upsert_record(
                    RunItem,
                    {"run_id": run.id, "query_id": retrieval.query_id},
                    {
                        "dataset_id": run.dataset_id,
                        "status": "completed",
                        "error": None,
                    },
                )
                return True
            generation = await upsert_record(
                Generation,
                {"run_id": run.id, "query_id": retrieval.query_id},
                {
                    "dataset_id": run.dataset_id,
                    "retrieval_id": retrieval.id,
                    "status": "running",
                    "error": None,
                },
            )
            item = await upsert_record(
                RunItem,
                {"run_id": run.id, "query_id": retrieval.query_id},
                {"dataset_id": run.dataset_id, "status": "running"},
            )
            await update_record(RunItem, item.id, attempts=item.attempts + 1)
            started = perf_counter()
            try:
                if retrieval.status != "completed":
                    raise ValueError("Upstream retrieval did not complete")
                query = await get_record(Query, retrieval.query_id)
                hits = sorted(
                    await find_records(RetrievalHit, retrieval_id=retrieval.id),
                    key=lambda hit: hit.rank,
                )[: config.context.page_top_k]
                content = [{"type": "text", "text": f"Question: {query.query}"}]
                manifest = []
                text_chars = 0
                image_bytes = 0
                position = 0
                for hit in hits:
                    for kind in kinds:
                        filters = {
                            "dataset_id": run.dataset_id,
                            "corpus_id": hit.corpus_id,
                            "kind": kind,
                        }
                        if kind != "original_image":
                            filters["run_id"] = config.context.representation_run_id
                        reps = await find_records(PageRepresentation, **filters)
                        if len(reps) != 1:
                            raise ValueError(
                                f"Expected one {kind} representation for page {hit.corpus_id}; found {len(reps)}"
                            )
                        rep = reps[0]
                        if rep.text is not None:
                            text_chars += len(rep.text)
                        if rep.asset_id:
                            asset = await get_record(Asset, rep.asset_id)
                            image_bytes += asset.size_bytes
                        if (
                            text_chars > config.context.max_text_chars
                            or image_bytes > config.context.max_image_bytes
                        ):
                            raise ValueError(
                                "Whole-page context exceeds configured budget; no pages were truncated"
                            )
                        content.append(
                            {
                                "type": "text",
                                "text": f"Page {hit.rank} ({hit.corpus_id}), representation {kind}:",
                            }
                        )
                        content.extend(await render_page(rep, store))
                        await upsert_record(
                            GenerationContext,
                            {"generation_id": generation.id, "position": position},
                            {
                                "dataset_id": run.dataset_id,
                                "hit_id": hit.id,
                                "representation_id": rep.id,
                                "kind": rep.kind,
                                "text": rep.text,
                                "asset_id": rep.asset_id,
                            },
                        )
                        manifest.append(
                            {
                                "page_id": str(hit.corpus_id),
                                "representation_id": str(rep.id),
                                "asset_id": str(rep.asset_id) if rep.asset_id else None,
                                "position": position,
                            }
                        )
                        position += 1
                messages = [
                    {"role": "system", "content": config.prompt},
                    {"role": "user", "content": content},
                ]
                request = {
                    "system": config.prompt,
                    "question": query.query,
                    "contexts": manifest,
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens,
                }
                await update_record(Generation, generation.id, request=request)
                result = await completion(messages)
                response = result.choices[0].message.content
                if response is None:
                    raise ValueError("Generation returned no text")
                usage = result.usage.model_dump(mode="json") if result.usage else {}
                await update_record(
                    Generation,
                    generation.id,
                    response=response,
                    status="completed",
                    error=None,
                    usage=usage,
                    latency_ms=(perf_counter() - started) * 1000,
                )
                await update_record(RunItem, item.id, status="completed", error=None)
                return True
            except Exception as exc:
                await update_record(
                    Generation,
                    generation.id,
                    status="failed",
                    error=str(exc),
                    latency_ms=(perf_counter() - started) * 1000,
                )
                await update_record(RunItem, item.id, status="failed", error=str(exc))
                return False

    try:

        async def tracked_process(retrieval):
            try:
                return await process(retrieval)
            finally:
                progress.update()

        with progress_bar(
            total=len(retrievals), desc="Generating answers", unit="query"
        ) as progress:
            outcomes = await asyncio.gather(
                *(tracked_process(row) for row in retrievals)
            )
        completed = sum(outcomes)
        await finish_run(
            run.id,
            status="completed" if completed == len(outcomes) else "partial",
            expected_count=len(outcomes),
            completed_count=completed,
            failed_count=len(outcomes) - completed,
        )
    except Exception as exc:
        await finish_run(run.id, status="failed", error=str(exc))
        raise
    finally:
        if client:
            await client.close()
    return run.id
