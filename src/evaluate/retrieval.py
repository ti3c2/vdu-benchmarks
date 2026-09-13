"""Retrieval uses stored query vectors and saves unique corpus-page rankings."""

import logging
import math
import time
from uuid import UUID

from src.config import RetrievalConfig
from src.evaluate.vector_store import create_vector_store
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
    EmbeddingRun,
    Retrieval,
    RetrievalHit,
    RetrievalRun,
    RunItem,
    StageRun,
)
from src.stor_rel.vector_queries import persist_ranking
from src.utils.progress import progress_bar

logger = logging.getLogger(__name__)


def validate_profiles(query: EmbeddingRun, corpus: EmbeddingRun, mode: str):
    if query.dataset_id != corpus.dataset_id:
        raise ValueError("Query and corpus embedding runs belong to different datasets")
    if query.role != "query" or corpus.role != "corpus":
        raise ValueError("Expected query and corpus embedding runs, respectively")
    if query.unit_kind != "query" or corpus.unit_kind not in {"chunk", "page"}:
        raise ValueError("Unsupported embedding point granularity")
    if mode in {"dense", "hybrid"}:
        qdense = query.profiles.get("dense")
        cdense = corpus.profiles.get("dense")
        if not qdense or not cdense:
            raise ValueError("Dense retrieval requires dense vectors in both runs")
        for field in ("space_id", "dimensions", "distance"):
            if qdense.get(field) != cdense.get(field) or qdense.get(field) is None:
                raise ValueError(f"Query and corpus dense {field} must match")
    if mode in {"sparse", "hybrid"}:
        qsparse = query.profiles.get("sparse")
        csparse = corpus.profiles.get("sparse")
        if not qsparse or qsparse != csparse:
            raise ValueError("Query and corpus sparse profiles must match")


def rank_page_groups(groups, dataset_id: UUID) -> list[dict]:
    """Check server output and make returned score ties deterministic."""
    hits = []
    seen = set()
    for group in groups:
        if not group.hits:
            continue
        point = group.hits[0]
        payload = point.payload or {}
        if payload.get("dataset_id") != str(dataset_id):
            raise ValueError("Retrieved point does not belong to the expected dataset")
        corpus_id = UUID(payload["corpus_id"])
        if str(group.id) != str(corpus_id) or corpus_id in seen:
            raise ValueError("Qdrant returned invalid or repeated corpus-page groups")
        if not math.isfinite(point.score):
            raise ValueError("Qdrant returned a non-finite score")
        seen.add(corpus_id)
        hits.append(
            {
                "corpus_id": corpus_id,
                "point_id": UUID(str(point.id)),
                "chunk_id": UUID(payload["chunk_id"])
                if payload.get("chunk_id")
                else None,
                "score": float(point.score),
            }
        )
    hits.sort(key=lambda hit: (-hit["score"], str(hit["corpus_id"])))
    for rank, hit in enumerate(hits, start=1):
        hit["rank"] = rank
    return hits


async def run_retrieval(
    query_embedding_run_id: UUID,
    corpus_embedding_run_id: UUID,
    config: RetrievalConfig,
    resume_run_id: UUID | None = None,
) -> UUID:
    await validate_run(query_embedding_run_id, kind="embed_queries")
    await validate_run(corpus_embedding_run_id)
    query_run = await get_record(EmbeddingRun, query_embedding_run_id)
    corpus_run = await get_record(EmbeddingRun, corpus_embedding_run_id)
    validate_profiles(query_run, corpus_run, config.mode)
    items = await find_records(
        RunItem, run_id=query_embedding_run_id, status="completed"
    )
    query_ids = sorted([item.query_id for item in items], key=str)
    if (
        not query_ids
        or any(query_id is None for query_id in query_ids)
        or len(query_ids) != query_run.point_count
    ):
        raise ValueError("Query embedding run has incomplete relational coverage")
    query_store = create_vector_store(query_run.collection_name)
    corpus_store = create_vector_store(corpus_run.collection_name)
    run = None
    completed = 0
    failed = 0
    shortfalls = 0
    try:
        for store, embedding in ((query_store, query_run), (corpus_store, corpus_run)):
            if not await store.collection_exists():
                raise ValueError(
                    f"Embedding collection is missing: {store.collection_name}"
                )
            if await store.point_count() != embedding.point_count:
                raise ValueError(
                    f"Embedding collection count changed: {store.collection_name}"
                )
            dense = embedding.profiles.get("dense")
            await store.ensure_collection(
                dimensions=dense["dimensions"] if dense else None,
                sparse=embedding.profiles.get("sparse") is not None,
                role=embedding.role,
                distance=dense["distance"] if dense else "cosine",
            )
        prefetch = min(
            corpus_run.point_count,
            config.prefetch_limit or max(100, 10 * config.page_top_k),
        )
        version = await corpus_store.server_version()
        run = await start_run(
            query_run.dataset_id,
            "retrieval",
            config.model_dump(mode="json"),
            inputs={
                "queries": query_embedding_run_id,
                "corpus": corpus_embedding_run_id,
            },
            selection=[str(query_id) for query_id in query_ids],
            resume_run_id=resume_run_id,
        )
        if run.status == "completed":
            return run.id
        await upsert_record(
            RetrievalRun,
            {"id": run.id},
            {
                "dataset_id": run.dataset_id,
                "query_embedding_run_id": query_embedding_run_id,
                "corpus_embedding_run_id": corpus_embedding_run_id,
            },
        )
        await update_record(
            StageRun,
            run.id,
            metadata_json={
                "qdrant_version": version,
                "effective_prefetch_limit": prefetch,
                "fusion": "rrf" if config.mode == "hybrid" else None,
                "rrf_k": 2 if config.mode == "hybrid" else None,
                "group_by": "corpus_id",
                "group_size": 1,
            },
        )
        with progress_bar(
            total=len(query_ids), desc="Retrieving pages", unit="query"
        ) as progress:
            for offset in range(0, len(query_ids), 64):
                batch = query_ids[offset : offset + 64]
                vectors = await query_store.load_vectors(batch)
                for query_id in batch:
                    previous = await find_records(
                        Retrieval, run_id=run.id, query_id=query_id
                    )
                    existing = previous[0] if previous else None
                    if existing and existing.status == "completed":
                        completed += 1
                        previous_hits = await find_records(
                            RetrievalHit, retrieval_id=existing.id
                        )
                        if len(previous_hits) < config.page_top_k:
                            shortfalls += 1
                        progress.update()
                        continue
                    retrieval = await upsert_record(
                        Retrieval,
                        {"run_id": run.id, "query_id": query_id},
                        {
                            "dataset_id": run.dataset_id,
                            "status": "running",
                            "error": None,
                        },
                    )
                    item = await upsert_record(
                        RunItem,
                        {"run_id": run.id, "query_id": query_id},
                        {
                            "dataset_id": run.dataset_id,
                            "status": "running",
                            "error": None,
                        },
                    )
                    await update_record(RunItem, item.id, attempts=item.attempts + 1)
                    started = time.perf_counter()
                    try:
                        saved = vectors.get(str(query_id))
                        if saved is None:
                            raise ValueError(
                                f"Persisted query vectors are missing for {query_id}"
                            )
                        groups = await corpus_store.retrieve_pages(
                            saved,
                            mode=config.mode,
                            page_top_k=config.page_top_k,
                            prefetch_limit=prefetch,
                        )
                        hits = rank_page_groups(groups, run.dataset_id)
                        if len(hits) < config.page_top_k:
                            shortfalls += 1
                        await persist_ranking(
                            run.dataset_id,
                            retrieval.id,
                            item.id,
                            hits,
                            (time.perf_counter() - started) * 1000,
                        )
                        completed += 1
                    except Exception as error:
                        logger.exception("Retrieval failed for query %s", query_id)
                        await update_record(
                            Retrieval,
                            retrieval.id,
                            status="failed",
                            error=str(error),
                            latency_ms=(time.perf_counter() - started) * 1000,
                        )
                        await update_record(
                            RunItem, item.id, status="failed", error=str(error)
                        )
                        failed += 1
                    finally:
                        progress.update()
        record = await get_record(StageRun, run.id)
        await update_record(
            StageRun,
            run.id,
            metadata_json={
                **record.metadata_json,
                "queries_with_result_shortfall": shortfalls,
            },
        )
        await finish_run(
            run.id,
            status="completed"
            if not failed
            else ("partial" if completed else "failed"),
            expected_count=len(query_ids),
            completed_count=completed,
            failed_count=failed,
        )
    except Exception:
        if run is not None:
            await finish_run(
                run.id,
                status="failed",
                expected_count=len(query_ids),
                completed_count=completed,
                failed_count=len(query_ids) - completed,
            )
        raise
    finally:
        await query_store.close()
        await corpus_store.close()
    return run.id
