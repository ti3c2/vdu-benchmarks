"""Persist query, chunk, and page vectors as independently reusable stages."""

import asyncio
import base64
import logging
from uuid import UUID

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode

from src.config import EmbeddingConfig
from src.evaluate.embeddings import BM25Encoder, create_dense_model, encode_dense
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
    Asset,
    Chunk,
    Corpus,
    Dataset,
    EmbeddingRun,
    PageRepresentation,
    Query,
    RunItem,
)
from src.utils.progress import progress_bar

logger = logging.getLogger(__name__)


async def vectorize_queries(
    dataset_id: UUID,
    config: EmbeddingConfig,
    query_ids: list[UUID] | None = None,
    resume_run_id: UUID | None = None,
) -> UUID:
    """Encode every selected query once, including normalized rephrases."""
    dataset = await get_record(Dataset, dataset_id)
    if dataset.status != "completed":
        raise ValueError("Dataset ingestion must be completed before vectorization")
    records = await find_records(Query, dataset_id=dataset_id)
    if query_ids is not None:
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("Query selection contains duplicates")
        selected = set(query_ids)
        records = [query for query in records if query.id in selected]
    rows = [
        {"id": query.id, "text": query.query}
        for query in sorted(records, key=lambda query: query.id)
    ]
    if query_ids is not None and len(rows) != len(query_ids):
        raise ValueError("Selected query is absent from this dataset")
    return await _vectorize(dataset_id, "query", rows, config, {}, resume_run_id)


async def vectorize_chunks(
    chunk_run_id: UUID, config: EmbeddingConfig, resume_run_id: UUID | None = None
) -> UUID:
    """Encode persisted chunks; dense and sparse representations share point IDs."""
    source = await validate_run(chunk_run_id, kind="chunks")
    if config.dense and config.dense.modality != "text":
        raise ValueError(
            "Chunk vectors require text embeddings; use vectorize_pages for images"
        )
    records = await find_records(Chunk, run_id=chunk_run_id)
    rows = [
        {
            "id": item.id,
            "text": item.text,
            "corpus_id": item.corpus_id,
            "representation_id": item.representation_id,
        }
        for item in sorted(records, key=lambda item: item.id)
    ]
    return await _vectorize(
        source.dataset_id,
        "chunk",
        rows,
        config,
        {"chunks": chunk_run_id},
        resume_run_id,
    )


async def vectorize_pages(
    dataset_id: UUID,
    config: EmbeddingConfig,
    representation_run_id: UUID | None = None,
    resume_run_id: UUID | None = None,
) -> UUID:
    """Encode original images and/or full-page text on one point per page."""
    dataset = await get_record(Dataset, dataset_id)
    if dataset.status != "completed":
        raise ValueError("Dataset ingestion must be completed before vectorization")
    needs_text = config.sparse is not None or (
        config.dense is not None and config.dense.modality == "text"
    )
    if needs_text and representation_run_id is None:
        raise ValueError("Page text embeddings require a representation_run_id")
    if representation_run_id:
        await validate_run(representation_run_id, dataset_id=dataset_id)
    pages = await find_records(Corpus, dataset_id=dataset_id)
    representations = {}
    if representation_run_id:
        records = await find_records(
            PageRepresentation, run_id=representation_run_id, dataset_id=dataset_id
        )
        for representation in records:
            if representation.text is not None:
                if representation.corpus_id in representations:
                    raise ValueError(
                        "Representation run has multiple text representations for a page"
                    )
                representations[representation.corpus_id] = representation
    rows = []
    for page in sorted(pages, key=lambda page: page.id):
        representation = representations.get(page.id)
        if needs_text and representation is None:
            raise ValueError(
                f"Page {page.id} has no full-page text in representation run"
            )
        rows.append(
            {
                "id": page.id,
                "corpus_id": page.id,
                "asset_id": page.asset_id,
                "text": representation.text if representation else "",
                "representation_id": representation.id if representation else None,
            }
        )
    return await _vectorize(
        dataset_id,
        "page",
        rows,
        config,
        {"representations": representation_run_id} if representation_run_id else {},
        resume_run_id,
    )


async def _vectorize(
    dataset_id: UUID,
    unit: str,
    rows: list[dict],
    config: EmbeddingConfig,
    inputs: dict[str, UUID],
    resume_run_id: UUID | None,
) -> UUID:
    """Own the shared run/checkpoint boundary for the three public stages."""
    if not rows:
        raise ValueError("Vectorization selection is empty")
    role = "query" if unit == "query" else "corpus"
    kind = {"query": "embed_queries", "chunk": "embed_chunks", "page": "embed_pages"}[
        unit
    ]
    run = await start_run(
        dataset_id,
        kind,
        config.model_dump(mode="json"),
        inputs=inputs,
        selection=[str(row["id"]) for row in rows],
        resume_run_id=resume_run_id,
    )
    if run.status == "completed":
        return run.id
    collection = f"vdu_{role}_{run.id.hex}"
    embedding_run = await upsert_record(
        EmbeddingRun,
        {"id": run.id},
        {
            "dataset_id": dataset_id,
            "role": role,
            "unit_kind": unit,
            "collection_name": collection,
            "profiles": config.model_dump(mode="json"),
            "point_count": 0,
        },
    )
    store = create_vector_store(collection)
    dense_model = None
    vector_index = None
    sparse_model = None
    image_store = None
    dimensions = config.dense.dimensions if config.dense else None
    completed = 0
    failed = 0
    try:
        # A resumed collection can establish inferred dimensions without a new model call.
        exists = await store.collection_exists()
        if exists and config.dense and dimensions is None:
            dimensions = await store.dense_dimensions()
        if exists and config.dense:
            profiles = config.model_dump(mode="json")
            profiles["dense"]["dimensions"] = dimensions
            await update_record(EmbeddingRun, embedding_run.id, profiles=profiles)
        if exists:
            await store.ensure_collection(
                dimensions=dimensions,
                sparse=config.sparse is not None,
                role=role,
                distance=config.dense.distance if config.dense else "cosine",
            )
        labels = {"query": "queries", "chunk": "chunks", "page": "pages"}
        with progress_bar(
            total=len(rows), desc=f"Embedding {labels[unit]}", unit=unit
        ) as progress:
            for offset in range(0, len(rows), config.batch_size):
                batch = rows[offset : offset + config.batch_size]
                persisted = (
                    await store.load_vectors([row["id"] for row in batch])
                    if exists
                    else {}
                )
                pending = []
                for row in batch:
                    keys = {
                        "run_id": run.id,
                        {"query": "query_id", "chunk": "chunk_id", "page": "corpus_id"}[
                            unit
                        ]: row["id"],
                    }
                    if str(row["id"]) in persisted:
                        await upsert_record(
                            RunItem,
                            keys,
                            {
                                "dataset_id": dataset_id,
                                "status": "completed",
                                "error": None,
                            },
                        )
                        completed += 1
                        progress.update()
                    else:
                        item = await upsert_record(
                            RunItem,
                            keys,
                            {
                                "dataset_id": dataset_id,
                                "status": "running",
                                "error": None,
                            },
                        )
                        await update_record(
                            RunItem, item.id, attempts=item.attempts + 1
                        )
                        pending.append((row, item.id))
                if not pending:
                    continue
                try:
                    texts = [row["text"] for row, _ in pending]
                    dense_vectors = None
                    if config.dense:
                        if dense_model is None:
                            dense_model = create_dense_model(
                                config.dense, config.batch_size
                            )
                        values = texts
                        if role == "corpus" and config.dense.modality == "image":
                            if image_store is None:
                                from src.stor_obj import ObjectStore

                                image_store = ObjectStore()
                            values = []
                            for row, _ in pending:
                                asset = await get_record(Asset, row["asset_id"])
                                data = await image_store.read_asset(asset)
                                values.append(
                                    f"data:{asset.mime_type};base64,{base64.b64encode(data).decode('ascii')}"
                                )
                        dense_vectors = await encode_dense(
                            dense_model, config.dense, values, role
                        )
                        observed = len(dense_vectors[0])
                        if dimensions is not None and dimensions != observed:
                            raise ValueError("Dense dimensions changed between batches")
                        dimensions = observed
                    sparse_vectors = None
                    if config.sparse:
                        if sparse_model is None:
                            sparse_model = await asyncio.to_thread(
                                BM25Encoder, config.sparse
                            )
                        sparse_vectors = await asyncio.to_thread(
                            sparse_model.encode, texts, role
                        )
                    if not exists:
                        await store.ensure_collection(
                            dimensions=dimensions,
                            sparse=config.sparse is not None,
                            role=role,
                            distance=config.dense.distance
                            if config.dense
                            else "cosine",
                        )
                        exists = True
                    profiles = config.model_dump(mode="json")
                    if config.dense:
                        profiles["dense"]["dimensions"] = dimensions
                    await update_record(
                        EmbeddingRun, embedding_run.id, profiles=profiles
                    )
                    nodes = []
                    for index, (row, _) in enumerate(pending):
                        metadata = {
                            "dataset_id": str(dataset_id),
                            "source_run_id": str(run.id),
                            "unit_kind": unit,
                        }
                        if role == "corpus":
                            metadata["corpus_id"] = str(row["corpus_id"])
                        if unit == "chunk":
                            metadata["chunk_id"] = str(row["id"])
                        if row.get("representation_id"):
                            metadata["representation_id"] = str(
                                row["representation_id"]
                            )
                        nodes.append(
                            TextNode(
                                id_=str(row["id"]),
                                text=row["text"],
                                metadata=metadata,
                                embedding=dense_vectors[index]
                                if dense_vectors
                                else None,
                                excluded_embed_metadata_keys=list(metadata),
                                excluded_llm_metadata_keys=list(metadata),
                            )
                        )
                    if sparse_vectors is None:
                        if vector_index is None:
                            vector_index = VectorStoreIndex(
                                nodes=[],
                                storage_context=StorageContext.from_defaults(
                                    vector_store=store
                                ),
                                embed_model=dense_model,
                            )
                        # LlamaIndex retains embeddings already attached to these nodes.
                        await vector_index.ainsert_nodes(nodes)
                    else:
                        await store.async_add(nodes, sparse_vectors=sparse_vectors)
                    # The Qdrant upsert is durable before the relational checkpoint.
                    for _, item_id in pending:
                        await update_record(
                            RunItem, item_id, status="completed", error=None
                        )
                    completed += len(pending)
                    progress.update(len(pending))
                except Exception as error:
                    logger.exception("Embedding batch failed in run %s", run.id)
                    for _, item_id in pending:
                        await update_record(
                            RunItem, item_id, status="failed", error=str(error)
                        )
                    failed += len(pending)
                    progress.update(len(pending))
        point_count = await store.point_count() if exists else 0
        if point_count != completed:
            raise ValueError(
                "Collection point count does not match completed source items"
            )
        await update_record(EmbeddingRun, embedding_run.id, point_count=point_count)
        status = "completed" if not failed else ("partial" if completed else "failed")
        await finish_run(
            run.id,
            status=status,
            expected_count=len(rows),
            completed_count=completed,
            failed_count=failed,
        )
    except Exception:
        await finish_run(
            run.id,
            status="failed",
            expected_count=len(rows),
            completed_count=completed,
            failed_count=len(rows) - completed,
        )
        raise
    finally:
        if hasattr(dense_model, "aclose"):
            await dense_model.aclose()
        await store.close()
    return run.id
