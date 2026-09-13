import asyncio
import os
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from llama_index.core.embeddings import MockEmbedding

from src.evaluate.retrieval import rank_page_groups, validate_profiles


def profiles(*, dataset_id=None, role="query", unit_kind="query", space_id="same"):
    return SimpleNamespace(
        dataset_id=dataset_id or UUID(int=1),
        role=role,
        unit_kind=unit_kind,
        profiles={
            "dense": {"space_id": space_id, "dimensions": 2, "distance": "cosine"},
            "sparse": {"model": "Qdrant/bm25", "language": "english"},
        },
    )


def test_profile_validation_accepts_text_queries_to_image_pages():
    query = profiles()
    corpus = profiles(role="corpus", unit_kind="page")
    query.profiles["dense"]["modality"] = "text"
    corpus.profiles["dense"]["modality"] = "image"
    validate_profiles(query, corpus, "hybrid")


@pytest.mark.parametrize(
    "change", ["dataset", "space", "dimensions", "sparse", "granularity"]
)
def test_profile_validation_rejects_incompatible_inputs(change):
    query = profiles()
    corpus = profiles(role="corpus", unit_kind="chunk")
    if change == "dataset":
        corpus.dataset_id = uuid4()
    elif change == "space":
        corpus.profiles["dense"]["space_id"] = "different"
    elif change == "dimensions":
        corpus.profiles["dense"]["dimensions"] = 3
    elif change == "sparse":
        corpus.profiles["sparse"]["language"] = "russian"
    else:
        corpus.unit_kind = "token"
    with pytest.raises(ValueError):
        validate_profiles(query, corpus, "hybrid")


def test_page_ranks_ties_and_duplicate_groups():
    dataset_id = uuid4()
    groups = [
        SimpleNamespace(
            id=str(UUID(int=number)),
            hits=[
                SimpleNamespace(
                    id=str(uuid4()),
                    score=0.5,
                    payload={
                        "dataset_id": str(dataset_id),
                        "corpus_id": str(UUID(int=number)),
                    },
                )
            ],
        )
        for number in (2, 1)
    ]
    hits = rank_page_groups(groups, dataset_id)
    assert [hit["corpus_id"].int for hit in hits] == [1, 2]
    assert [hit["rank"] for hit in hits] == [1, 2]
    assert rank_page_groups([], dataset_id) == []
    with pytest.raises(ValueError, match="repeated"):
        rank_page_groups([groups[0], groups[0]], dataset_id)
    with pytest.raises(ValueError, match="dataset"):
        rank_page_groups(groups, uuid4())


@pytest.mark.skipif(
    os.environ.get("VDU_INTEGRATION") != "1", reason="requires PostgreSQL and Qdrant"
)
@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_persisted_stages_reuse_query_vectors(monkeypatch, mode):
    from qdrant_client import AsyncQdrantClient
    from sqlalchemy import delete, select

    from src.config import (
        DenseConfig,
        EmbeddingConfig,
        EndpointProfile,
        RetrievalConfig,
        SparseConfig,
    )
    from src.evaluate import vectorize
    from src.evaluate.retrieval import run_retrieval
    from src.settings import get_settings
    from src.stor_rel.crud import (
        finish_run,
        get_record,
        save_record,
        start_run,
        update_record,
    )
    from src.stor_rel.entry import dispose_engine, get_db, get_engine
    from src.stor_rel.schema import (
        Asset,
        Base,
        Chunk,
        Corpus,
        Dataset,
        Doc,
        EmbeddingRun,
        PageRepresentation,
        Query,
        Retrieval,
        RetrievalHit,
        RunItem,
        StageRun,
    )

    calls = []

    async def fake_encode(model, config, values, role):
        calls.extend((role, value) for value in values)
        return [[1.0, 0.0] if "alpha" in value else [0.0, 1.0] for value in values]

    async def skip_preflight(*args, **kwargs):
        return None

    monkeypatch.setattr(
        vectorize, "create_dense_model", lambda *args: MockEmbedding(embed_dim=2)
    )
    monkeypatch.setattr(vectorize, "encode_dense", fake_encode)
    monkeypatch.setattr(vectorize, "verify_embedding_endpoint", skip_preflight)
    monkeypatch.setattr(
        "src.stor_rel.crud.implementation_provenance",
        lambda: {"implementation_sha256": "test-vectors"},
    )

    async def exercise():
        async with get_engine().begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        identity = uuid4().hex
        dataset = await save_record(
            Dataset,
            source="test-vectors",
            subset=identity,
            revision=identity,
            fingerprint=identity,
            status="completed",
        )
        collection_names = []
        try:
            document = await save_record(Doc, dataset_id=dataset.id, original_id="doc")
            asset = await save_record(
                Asset,
                dataset_id=dataset.id,
                bucket="test",
                object_key=identity,
                sha256="0" * 64,
                mime_type="image/png",
                size_bytes=0,
            )
            pages = [
                await save_record(
                    Corpus,
                    dataset_id=dataset.id,
                    doc_id=document.id,
                    asset_id=asset.id,
                    original_id=str(index),
                )
                for index in range(2)
            ]
            queries = [
                await save_record(
                    Query,
                    dataset_id=dataset.id,
                    original_id=str(index),
                    query=term,
                    language="en",
                )
                for index, term in enumerate(["alpha", "beta"])
            ]
            preprocessing = await start_run(dataset.id, "preprocess", {})
            reps = [
                await save_record(
                    PageRepresentation,
                    dataset_id=dataset.id,
                    run_id=preprocessing.id,
                    corpus_id=page.id,
                    kind="ocr",
                    text=term,
                )
                for page, term in zip(pages, ["alpha", "beta"])
            ]
            await finish_run(preprocessing.id, expected_count=2, completed_count=2)
            chunk_run = await start_run(
                dataset.id, "chunks", {}, inputs={"representations": preprocessing.id}
            )
            for rep in reps:
                await save_record(
                    Chunk,
                    dataset_id=dataset.id,
                    run_id=chunk_run.id,
                    representation_id=rep.id,
                    corpus_id=rep.corpus_id,
                    ordinal=0,
                    kind="prose",
                    text=rep.text,
                    content_hash="0" * 64,
                )
            await finish_run(chunk_run.id, expected_count=2, completed_count=2)
            config = EmbeddingConfig(
                dense=DenseConfig(
                    endpoint=EndpointProfile(model="fake"), space_id="fixture"
                )
                if mode != "sparse"
                else None,
                sparse=SparseConfig() if mode != "dense" else None,
            )
            query_run_id = await vectorize.vectorize_queries(dataset.id, config)
            corpus_run_id = await vectorize.vectorize_chunks(chunk_run.id, config)
            for run_id in (query_run_id, corpus_run_id):
                collection_names.append(
                    (await get_record(EmbeddingRun, run_id)).collection_name
                )
                assert (await get_record(StageRun, run_id)).status == "completed"
            assert len(calls) == (0 if mode == "sparse" else 4)
            retrieval_id = await run_retrieval(
                query_run_id, corpus_run_id, RetrievalConfig(page_top_k=2, mode=mode)
            )
            assert len(calls) == (0 if mode == "sparse" else 4), (
                "retrieval must not make embedding calls"
            )
            assert (await get_record(StageRun, retrieval_id)).status == "completed"
            assert await vectorize.vectorize_queries(dataset.id, config) == query_run_id
            assert (
                await run_retrieval(
                    query_run_id,
                    corpus_run_id,
                    RetrievalConfig(page_top_k=2, mode=mode),
                )
                == retrieval_id
            )
            assert len(calls) == (0 if mode == "sparse" else 4)
            # Recover a durable Qdrant write whose relational checkpoint was lost.
            await update_record(StageRun, query_run_id, status="partial")
            async with get_db() as db:
                await db.execute(delete(RunItem).where(RunItem.run_id == query_run_id))
            assert (
                await vectorize.vectorize_queries(
                    dataset.id, config, resume_run_id=query_run_id
                )
                == query_run_id
            )
            assert len(calls) == (0 if mode == "sparse" else 4)
            if config.dense:
                assert (await get_record(EmbeddingRun, query_run_id)).profiles["dense"][
                    "dimensions"
                ] == 2
            async with get_db() as db:
                for query, expected_page in zip(queries, pages):
                    hit = await db.scalar(
                        select(RetrievalHit)
                        .join(Retrieval, Retrieval.id == RetrievalHit.retrieval_id)
                        .where(
                            Retrieval.run_id == retrieval_id,
                            Retrieval.query_id == query.id,
                            RetrievalHit.rank == 1,
                        )
                    )
                    assert hit.corpus_id == expected_page.id
        finally:
            client = AsyncQdrantClient(url=get_settings().qdrant_url)
            async with get_db() as db:
                all_names = (
                    await db.scalars(
                        select(EmbeddingRun.collection_name).where(
                            EmbeddingRun.dataset_id == dataset.id
                        )
                    )
                ).all()
            for name in set(collection_names) | set(all_names):
                if await client.collection_exists(name):
                    await client.delete_collection(name)
            await client.close()
            async with get_db() as db:
                for table in reversed(Base.metadata.sorted_tables):
                    if "dataset_id" in table.c:
                        await db.execute(
                            delete(table).where(table.c.dataset_id == dataset.id)
                        )
                await db.execute(delete(Dataset).where(Dataset.id == dataset.id))
            await dispose_engine()

    asyncio.run(exercise())


@pytest.mark.skipif(
    os.environ.get("VDU_INTEGRATION") != "1",
    reason="requires PostgreSQL, MinIO, and Qdrant",
)
def test_image_page_hybrid_reads_minio_and_persists_page_points(monkeypatch):
    import base64
    import io
    from dataclasses import asdict

    from PIL import Image
    from qdrant_client import AsyncQdrantClient
    from sqlalchemy import delete, select

    from src.config import (
        DenseConfig,
        EmbeddingConfig,
        EndpointProfile,
        RetrievalConfig,
        SparseConfig,
    )
    from src.evaluate import vectorize
    from src.evaluate.retrieval import run_retrieval
    from src.settings import get_settings
    from src.stor_obj import ObjectStore
    from src.stor_rel.crud import finish_run, get_record, save_record, start_run
    from src.stor_rel.entry import dispose_engine, get_db, get_engine
    from src.stor_rel.schema import (
        Asset,
        Base,
        Corpus,
        Dataset,
        Doc,
        EmbeddingRun,
        PageRepresentation,
        Query,
        Retrieval,
        RetrievalHit,
        StageRun,
    )

    calls = []

    async def fake_encode(model, config, values, role):
        result = []
        for value in values:
            calls.append((role, value))
            if role == "query":
                red = "red" in value
            else:
                assert value.startswith("data:image/png;base64,")
                data = base64.b64decode(value.split(",", 1)[1])
                with Image.open(io.BytesIO(data)) as image:
                    red = image.getpixel((0, 0))[0] > image.getpixel((0, 0))[2]
            result.append([1.0, 0.0] if red else [0.0, 1.0])
        return result

    async def skip_preflight(*args, **kwargs):
        return None

    monkeypatch.setattr(
        vectorize, "create_dense_model", lambda *args: MockEmbedding(embed_dim=2)
    )
    monkeypatch.setattr(vectorize, "encode_dense", fake_encode)
    monkeypatch.setattr(vectorize, "verify_embedding_endpoint", skip_preflight)
    monkeypatch.setattr(
        "src.stor_rel.crud.implementation_provenance",
        lambda: {"implementation_sha256": "test-vectors"},
    )

    async def exercise():
        async with get_engine().begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        identity = uuid4().hex
        dataset = await save_record(
            Dataset,
            source="test-image-vectors",
            subset=identity,
            revision=identity,
            fingerprint=identity,
            status="completed",
        )
        object_store = ObjectStore(bucket=f"vdu-test-{identity}")
        objects = []
        try:
            document = await save_record(Doc, dataset_id=dataset.id, original_id="doc")
            preprocessing = await start_run(dataset.id, "preprocess", {})
            pages, queries = [], []
            for ordinal, color in enumerate(["red", "blue"]):
                image_bytes = io.BytesIO()
                Image.new("RGB", (2, 2), color=color).save(image_bytes, format="PNG")
                obj = await object_store.put_bytes(image_bytes.getvalue(), "image/png")
                objects.append(obj)
                asset = await save_record(Asset, dataset_id=dataset.id, **asdict(obj))
                page = await save_record(
                    Corpus,
                    dataset_id=dataset.id,
                    doc_id=document.id,
                    asset_id=asset.id,
                    original_id=str(ordinal),
                )
                pages.append(page)
                queries.append(
                    await save_record(
                        Query,
                        dataset_id=dataset.id,
                        original_id=str(ordinal),
                        query=f"{color} business report",
                        language="en",
                    )
                )
                await save_record(
                    PageRepresentation,
                    dataset_id=dataset.id,
                    run_id=preprocessing.id,
                    corpus_id=page.id,
                    kind="ocr",
                    text=f"{color} business report entire page",
                )
            await finish_run(preprocessing.id, expected_count=2, completed_count=2)
            config = EmbeddingConfig(
                dense=DenseConfig(
                    endpoint=EndpointProfile(model="fake-image"),
                    space_id="colors",
                    modality="image",
                    adapter="vllm",
                ),
                sparse=SparseConfig(),
            )
            query_id = await vectorize.vectorize_queries(dataset.id, config)
            corpus_id = await vectorize.vectorize_pages(
                dataset.id, config, representation_run_id=preprocessing.id
            )
            corpus_run = await get_record(EmbeddingRun, corpus_id)
            assert corpus_run.unit_kind == "page"
            assert corpus_run.point_count == 2
            assert (await get_record(StageRun, corpus_id)).status == "completed"
            retrieval_id = await run_retrieval(
                query_id, corpus_id, RetrievalConfig(mode="hybrid", page_top_k=2)
            )
            assert (await get_record(StageRun, retrieval_id)).status == "completed"
            assert len(calls) == 4
            async with get_db() as db:
                for page, query in zip(pages, queries):
                    hit = await db.scalar(
                        select(RetrievalHit)
                        .join(Retrieval, Retrieval.id == RetrievalHit.retrieval_id)
                        .where(
                            Retrieval.run_id == retrieval_id,
                            Retrieval.query_id == query.id,
                            RetrievalHit.rank == 1,
                        )
                    )
                    assert hit.corpus_id == hit.point_id == page.id
                    assert hit.chunk_id is None
        finally:
            client = AsyncQdrantClient(url=get_settings().qdrant_url)
            async with get_db() as db:
                names = (
                    await db.scalars(
                        select(EmbeddingRun.collection_name).where(
                            EmbeddingRun.dataset_id == dataset.id
                        )
                    )
                ).all()
            for name in names:
                if await client.collection_exists(name):
                    await client.delete_collection(name)
            await client.close()
            for obj in objects:
                await asyncio.to_thread(
                    object_store.client.remove_object, obj.bucket, obj.object_key
                )
            if objects:
                await asyncio.to_thread(
                    object_store.client.remove_bucket, object_store.bucket
                )
            async with get_db() as db:
                for table in reversed(Base.metadata.sorted_tables):
                    if "dataset_id" in table.c:
                        await db.execute(
                            delete(table).where(table.c.dataset_id == dataset.id)
                        )
                await db.execute(delete(Dataset).where(Dataset.id == dataset.id))
            await dispose_engine()

    asyncio.run(exercise())
