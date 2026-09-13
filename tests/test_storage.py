"""Database isolation and checkpoint behavior against a real PostgreSQL server."""

import os
from uuid import uuid4

import pytest
from sqlalchemy import ForeignKeyConstraint, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.schema import CreateTable

from src.settings import get_settings
from src.stor_rel import crud, entry
from src.stor_rel.schema import (
    Asset,
    Base,
    Corpus,
    Dataset,
    Doc,
    Experiment,
    Qrel,
    Query,
    RunItem,
    StageRun,
)


def test_schema_enforces_dataset_scoped_references():
    assert len(Base.metadata.tables) == 24
    for table in Base.metadata.tables.values():
        str(CreateTable(table).compile(dialect=postgresql.dialect()))
        if table.name == "datasets":
            continue
        assert "dataset_id" in table.c
        for constraint in table.constraints:
            if isinstance(constraint, ForeignKeyConstraint):
                target = constraint.elements[0].column.table.name
                if target != "datasets":
                    assert "dataset_id" in constraint.column_keys


def test_config_redaction_preserves_model_parameters():
    result = crud.canonical_config(
        {
            "api_key": "secret",
            "model": "test",
            "dense": {"provider_api_key": "hidden", "max_tokens": 32},
            "selected": [uuid4()],
        }
    )
    assert "api_key" not in result
    assert result["dense"] == {"max_tokens": 32}
    assert isinstance(result["selected"][0], str)


@pytest.fixture
async def storage_db(monkeypatch):
    if os.getenv("VDU_INTEGRATION") != "1":
        pytest.skip("Set VDU_INTEGRATION=1 with PostgreSQL running")
    schema = f"vdu_storage_test_{uuid4().hex}"
    admin = create_async_engine(get_settings().sql_database_url)
    async with admin.begin() as connection:
        await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_async_engine(
        get_settings().sql_database_url,
        connect_args={"server_settings": {"search_path": schema}},
    )
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    monkeypatch.setattr(entry, "get_engine", lambda: engine)
    monkeypatch.setattr(
        crud,
        "implementation_provenance",
        lambda: {"implementation_sha256": "test-fixture"},
    )
    try:
        yield engine
    finally:
        await engine.dispose()
        async with admin.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await admin.dispose()


async def source_records(name="fixture"):
    dataset = await crud.save_record(
        Dataset, source=name, fingerprint=uuid4().hex, status="completed"
    )
    doc = await crud.save_record(Doc, dataset_id=dataset.id, original_id="doc-1")
    asset = await crud.save_record(
        Asset,
        dataset_id=dataset.id,
        bucket="fixture",
        object_key="page-1.png",
        sha256="a" * 64,
        mime_type="image/png",
        size_bytes=42,
    )
    corpus = await crud.save_record(
        Corpus,
        dataset_id=dataset.id,
        original_id="page-1",
        doc_id=doc.id,
        asset_id=asset.id,
    )
    query = await crud.save_record(
        Query, dataset_id=dataset.id, original_id="query-1", query="What is the answer?"
    )
    return dataset, doc, corpus, query


@pytest.mark.integration
async def test_original_ids_and_rephrase_judgments_are_isolated(storage_db):
    first, doc, corpus, query = await source_records("first")
    second, other_doc, _, other_query = await source_records("second")
    assert query.id != other_query.id
    rephrase = await crud.save_record(
        Query,
        dataset_id=first.id,
        original_id="query-1",
        query="Answer please?",
        rephrase_of_id=query.id,
        rephrase_level=1,
    )
    await crud.save_record(
        Qrel,
        dataset_id=first.id,
        query_id=query.id,
        corpus_id=corpus.id,
        answer="42",
        score=2,
    )
    qrels = await crud.get_effective_qrels(first.id, [query.id, rephrase.id])
    assert {row["query_id"] for row in qrels} == {query.id, rephrase.id}
    assert all(row["corpus_id"] == corpus.id and row["answer"] == "42" for row in qrels)
    with pytest.raises(ValueError, match="another dataset"):
        await crud.get_effective_qrels(first.id, [other_query.id])
    with pytest.raises(IntegrityError):
        await crud.save_record(
            Query,
            dataset_id=first.id,
            original_id="chain",
            query="Invalid chain",
            rephrase_of_id=rephrase.id,
            rephrase_level=2,
        )
    with pytest.raises(IntegrityError):
        await crud.save_record(
            Query,
            dataset_id=second.id,
            original_id="cross",
            query="Invalid parent",
            rephrase_of_id=query.id,
            rephrase_level=1,
        )
    with pytest.raises(IntegrityError):
        await crud.save_record(
            Qrel,
            dataset_id=first.id,
            query_id=rephrase.id,
            corpus_id=corpus.id,
            answer="invalid",
            score=1,
        )
    with pytest.raises(IntegrityError):
        await crud.update_record(Corpus, corpus.id, doc_id=other_doc.id)
    # Failed short transactions leave the successful source rows intact.
    assert (await crud.get_record(Corpus, corpus.id)).doc_id == doc.id


@pytest.mark.integration
async def test_run_reuse_resume_and_dependency_validation(storage_db):
    dataset, _, corpus, query = await source_records()
    run = await crud.start_run(
        dataset.id,
        "preprocess",
        {"model": "mock", "api_key": "hidden"},
        selection=[str(corpus.id)],
    )
    assert run.status == "running" and "api_key" not in run.config
    with pytest.raises(ValueError, match="already running"):
        await crud.start_run(
            dataset.id, "preprocess", {"model": "mock"}, selection=[str(corpus.id)]
        )
    with pytest.raises(ValueError, match="expected completed"):
        await crud.start_run(
            dataset.id, "chunks", {}, inputs={"representations": run.id}
        )
    await crud.upsert_record(
        RunItem,
        {"run_id": run.id, "corpus_id": corpus.id},
        {"dataset_id": dataset.id, "status": "completed", "attempts": 1},
    )
    await crud.finish_run(
        run.id, status="partial", expected_count=2, completed_count=1, failed_count=1
    )
    resumed = await crud.start_run(
        dataset.id,
        "preprocess",
        {"model": "mock"},
        selection=[str(corpus.id)],
        resume_run_id=run.id,
    )
    assert resumed.id == run.id and resumed.status == "running"
    assert len(await crud.find_records(RunItem, run_id=run.id)) == 1
    await crud.finish_run(run.id, expected_count=1, completed_count=1, failed_count=0)
    reused = await crud.start_run(
        dataset.id, "preprocess", {"model": "mock"}, selection=[str(corpus.id)]
    )
    assert reused.id == run.id and reused.status == "completed"
    chunks = await crud.start_run(
        dataset.id, "chunks", {}, inputs={"representations": run.id}
    )
    assert chunks.id != run.id
    changed = await crud.start_run(
        dataset.id, "preprocess", {"model": "other"}, selection=[str(corpus.id)]
    )
    assert changed.id != run.id
    with pytest.raises(IntegrityError):
        await crud.save_record(
            RunItem,
            dataset_id=dataset.id,
            run_id=run.id,
            corpus_id=corpus.id,
            query_id=query.id,
        )


@pytest.mark.integration
async def test_atomic_upserts_and_experiment_pinning(storage_db):
    dataset, _, _, query = await source_records()
    existing = await crud.upsert_record(
        Query,
        {
            "dataset_id": dataset.id,
            "original_id": query.original_id,
            "rephrase_level": 0,
        },
        {"query": "Updated"},
    )
    assert existing.id == query.id and existing.query == "Updated"
    assert existing.created_at is not None
    experiment = await crud.save_record(Experiment, dataset_id=dataset.id, name="test")
    first = await crud.start_run(dataset.id, "retrieval", {"mode": "first"})
    second = await crud.start_run(dataset.id, "retrieval", {"mode": "second"})
    attached = await crud.attach_experiment_run(experiment.id, first.id, "retrieval")
    assert (
        await crud.attach_experiment_run(experiment.id, first.id, "retrieval")
    ).id == attached.id
    with pytest.raises(ValueError, match="already pinned"):
        await crud.attach_experiment_run(experiment.id, second.id, "retrieval")
    assert (await crud.get_record(StageRun, first.id)).config == {"mode": "first"}
