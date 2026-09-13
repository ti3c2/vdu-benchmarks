"""Exercise ingestion and resumable processing without external model requests."""

import csv
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from PIL import Image

from src.config import ChunkConfig, EndpointProfile, PreprocessConfig
from src.etls.chunks import build_chunks
from src.etls.load_datasets import image_payload, ingest_dataset, validate_source
from src.etls.process_images import preprocess_pages, process_image
from src.stor_obj import StoredObject
from src.stor_rel import crud
from src.stor_rel.schema import (
    Asset,
    Chunk,
    Corpus,
    Dataset,
    Doc,
    PageRepresentation,
    Qrel,
    Query,
    StageRun,
)


@pytest.fixture
def source_data():
    return {
        "docs": [{"doc-id": "document-1"}],
        "corpus": [
            {
                "corpus-id": str(index),
                "doc-id": "document-1",
                "image": Image.new("RGB", (3, 3), color=(index * 40, 100, 150)),
                "image_filename": f"source-page-{index}.png",
            }
            for index in range(4)
        ],
        "queries": [
            {
                "query-id": "0",
                "query": "How much revenue?",
                "rephrase_level_1": "What was revenue?",
                "rephrase_level_2": "Report the revenue.",
                "rephrase_level_3": "Revenue amount?",
                "language": "en",
            }
        ],
        "qrels": [{"query-id": "0", "corpus-id": "0", "answer": "$1m", "score": 1}],
    }


@pytest.fixture
def memory_database(monkeypatch):
    """Replace I/O boundaries only; use production ETL business logic."""
    records = defaultdict(list)

    async def find_records(model, **filters):
        return [
            record
            for record in records[model]
            if all(
                getattr(record, key, None) == value for key, value in filters.items()
            )
        ]

    async def get_record(model, record_id):
        return next(
            (record for record in records[model] if record.id == record_id), None
        )

    async def upsert_record(model, keys, values):
        assert not keys.keys() & values.keys()
        if model is not StageRun:
            from sqlalchemy import UniqueConstraint

            unique_keys = [
                {column.name for column in constraint.columns}
                for constraint in model.__table__.constraints
                if isinstance(constraint, UniqueConstraint)
            ]
            assert set(keys) in unique_keys
        matches = await find_records(model, **keys)
        if matches:
            record = matches[0]
            for key, value in values.items():
                setattr(record, key, value)
        else:
            record = SimpleNamespace(id=uuid4(), **(keys | values))
            records[model].append(record)
        return record

    async def update_record(model, record_id, **values):
        record = await get_record(model, record_id)
        for key, value in values.items():
            setattr(record, key, value)
        return record

    async def start_run(
        dataset_id, kind, config, inputs=None, selection=None, resume_run_id=None
    ):
        identity = json.dumps(
            [str(dataset_id), kind, config, inputs, selection],
            sort_keys=True,
            default=str,
        )
        if resume_run_id:
            record = await get_record(StageRun, resume_run_id)
            assert record.fingerprint == identity
        else:
            existing = await find_records(StageRun, fingerprint=identity)
            record = (
                existing[0]
                if existing
                else await upsert_record(
                    StageRun,
                    {"dataset_id": dataset_id, "fingerprint": identity},
                    {"kind": kind, "config": config, "status": "pending"},
                )
            )
        if record.status != "completed":
            record.status = "running"
        return record

    async def finish_run(run_id, status="completed", **counts):
        return await update_record(StageRun, run_id, status=status, **counts)

    async def validate_run(run_id, kind=None, dataset_id=None):
        record = await get_record(StageRun, run_id)
        if record is None or record.status != "completed":
            raise ValueError("Run is not completed")
        if kind and record.kind != kind:
            raise ValueError("Wrong run kind")
        if dataset_id and record.dataset_id != dataset_id:
            raise ValueError("Wrong dataset")
        return record

    for name, function in locals().copy().items():
        if name in {
            "get_record",
            "find_records",
            "upsert_record",
            "update_record",
            "start_run",
            "finish_run",
            "validate_run",
        }:
            monkeypatch.setattr(crud, name, function)
    return records


@pytest.fixture
def object_store():
    class MemoryObjects:
        def __init__(self):
            self.objects = {}
            self.uploads = 0

        async def put_bytes(self, data, mime_type):
            digest = hashlib.sha256(data).hexdigest()
            self.objects[digest] = data
            self.uploads += 1
            return StoredObject("test-bucket", digest, digest, mime_type, len(data))

        async def read_asset(self, asset):
            return self.objects[asset.object_key]

    return MemoryObjects()


def test_validate_original_ids_rephrases_and_relationships(source_data):
    normalized = validate_source(source_data)
    assert len(normalized["queries"]) == 4
    assert {row["original_id"] for row in normalized["queries"]} == {"0"}
    assert len(normalized["qrels"]) == 1
    source_data["qrels"][0]["corpus-id"] = "missing"
    with pytest.raises(ValueError, match="unknown source relationship"):
        validate_source(source_data)


def test_duplicate_query_and_empty_base_fail(source_data):
    source_data["queries"].append(source_data["queries"][0].copy())
    with pytest.raises(ValueError, match="Duplicate query-id"):
        validate_source(source_data)
    source_data["queries"].pop()
    source_data["queries"][0]["query"] = "  "
    with pytest.raises(ValueError, match="base query text"):
        validate_source(source_data)


def test_image_bytes_and_mime_are_preserved():
    image = Image.new("RGB", (3, 3), "red")
    output = io.BytesIO()
    image.save(output, format="JPEG")
    raw = output.getvalue()
    assert image_payload({"bytes": raw}) == (raw, "image/jpeg")
    encoded, mime_type = image_payload(image)
    assert mime_type == "image/png"
    assert encoded.startswith(b"\x89PNG")


async def test_ingestion_is_idempotent_and_dataset_scoped(
    source_data, memory_database, object_store
):
    first = await ingest_dataset(
        "fixture/one", revision="v1", data=source_data, object_store=object_store
    )
    same = await ingest_dataset(
        "fixture/one", revision="v1", data=source_data, object_store=object_store
    )
    second = await ingest_dataset(
        "fixture/two", revision="v1", data=source_data, object_store=object_store
    )
    assert first == same and first != second
    assert object_store.uploads == 8
    assert len(memory_database[Corpus]) == 8
    assert len({page.id for page in memory_database[Corpus]}) == 8
    assert len(memory_database[Doc]) == 2
    assert len(memory_database[Query]) == 8
    assert len(memory_database[Qrel]) == 2
    for dataset_id in (first, second):
        queries = await crud.find_records(Query, dataset_id=dataset_id)
        base = next(query for query in queries if query.rephrase_level == 0)
        assert all(
            query.rephrase_of_id == base.id for query in queries if query.rephrase_level
        )
        qrels = await crud.find_records(Qrel, dataset_id=dataset_id)
        assert qrels[0].query_id == base.id
        assert (await crud.get_record(Dataset, dataset_id)).status == "completed"
    original = await crud.find_records(PageRepresentation, dataset_id=first)
    assert len(original) == 4 and all(rep.kind == "original_image" for rep in original)
    # An immutable completed snapshot remains reusable if run implementation
    # fingerprints change later; it must not duplicate original-image records.
    for stage in memory_database[StageRun]:
        stage.fingerprint = "previous implementation"
    assert (
        await ingest_dataset(
            "fixture/one", revision="v1", data=source_data, object_store=object_store
        )
        == first
    )
    assert len(await crud.find_records(PageRepresentation, dataset_id=first)) == 4
    assert object_store.uploads == 8


async def test_changed_revision_is_new_snapshot(
    source_data, memory_database, object_store
):
    first = await ingest_dataset(
        "fixture/one", revision="v1", data=source_data, object_store=object_store
    )
    source_data["queries"][0]["query"] = "Changed query"
    with pytest.raises(ValueError, match="same immutable revision"):
        await ingest_dataset(
            "fixture/one", revision="v1", data=source_data, object_store=object_store
        )
    second = await ingest_dataset(
        "fixture/one", revision="v2", data=source_data, object_store=object_store
    )
    assert first != second


async def test_preprocessing_resume_and_chunks_reuse_cached_ocr(
    source_data, memory_database, object_store
):
    fixture = (
        Path(__file__).parents[1]
        / "data/processed/20260907-212356_ibm-research-REAL-MM-RAG_FinReport_BEIR_deepseek-ai"
        / "i2t/DeepSeek-OCR-2_deepseek-ocr_i2t.csv"
    )
    if fixture.exists():
        with fixture.open(encoding="utf-8", newline="") as stream:
            ocr = list(csv.DictReader(stream))[:4]
    else:
        ocr = json.loads(
            (Path(__file__).parent / "fixtures/ocr_samples.json").read_text()
        )["rows"]
    dataset_id = await ingest_dataset(
        "fixture/ocr", revision="v1", data=source_data, object_store=object_store
    )
    pages = sorted(
        await crud.find_records(Corpus, dataset_id=dataset_id),
        key=lambda page: page.original_id,
    )
    expected = {}
    for page, row in zip(pages, ocr, strict=True):
        asset = await crud.get_record(Asset, page.asset_id)
        expected[await object_store.read_asset(asset)] = row["text"]
    calls = defaultdict(int)
    failure_image = next(iter(expected))

    async def cached_ocr(image_bytes, mime_type, config):
        assert mime_type == "image/png"
        calls[image_bytes] += 1
        if image_bytes == failure_image and calls[image_bytes] == 1:
            raise RuntimeError("Simulated interrupted OCR request")
        return {"text": expected[image_bytes], "metadata": {"fixture": True}}

    config = PreprocessConfig(endpoint=EndpointProfile(model="cached-ocr"))
    run_id = await preprocess_pages(
        dataset_id, config, object_store=object_store, ocr_processor=cached_ocr
    )
    assert (await crud.get_record(StageRun, run_id)).status == "partial"
    with pytest.raises(ValueError, match="not completed"):
        await build_chunks(run_id, ChunkConfig())
    resumed = await preprocess_pages(
        dataset_id,
        config,
        resume_run_id=run_id,
        object_store=object_store,
        ocr_processor=cached_ocr,
    )
    assert resumed == run_id
    assert (await crud.get_record(StageRun, run_id)).status == "completed"
    assert sorted(calls.values()) == [1, 1, 1, 2]
    assert (
        await preprocess_pages(
            dataset_id, config, object_store=object_store, ocr_processor=cached_ocr
        )
        == run_id
    )
    assert sum(calls.values()) == 5
    chunk_run_id = await build_chunks(run_id, ChunkConfig(max_chars=600))
    chunks = await crud.find_records(Chunk, run_id=chunk_run_id)
    assert chunks and any(chunk.kind == "table" for chunk in chunks)
    assert all(len(chunk.text) <= 600 for chunk in chunks)
    assert await build_chunks(run_id, ChunkConfig(max_chars=600)) == chunk_run_id
    assert len(await crud.find_records(Chunk, run_id=chunk_run_id)) == len(chunks)
    assert await build_chunks(run_id, ChunkConfig(max_chars=900)) != chunk_run_id
    for rep in await crud.find_records(PageRepresentation, run_id=run_id):
        assert (
            rep.text
            == expected[
                await object_store.read_asset(
                    await crud.get_record(
                        Asset, next(p.asset_id for p in pages if p.id == rep.corpus_id)
                    )
                )
            ]
        )


async def test_ocr_request_uses_actual_mime_and_seconds():
    requests = []

    async def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(
            id="response-id",
            model="ocr-model",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="  raw OCR  "), finish_reason="stop"
                )
            ],
            usage=None,
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    config = PreprocessConfig(
        endpoint=EndpointProfile(model="ocr-model", timeout_seconds=7)
    )
    result = await process_image(b"image-content", "image/png", config, client=client)
    assert result["text"] == "  raw OCR  "
    assert requests[0]["timeout"] == 7
    assert requests[0]["messages"][0]["content"][0]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )


async def test_interrupted_ingestion_reuses_originals_after_implementation_change(
    source_data, memory_database, object_store, monkeypatch
):
    put_bytes = object_store.put_bytes

    async def interrupted_upload(data, mime_type):
        if object_store.uploads == 2:
            raise RuntimeError("Simulated object storage interruption")
        return await put_bytes(data, mime_type)

    monkeypatch.setattr(object_store, "put_bytes", interrupted_upload)
    with pytest.raises(RuntimeError, match="interruption"):
        await ingest_dataset(
            "fixture/interrupted",
            revision="v1",
            data=source_data,
            object_store=object_store,
        )
    dataset = memory_database[Dataset][0]
    assert dataset.status == "failed"
    assert len(memory_database[PageRepresentation]) == 2
    old_original_ids = {rep.id for rep in memory_database[PageRepresentation]}
    for run in memory_database[StageRun]:
        run.fingerprint = "previous implementation"
    monkeypatch.setattr(object_store, "put_bytes", put_bytes)
    assert (
        await ingest_dataset(
            "fixture/interrupted",
            revision="v1",
            data=source_data,
            object_store=object_store,
        )
        == dataset.id
    )
    assert dataset.status == "completed"
    assert len(memory_database[PageRepresentation]) == 4
    assert old_original_ids <= {rep.id for rep in memory_database[PageRepresentation]}
    assert len(memory_database[Corpus]) == 4
