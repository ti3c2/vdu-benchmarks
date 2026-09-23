"""Ingest immutable REAL-MM-RAG snapshots into relational and object storage."""

import asyncio
import hashlib
import io
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import UUID

from PIL import Image

from src.settings import get_settings
from src.stor_obj import ObjectStore
from src.utils.progress import progress_bar


def image_payload(image: Any) -> tuple[bytes, str]:
    """Preserve encoded source bytes; encode decoded fixture images as PNG."""
    if isinstance(image, dict):
        data = image.get("bytes")
        if data is None and image.get("path"):
            data = Path(image["path"]).read_bytes()
    elif isinstance(image, bytes):
        data = image
    elif isinstance(image, (str, Path)):
        data = Path(image).read_bytes()
    elif isinstance(image, Image.Image):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        data = buffer.getvalue()
    else:
        raise ValueError(f"Unsupported source image type: {type(image).__name__}")
    if not data:
        raise ValueError("Page image is empty")
    with Image.open(io.BytesIO(data)) as decoded:
        mime_type = Image.MIME.get(decoded.format)
        decoded.verify()
    if mime_type is None:
        raise ValueError("Cannot determine source image MIME type")
    return data, mime_type


def validate_source(data: dict[str, Any]) -> dict[str, list[dict]]:
    """Normalize metadata and validate source relationships before any writes."""
    required = {"docs", "corpus", "queries", "qrels"}
    if missing := required - data.keys():
        raise ValueError(f"Missing source tables: {sorted(missing)}")
    normalized = {name: [] for name in required}
    doc_ids = set()
    for row in data["docs"]:
        if row.get("doc-id") is None:
            raise ValueError("Document is missing doc-id")
        original_id = str(row["doc-id"])
        if original_id in doc_ids:
            raise ValueError(f"Duplicate doc-id: {original_id}")
        doc_ids.add(original_id)
        normalized["docs"].append({"original_id": original_id})
    pages = data["corpus"]
    # HuggingFace can read just metadata without loading thousands of images.
    if hasattr(pages, "remove_columns"):
        pages = pages.remove_columns("image")
    corpus_ids = set()
    for ordinal, row in enumerate(pages):
        if row.get("corpus-id") is None or row.get("doc-id") is None:
            raise ValueError("Corpus page is missing corpus-id or doc-id")
        original_id, doc_original_id = str(row["corpus-id"]), str(row["doc-id"])
        if original_id in corpus_ids:
            raise ValueError(f"Duplicate corpus-id: {original_id}")
        if doc_original_id not in doc_ids:
            raise ValueError(
                f"Page {original_id} refers to unknown doc-id {doc_original_id}"
            )
        corpus_ids.add(original_id)
        normalized["corpus"].append(
            {
                "original_id": original_id,
                "doc_original_id": doc_original_id,
                "image_filename": row.get("image_filename"),
                "source_ordinal": ordinal,
                "metadata": {
                    key: value
                    for key, value in row.items()
                    if key not in {"image", "corpus-id", "doc-id", "image_filename"}
                },
            }
        )
    query_ids = set()
    for row in data["queries"]:
        if row.get("query-id") is None:
            raise ValueError("Query is missing query-id")
        original_id = str(row["query-id"])
        if original_id in query_ids:
            raise ValueError(f"Duplicate query-id: {original_id}")
        query_ids.add(original_id)
        for level in range(4):
            value = row.get("query" if level == 0 else f"rephrase_level_{level}")
            if not isinstance(value, str) or not value.strip():
                if level == 0:
                    raise ValueError(
                        f"Query {original_id} is missing its base query text"
                    )
                continue
            normalized["queries"].append(
                {
                    "original_id": original_id,
                    "query": value,
                    "language": row.get("language"),
                    "rephrase_level": level,
                }
            )
    qrel_ids = set()
    for row in data["qrels"]:
        if row.get("query-id") is None or row.get("corpus-id") is None:
            raise ValueError("Qrel is missing query-id or corpus-id")
        query_original_id, corpus_original_id = (
            str(row["query-id"]),
            str(row["corpus-id"]),
        )
        if query_original_id not in query_ids or corpus_original_id not in corpus_ids:
            raise ValueError(
                f"Qrel has unknown source relationship: {query_original_id}/{corpus_original_id}"
            )
        key = (query_original_id, corpus_original_id)
        if key in qrel_ids:
            raise ValueError(f"Duplicate query/page relevance judgment: {key}")
        qrel_ids.add(key)
        score = float(row["score"])
        if not math.isfinite(score) or score < 0:
            raise ValueError(f"Invalid relevance score: {row['score']}")
        normalized["qrels"].append(
            {
                "query_original_id": query_original_id,
                "corpus_original_id": corpus_original_id,
                "answer": row.get("answer"),
                "score": score,
            }
        )
    if not normalized["corpus"] or not normalized["queries"]:
        raise ValueError("A benchmark dataset must have pages and queries")
    return normalized


def load_real_mm_rag(source: str, revision: str | None = None, split: str = "test"):
    """Resolve a source revision once and load all four tables at that commit."""
    import datasets
    from huggingface_hub import HfApi

    resolved_revision = HfApi().dataset_info(source, revision=revision).sha
    if not resolved_revision:
        raise ValueError(f"Unable to resolve an immutable source revision for {source}")
    cache_dir = get_settings().path_data_raw / source.replace("/", "--")
    data = {}
    for name in ("docs", "corpus", "queries", "qrels"):
        data[name] = datasets.load_dataset(
            source,
            name=name,
            split=split,
            revision=resolved_revision,
            cache_dir=str(cache_dir),
        )
    # Keep original encoded bytes; decoding to PIL then reencoding loses the
    # original image format and prevents an exact byte-for-byte round trip.
    data["corpus"] = data["corpus"].cast_column("image", datasets.Image(decode=False))
    return resolved_revision, data


async def ingest_dataset(
    source: str,
    revision: str | None = None,
    split: str = "test",
    subset: str | None = None,
    *,
    data: dict[str, Any] | None = None,
    object_store: ObjectStore | None = None,
    resume_run_id: UUID | None = None,
) -> UUID:
    """Ingest a dataset snapshot; original IDs never become database identities.

    ``data`` and ``object_store`` allow a small in-memory fixture to exercise the
    exact production ingestion path without downloading or reprocessing OCR.
    """
    from src.stor_rel import crud
    from src.stor_rel.schema import (
        Asset,
        Corpus,
        Dataset,
        Doc,
        PageRepresentation,
        Qrel,
        Query,
        RunItem,
    )

    if subset:
        raise ValueError(
            "REAL-MM-RAG uses corpus/docs/queries/qrels configs; subset must be omitted"
        )
    supplied_data = data is not None
    if data is None:
        revision, data = await asyncio.to_thread(
            load_real_mm_rag, source, revision, split
        )
    normalized = validate_source(data)
    identity = {
        "source": source,
        "subset": subset,
        "split": split,
        "revision": revision,
    }
    content_hash = hashlib.sha256(
        json.dumps(
            {"identity": identity, "records": normalized}, sort_keys=True, default=str
        ).encode()
    )
    if supplied_data:
        for row in data["corpus"]:
            image_bytes, _ = image_payload(row["image"])
            content_hash.update(hashlib.sha256(image_bytes).digest())
    fingerprint = content_hash.hexdigest()
    revision = revision or f"fixture-{fingerprint}"
    snapshot_key = {
        "source": source,
        "subset": subset or "",
        "split": split,
        "revision": revision,
    }
    existing = await crud.find_records(Dataset, **snapshot_key)
    if existing:
        dataset = existing[0]
        if dataset.fingerprint != fingerprint:
            raise ValueError(
                "Source contents changed under the same immutable revision"
            )
        if dataset.status == "completed" and resume_run_id is None:
            return dataset.id
    else:
        dataset = await crud.upsert_record(
            Dataset,
            keys=snapshot_key,
            values={
                "fingerprint": fingerprint,
                "status": "pending",
                "metadata_json": {},
            },
        )
    config = {**identity, "revision": revision, "fingerprint": fingerprint}
    run = await crud.start_run(
        dataset.id, "ingest", config, resume_run_id=resume_run_id
    )
    if run.status == "completed":
        await crud.update_record(Dataset, dataset.id, status="completed")
        return dataset.id
    await crud.update_record(Dataset, dataset.id, status="running")
    store = object_store or ObjectStore()
    try:
        docs = {}
        with progress_bar(
            total=len(normalized["docs"]), desc="Ingesting docs", unit="doc"
        ) as progress:
            for row in normalized["docs"]:
                record = await crud.upsert_record(
                    Doc,
                    {"dataset_id": dataset.id, "original_id": row["original_id"]},
                    {},
                )
                docs[row["original_id"]] = record.id
                progress.update()
        queries = {}
        with progress_bar(
            total=len(normalized["queries"]), desc="Ingesting queries", unit="query"
        ) as progress:
            for row in normalized["queries"]:
                level = row["rephrase_level"]
                record = await crud.upsert_record(
                    Query,
                    {
                        "dataset_id": dataset.id,
                        "original_id": row["original_id"],
                        "rephrase_level": level,
                    },
                    {
                        "query": row["query"],
                        "language": row["language"],
                        "rephrase_of_id": queries[row["original_id"]]
                        if level
                        else None,
                    },
                )
                if level == 0:
                    queries[row["original_id"]] = record.id
                progress.update()
        corpus = {}
        existing_pages = {
            page.original_id: page
            for page in await crud.find_records(Corpus, dataset_id=dataset.id)
        }
        original_representations = {
            representation.corpus_id: representation
            for representation in await crud.find_records(
                PageRepresentation, dataset_id=dataset.id, kind="original_image"
            )
        }
        completed_pages = {
            item.corpus_id
            for item in await crud.find_records(
                RunItem, run_id=run.id, status="completed"
            )
        }
        with progress_bar(
            total=len(normalized["corpus"]), desc="Ingesting pages", unit="page"
        ) as progress:
            for row in normalized["corpus"]:
                previous = existing_pages.get(row["original_id"])
                if previous and previous.id in completed_pages:
                    corpus[row["original_id"]] = previous.id
                    progress.update()
                    continue
                image = data["corpus"][row["source_ordinal"]]["image"]
                image_bytes, mime_type = await asyncio.to_thread(image_payload, image)
                stored = await store.put_bytes(image_bytes, mime_type)
                asset = await crud.upsert_record(
                    Asset,
                    {
                        "dataset_id": dataset.id,
                        "bucket": stored.bucket,
                        "object_key": stored.object_key,
                    },
                    {
                        key: value
                        for key, value in asdict(stored).items()
                        if key not in {"bucket", "object_key"}
                    },
                )
                page = await crud.upsert_record(
                    Corpus,
                    {"dataset_id": dataset.id, "original_id": row["original_id"]},
                    {
                        "doc_id": docs[row["doc_original_id"]],
                        "image_filename": row["image_filename"],
                        "asset_id": asset.id,
                    },
                )
                corpus[row["original_id"]] = page.id
                original = original_representations.get(page.id)
                if original is not None:
                    if original.asset_id != asset.id:
                        raise ValueError(
                            "Original image changed inside an immutable dataset"
                        )
                else:
                    await crud.upsert_record(
                        PageRepresentation,
                        {
                            "run_id": run.id,
                            "corpus_id": page.id,
                            "kind": "original_image",
                        },
                        {
                            "dataset_id": dataset.id,
                            "asset_id": asset.id,
                            "metadata_json": {"source_metadata": row["metadata"]},
                        },
                    )
                await crud.upsert_record(
                    RunItem,
                    {"run_id": run.id, "corpus_id": page.id},
                    {
                        "dataset_id": dataset.id,
                        "status": "completed",
                        "attempts": 1,
                        "error": None,
                    },
                )
                progress.update()
        with progress_bar(
            total=len(normalized["qrels"]), desc="Ingesting qrels", unit="qrel"
        ) as progress:
            for row in normalized["qrels"]:
                await crud.upsert_record(
                    Qrel,
                    {
                        "dataset_id": dataset.id,
                        "query_id": queries[row["query_original_id"]],
                        "corpus_id": corpus[row["corpus_original_id"]],
                    },
                    {"answer": row["answer"], "score": row["score"]},
                )
                progress.update()
        counts = {name: len(rows) for name, rows in normalized.items()}
        await crud.finish_run(
            run.id,
            expected_count=len(corpus),
            completed_count=len(corpus),
            failed_count=0,
        )
        await crud.update_record(
            Dataset, dataset.id, status="completed", metadata_json={"counts": counts}
        )
    except Exception:
        await crud.update_record(Dataset, dataset.id, status="failed")
        await crud.finish_run(run.id, status="failed")
        raise
    return dataset.id
