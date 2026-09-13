"""Real PostgreSQL/MinIO integration; cached OCR and fake generation only."""

import asyncio
import base64
import csv
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from PIL import Image
from sqlalchemy.exc import IntegrityError
from test_storage import storage_db as storage_db

from src.config import (
    ChunkConfig,
    ContextConfig,
    EndpointProfile,
    GenerationConfig,
    IRConfig,
    MetricConfig,
    PreprocessConfig,
    RagasConfig,
)
from src.etls.chunks import build_chunks
from src.etls.load_datasets import image_payload, ingest_dataset
from src.etls.process_images import preprocess_pages
from src.evaluate.generation import run_generation
from src.evaluate.ir import calculate_ir, evaluate_ir
from src.evaluate.ragas import evaluate_ragas
from src.stor_obj import ObjectStore
from src.stor_rel import crud
from src.stor_rel.schema import (
    Asset,
    Chunk,
    Corpus,
    Dataset,
    EmbeddingRun,
    Experiment,
    ExperimentQuery,
    Generation,
    GenerationContext,
    MetricAggregate,
    MetricResult,
    Qrel,
    Query,
    Retrieval,
    RetrievalHit,
    RetrievalRun,
    StageRun,
)

REFERENCE = "REFERENCE_ONLY_42"
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("VDU_INTEGRATION") != "1", reason="requires PostgreSQL and MinIO"
    ),
]


@pytest.fixture
async def pipeline_source(storage_db):
    fixture = (
        Path(__file__).parents[1]
        / "data/processed/20260907-212356_ibm-research-REAL-MM-RAG_FinReport_BEIR_deepseek-ai"
        / "i2t/DeepSeek-OCR-2_deepseek-ocr_i2t.csv"
    )
    if fixture.exists():
        with fixture.open(encoding="utf-8", newline="") as stream:
            ocr_rows = list(csv.DictReader(stream))[:4]
    else:
        ocr_rows = json.loads(
            (Path(__file__).parent / "fixtures/ocr_samples.json").read_text()
        )["rows"]
    source = {
        "docs": [{"doc-id": "report"}],
        "corpus": [
            {
                "corpus-id": row["corpus-id"],
                "doc-id": "report",
                "image": Image.new("RGB", (5, 5), (index * 40, 90, 120)),
                "image_filename": f"fixture-page-{index}.png",
            }
            for index, row in enumerate(ocr_rows)
        ],
        "queries": [
            {
                "query-id": "0",
                "query": "What does the report say?",
                "rephrase_level_1": "What information is reported?",
                "rephrase_level_2": "Tell me about the report.",
                "rephrase_level_3": "Describe the report.",
                "language": "en",
            }
        ],
        "qrels": [{"query-id": "0", "corpus-id": "0", "answer": REFERENCE, "score": 1}],
    }
    store = ObjectStore(bucket=f"vdu-generation-test-{uuid4().hex}")
    cached = {
        image_payload(page["image"])[0]: row["text"]
        for page, row in zip(source["corpus"], ocr_rows, strict=True)
    }

    async def cached_ocr(image_bytes, mime_type, config):
        return {"text": cached[image_bytes], "metadata": {"fixture": True}}

    try:
        dataset_id = await ingest_dataset(
            f"fixture/generation-{uuid4().hex}",
            revision="v1",
            data=source,
            object_store=store,
        )
        assert (
            await ingest_dataset(
                source=(await crud.get_record(Dataset, dataset_id)).source,
                revision="v1",
                data=source,
                object_store=store,
            )
            == dataset_id
        )
        preprocess_id = await preprocess_pages(
            dataset_id,
            PreprocessConfig(endpoint=EndpointProfile(model="cached-ocr")),
            object_store=store,
            ocr_processor=cached_ocr,
        )
        assert (await crud.get_record(StageRun, preprocess_id)).status == "completed"
        chunks_id = await build_chunks(preprocess_id, ChunkConfig(max_chars=600))
        assert (await crud.get_record(StageRun, chunks_id)).status == "completed"
        queries = sorted(
            await crud.find_records(Query, dataset_id=dataset_id),
            key=lambda q: q.rephrase_level,
        )
        pages = sorted(
            await crud.find_records(Corpus, dataset_id=dataset_id),
            key=lambda p: p.original_id,
        )
        chunks = await crud.find_records(Chunk, run_id=chunks_id)
        experiment = await crud.save_record(
            Experiment, dataset_id=dataset_id, name="fixture"
        )
        for query in queries:
            await crud.save_record(
                ExperimentQuery,
                dataset_id=dataset_id,
                experiment_id=experiment.id,
                query_id=query.id,
            )
        yield SimpleNamespace(
            dataset_id=dataset_id,
            pages=pages,
            queries=queries,
            chunks=chunks,
            preprocess_id=preprocess_id,
            chunks_id=chunks_id,
            experiment_id=experiment.id,
            ocr_rows=ocr_rows,
            store=store,
        )
    finally:

        def cleanup():
            if store.client.bucket_exists(store.bucket):
                for obj in store.client.list_objects(store.bucket, recursive=True):
                    store.client.remove_object(store.bucket, obj.object_name)
                store.client.remove_bucket(store.bucket)

        await asyncio.to_thread(cleanup)


async def saved_retrieval(source, *, empty_level=None):
    query_run = await crud.start_run(
        source.dataset_id, "vectorize_queries", {"fixture": True}
    )
    corpus_run = await crud.start_run(
        source.dataset_id,
        "vectorize_chunks",
        {"fixture": True},
        inputs={"chunks": source.chunks_id},
    )
    for stage, role, unit in (
        (query_run, "query", "query"),
        (corpus_run, "corpus", "chunk"),
    ):
        await crud.upsert_record(
            EmbeddingRun,
            {"id": stage.id},
            {
                "dataset_id": source.dataset_id,
                "role": role,
                "unit_kind": unit,
                "collection_name": f"fixture-{stage.id}",
            },
        )
        await crud.finish_run(stage.id)
    run = await crud.start_run(
        source.dataset_id,
        "retrieval",
        {"page_top_k": 4, "empty_level": empty_level},
        inputs={"query_embeddings": query_run.id, "corpus_embeddings": corpus_run.id},
    )
    await crud.save_record(
        RetrievalRun,
        id=run.id,
        dataset_id=source.dataset_id,
        query_embedding_run_id=query_run.id,
        corpus_embedding_run_id=corpus_run.id,
    )
    for query in source.queries:
        retrieval = await crud.save_record(
            Retrieval,
            dataset_id=source.dataset_id,
            run_id=run.id,
            query_id=query.id,
            status="completed",
        )
        if query.rephrase_level == empty_level:
            continue
        for rank, page in enumerate((source.pages[1], source.pages[0]), start=1):
            chunk = next(chunk for chunk in source.chunks if chunk.corpus_id == page.id)
            await crud.save_record(
                RetrievalHit,
                dataset_id=source.dataset_id,
                retrieval_id=retrieval.id,
                corpus_id=page.id,
                chunk_id=chunk.id,
                point_id=chunk.id,
                rank=rank,
                score=1 / rank,
            )
    await crud.finish_run(
        run.id,
        expected_count=len(source.queries),
        completed_count=len(source.queries),
        failed_count=0,
    )
    return run.id


def completion_recorder(calls):
    async def complete(messages):
        calls.append(messages)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=REFERENCE))],
            usage=None,
        )

    return complete


async def test_whole_pages_images_no_reference_leak_and_classical_ragas(
    pipeline_source,
):
    source = pipeline_source
    retrieval_id = await saved_retrieval(source)
    config = GenerationConfig(
        endpoint=EndpointProfile(model="fixture-generation"),
        context=ContextConfig(
            representations=["ocr", "original_image"],
            representation_run_id=source.preprocess_id,
            page_top_k=2,
        ),
    )
    calls = []
    generation_id = await run_generation(
        retrieval_id,
        config,
        object_store=source.store,
        completion=completion_recorder(calls),
    )
    assert (await crud.get_record(StageRun, generation_id)).status == "completed"
    assert len(calls) == 4
    expected_images = []
    for page in (source.pages[1], source.pages[0]):
        asset = await crud.get_record(Asset, page.asset_id)
        expected_images.append(await source.store.read_asset(asset))
    for messages in calls:
        assert REFERENCE not in str(messages)
        parts = messages[1]["content"]
        text_parts = [part["text"] for part in parts if part["type"] == "text"]
        assert source.ocr_rows[1]["text"] in text_parts
        assert source.ocr_rows[0]["text"] in text_parts
        assert text_parts.index(source.ocr_rows[1]["text"]) < text_parts.index(
            source.ocr_rows[0]["text"]
        )
        assert len(source.ocr_rows[0]["text"]) > max(
            len(chunk.text) for chunk in source.chunks
        )
        image_parts = [
            part["image_url"]["url"] for part in parts if part["type"] == "image_url"
        ]
        assert all(url.startswith("data:image/png;base64,") for url in image_parts)
        assert [
            base64.b64decode(url.split(",", 1)[1]) for url in image_parts
        ] == expected_images
    generations = await crud.find_records(Generation, run_id=generation_id)
    for generation in generations:
        contexts = await crud.find_records(
            GenerationContext, generation_id=generation.id
        )
        assert sorted(context.position for context in contexts) == [0, 1, 2, 3]
        assert len(generation.request["contexts"]) == 4
    assert (
        await run_generation(
            retrieval_id,
            config,
            object_store=source.store,
            completion=completion_recorder(calls),
        )
        == generation_id
    )
    assert len(calls) == 4
    metrics = RagasConfig(
        metrics=[MetricConfig(id="exact_match"), MetricConfig(id="string_presence")]
    )
    evaluation_id = await evaluate_ragas(
        source.experiment_id,
        retrieval_id,
        generation_id,
        metrics,
        object_store=source.store,
    )
    results = await crud.find_records(MetricResult, evaluation_run_id=evaluation_id)
    assert len(results) == 8
    assert all(result.status == "completed" and result.value == 1 for result in results)
    aggregates = await crud.find_records(
        MetricAggregate, evaluation_run_id=evaluation_id, group_by="dataset"
    )
    assert len(aggregates) == 2
    assert all(
        row.value == 1 and row.scored_count == 4 and row.failed_count == 0
        for row in aggregates
    )
    assert (
        await evaluate_ragas(
            source.experiment_id,
            retrieval_id,
            generation_id,
            metrics,
            object_store=source.store,
        )
        == evaluation_id
    )


async def test_ir_rephrases_empty_results_and_duplicate_page_constraint(
    pipeline_source,
):
    source = pipeline_source
    retrieval_id = await saved_retrieval(source, empty_level=3)
    evaluation_id = await evaluate_ir(
        source.experiment_id, retrieval_id, IRConfig(cutoffs=[1, 2])
    )
    results = await crud.find_records(MetricResult, evaluation_run_id=evaluation_id)
    assert len(results) == 32
    values = {(row.query_id, row.metric_id): row.value for row in results}
    for query in source.queries:
        assert values[query.id, "P@1"] == 0
        assert values[query.id, "R@1"] == 0
        expected = 0 if query.rephrase_level == 3 else 1
        assert values[query.id, "R@2"] == expected
        assert values[query.id, "RR@2"] == expected / 2
        assert values[query.id, "nDCG@2"] == pytest.approx(expected / math.log2(3))
    assert (
        await evaluate_ir(source.experiment_id, retrieval_id, IRConfig(cutoffs=[1, 2]))
        == evaluation_id
    )
    retrieval = (
        await crud.find_records(
            Retrieval, run_id=retrieval_id, query_id=source.queries[0].id
        )
    )[0]
    hit = (await crud.find_records(RetrievalHit, retrieval_id=retrieval.id))[0]
    with pytest.raises(IntegrityError):
        await crud.save_record(
            RetrievalHit,
            dataset_id=source.dataset_id,
            retrieval_id=retrieval.id,
            corpus_id=hit.corpus_id,
            point_id=hit.point_id,
            rank=3,
            score=0.1,
        )
    with pytest.raises(ValueError, match="duplicate"):
        calculate_ir(
            [{"query_id": "q", "corpus_id": "p", "score": 1}],
            [
                {"query_id": "q", "corpus_id": "p", "rank": 1},
                {"query_id": "q", "corpus_id": "p", "rank": 2},
            ],
            IRConfig(cutoffs=[1]),
        )
    with pytest.raises(ValueError, match="integer relevance"):
        calculate_ir(
            [{"query_id": "q", "corpus_id": "p", "score": 0.5}],
            [],
            IRConfig(cutoffs=[1]),
        )


async def test_generation_missing_context_and_budget_are_recorded_failures(
    pipeline_source,
):
    source = pipeline_source
    retrieval_id = await saved_retrieval(source)
    calls = []
    for context in (
        ContextConfig(
            representations=["missing"],
            representation_run_id=source.preprocess_id,
            page_top_k=2,
        ),
        ContextConfig(
            representations=["ocr"],
            representation_run_id=source.preprocess_id,
            page_top_k=2,
            max_text_chars=10,
        ),
    ):
        config = GenerationConfig(
            endpoint=EndpointProfile(model="fixture-generation"), context=context
        )
        run_id = await run_generation(
            retrieval_id,
            config,
            object_store=source.store,
            completion=completion_recorder(calls),
        )
        assert (await crud.get_record(StageRun, run_id)).status in {"failed", "partial"}
        generations = await crud.find_records(Generation, run_id=run_id)
        assert len(generations) == 4
        assert all(
            generation.status == "failed" and generation.error
            for generation in generations
        )
    assert calls == []


async def test_ambiguous_references_are_unscored_and_failure_coverage_is_preserved(
    pipeline_source,
):
    source = pipeline_source
    retrieval_id = await saved_retrieval(source)
    calls = []
    config = GenerationConfig(
        endpoint=EndpointProfile(model="fixture-generation"),
        context=ContextConfig(
            representations=["ocr"],
            representation_run_id=source.preprocess_id,
            page_top_k=2,
        ),
    )
    generation_id = await run_generation(
        retrieval_id,
        config,
        object_store=source.store,
        completion=completion_recorder(calls),
    )
    await crud.save_record(
        Qrel,
        dataset_id=source.dataset_id,
        query_id=source.queries[0].id,
        corpus_id=source.pages[2].id,
        answer="A different reference answer",
        score=1,
    )
    run_id = await evaluate_ragas(
        source.experiment_id,
        retrieval_id,
        generation_id,
        RagasConfig(metrics=[MetricConfig(id="exact_match")]),
        object_store=source.store,
    )
    results = await crud.find_records(MetricResult, evaluation_run_id=run_id)
    assert len(results) == 4
    assert all(
        row.status == "skipped" and row.value is None and "reference" in row.reason
        for row in results
    )
    aggregate = (
        await crud.find_records(
            MetricAggregate, evaluation_run_id=run_id, group_by="dataset"
        )
    )[0]
    assert (
        aggregate.expected_count == 4
        and aggregate.skipped_count == 4
        and aggregate.scored_count == 0
    )
    assert aggregate.value is None


async def test_llm_metric_adapters_receive_whole_text_and_actual_images(
    pipeline_source,
):
    from src.evaluate.metrics import resolve_metric

    source = pipeline_source
    retrieval_id = await saved_retrieval(source)
    generation_id = await run_generation(
        retrieval_id,
        GenerationConfig(
            endpoint=EndpointProfile(model="fixture-generation"),
            context=ContextConfig(
                representations=["ocr", "original_image"],
                representation_run_id=source.preprocess_id,
                page_top_k=2,
            ),
        ),
        object_store=source.store,
        completion=completion_recorder([]),
    )
    text_samples, image_samples = [], []

    class TextMetric:
        async def ascore(self, user_input, response, retrieved_contexts):
            text_samples.append(retrieved_contexts)
            assert retrieved_contexts == [
                source.ocr_rows[1]["text"],
                source.ocr_rows[0]["text"],
            ]
            return SimpleNamespace(
                value=0.75,
                reason="Captured complete OCR pages",
                traces={"fixture": True},
            )

    class ImageMetric:
        async def ascore(self, response, retrieved_contexts):
            image_samples.append(retrieved_contexts)
            assert len(retrieved_contexts) == 4
            assert retrieved_contexts[0] == source.ocr_rows[1]["text"]
            assert retrieved_contexts[2] == source.ocr_rows[0]["text"]
            assert retrieved_contexts[1].startswith("data:image/png;base64,")
            assert retrieved_contexts[3].startswith("data:image/png;base64,")
            for page, url in zip(
                (source.pages[1], source.pages[0]),
                retrieved_contexts[1::2],
                strict=True,
            ):
                asset = await crud.get_record(Asset, page.asset_id)
                assert base64.b64decode(
                    url.split(",", 1)[1]
                ) == await source.store.read_asset(asset)
            return SimpleNamespace(
                value=0.5,
                reason="Captured complete image bytes",
                traces={"fixture": True},
            )

    config = RagasConfig(
        metrics=[
            MetricConfig(id="faithfulness"),
            MetricConfig(id="multi_modal_faithfulness"),
        ],
        judge=EndpointProfile(model="never-called-fixture-judge"),
    )

    def metric_factory(config):
        return [
            (
                selected,
                resolve_metric(selected.id),
                TextMetric() if selected.id == "faithfulness" else ImageMetric(),
            )
            for selected in config.metrics
        ], None

    evaluation_id = await evaluate_ragas(
        source.experiment_id,
        retrieval_id,
        generation_id,
        config,
        object_store=source.store,
        metric_factory=metric_factory,
    )
    assert len(text_samples) == len(image_samples) == 4
    rows = await crud.find_records(MetricResult, evaluation_run_id=evaluation_id)
    assert all(
        row.status == "completed" and row.traces == {"fixture": True} for row in rows
    )
    assert {row.value for row in rows} == {0.75, 0.5}


async def test_metric_failures_nonfinite_values_and_resume_preserve_successes(
    pipeline_source,
):
    from src.evaluate.metrics import resolve_metric

    source = pipeline_source
    retrieval_id = await saved_retrieval(source)
    generation_id = await run_generation(
        retrieval_id,
        GenerationConfig(
            endpoint=EndpointProfile(model="fixture-generation"),
            context=ContextConfig(
                representations=["ocr"],
                representation_run_id=source.preprocess_id,
                page_top_k=2,
            ),
        ),
        object_store=source.store,
        completion=completion_recorder([]),
    )
    calls = []

    class FlakyMetric:
        async def ascore(self, response, reference):
            calls.append((response, reference))
            if len(calls) == 1:
                raise RuntimeError("Simulated metric failure")
            if len(calls) == 2:
                return SimpleNamespace(value=float("nan"))
            return SimpleNamespace(value=1, reason="Successful metric response")

    metric = FlakyMetric()
    config = RagasConfig(metrics=[MetricConfig(id="exact_match")])

    def metric_factory(config):
        return [(config.metrics[0], resolve_metric("exact_match"), metric)], None

    evaluation_id = await evaluate_ragas(
        source.experiment_id,
        retrieval_id,
        generation_id,
        config,
        object_store=source.store,
        metric_factory=metric_factory,
    )
    assert len(calls) == 4
    assert (await crud.get_record(StageRun, evaluation_id)).status == "partial"
    rows = await crud.find_records(MetricResult, evaluation_run_id=evaluation_id)
    failed = [row for row in rows if row.status == "failed"]
    assert len(failed) == 2
    assert any("nonfinite" in row.error for row in failed)
    assert any("Simulated metric failure" in row.error for row in failed)
    successful_ids = {row.id for row in rows if row.status == "completed"}
    aggregate = (
        await crud.find_records(
            MetricAggregate, evaluation_run_id=evaluation_id, group_by="dataset"
        )
    )[0]
    assert (
        aggregate.expected_count == 4
        and aggregate.scored_count == 2
        and aggregate.failed_count == 2
    )
    assert (
        await evaluate_ragas(
            source.experiment_id,
            retrieval_id,
            generation_id,
            config,
            resume_run_id=evaluation_id,
            object_store=source.store,
            metric_factory=metric_factory,
        )
        == evaluation_id
    )
    assert len(calls) == 6
    rows = await crud.find_records(MetricResult, evaluation_run_id=evaluation_id)
    assert len(rows) == 4 and all(
        row.status == "completed" and row.value == 1 for row in rows
    )
    assert successful_ids <= {row.id for row in rows}
    assert (await crud.get_record(StageRun, evaluation_id)).status == "completed"
    assert (
        await evaluate_ragas(
            source.experiment_id,
            retrieval_id,
            generation_id,
            config,
            object_store=source.store,
            metric_factory=metric_factory,
        )
        == evaluation_id
    )
    assert len(calls) == 6


async def test_ragas_max_concurrency_applies_to_metric_calls(pipeline_source):
    from src.evaluate.metrics import resolve_metric

    source = pipeline_source
    retrieval_id = await saved_retrieval(source)
    generation_id = await run_generation(
        retrieval_id,
        GenerationConfig(
            endpoint=EndpointProfile(model="fixture-generation"),
            context=ContextConfig(
                representations=["ocr"],
                representation_run_id=source.preprocess_id,
                page_top_k=2,
            ),
        ),
        object_store=source.store,
        completion=completion_recorder([]),
    )
    state = {"active": 0, "max_active": 0, "started": 0}
    lock = asyncio.Lock()

    class SlowMetric:
        async def ascore(self, response, reference):
            async with lock:
                state["active"] += 1
                state["started"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            try:
                await asyncio.sleep(0.05)
                return SimpleNamespace(value=1)
            finally:
                async with lock:
                    state["active"] -= 1

    config = RagasConfig(
        metrics=[MetricConfig(id="exact_match"), MetricConfig(id="string_presence")],
        ragas_max_concurrency=8,
    )

    def metric_factory(config):
        return [
            (selected, resolve_metric(selected.id), SlowMetric())
            for selected in config.metrics
        ], None

    evaluation_id = await evaluate_ragas(
        source.experiment_id,
        retrieval_id,
        generation_id,
        config,
        object_store=source.store,
        metric_factory=metric_factory,
    )
    assert (await crud.get_record(StageRun, evaluation_id)).status == "completed"
    assert state["started"] == len(source.queries) * len(config.metrics)
    assert state["max_active"] == 8
    rows = await crud.find_records(MetricResult, evaluation_run_id=evaluation_id)
    assert len(rows) == 8 and all(
        row.status == "completed" and row.value == 1 for row in rows
    )
