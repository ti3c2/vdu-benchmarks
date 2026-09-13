"""Metric inputs and partial recovery, with cached OCR and mocked judges."""

import os
from types import SimpleNamespace

import pytest
from test_generation_evaluation import (
    REFERENCE,
    completion_recorder,
    saved_retrieval,
)
from test_generation_evaluation import pipeline_source as pipeline_source
from test_storage import storage_db as storage_db

from src.config import (
    ContextConfig,
    EndpointProfile,
    GenerationConfig,
    MetricConfig,
    RagasConfig,
)
from src.evaluate.generation import run_generation
from src.evaluate.metrics import resolve_metric
from src.evaluate.ragas import create_metrics, evaluate_ragas
from src.stor_rel import crud
from src.stor_rel.schema import MetricAggregate, MetricResult, StageRun

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("VDU_INTEGRATION") != "1", reason="requires PostgreSQL and MinIO"
    ),
]


async def test_reference_free_rubrics_do_not_receive_gold_inputs(pipeline_source):
    source = pipeline_source
    retrieval_id = await saved_retrieval(source)
    generation_id = await run_generation(
        retrieval_id,
        GenerationConfig(
            endpoint=EndpointProfile(model="fake"),
            context=ContextConfig(representations=["original_image"], page_top_k=2),
        ),
        object_store=source.store,
        completion=completion_recorder([]),
    )
    captures = []

    class Rubric:
        def __init__(self, with_reference):
            self.with_reference = with_reference

        async def ascore(
            self,
            user_input,
            response,
            reference=None,
            reference_contexts=None,
            retrieved_contexts=None,
        ):
            captures.append(
                (self.with_reference, reference, reference_contexts, retrieved_contexts)
            )
            return SimpleNamespace(value=1.0)

    def factory(config):
        return [
            (
                selected,
                resolve_metric(selected.id),
                Rubric(selected.parameters.get("with_reference", False)),
            )
            for selected in config.metrics
        ], None

    config = RagasConfig(
        metrics=[
            MetricConfig(id="rubrics_score_without_reference"),
            MetricConfig(id="domain_specific_rubrics"),
            MetricConfig(
                id="domain_specific_rubrics", parameters={"with_reference": True}
            ),
        ],
        judge=EndpointProfile(model="fake"),
    )
    run_id = await evaluate_ragas(
        source.experiment_id,
        retrieval_id,
        generation_id,
        config,
        object_store=source.store,
        metric_factory=factory,
    )
    assert (await crud.get_record(StageRun, run_id)).status == "completed"
    assert len(captures) == 12
    for with_reference, reference, gold_contexts, retrieved_contexts in captures:
        assert reference == (REFERENCE if with_reference else None)
        assert gold_contexts is None
        assert retrieved_contexts is None


async def test_unrelated_image_failure_does_not_fail_answer_metrics_and_resume_skips_completed(
    pipeline_source,
):
    source = pipeline_source
    retrieval_id = await saved_retrieval(source)
    generation_id = await run_generation(
        retrieval_id,
        GenerationConfig(
            endpoint=EndpointProfile(model="fake"),
            context=ContextConfig(representations=["original_image"], page_top_k=2),
        ),
        object_store=source.store,
        completion=completion_recorder([]),
    )
    judge_calls = []

    class ImageMetric:
        async def ascore(self, response, retrieved_contexts):
            assert len(retrieved_contexts) == 2
            assert all(
                value.startswith("data:image/png;base64,")
                for value in retrieved_contexts
            )
            judge_calls.append(response)
            return SimpleNamespace(value=1.0)

    class FailOnceStore:
        calls = 0

        async def read_asset(self, asset):
            self.calls += 1
            if self.calls == 1:
                raise OSError("Simulated unavailable image")
            return await source.store.read_asset(asset)

    def factory(config):
        exact_entries, _ = create_metrics(
            RagasConfig(metrics=[MetricConfig(id="exact_match")])
        )
        return [
            exact_entries[0],
            (config.metrics[1], resolve_metric(config.metrics[1].id), ImageMetric()),
        ], None

    store = FailOnceStore()
    config = RagasConfig(
        metrics=[
            MetricConfig(id="exact_match"),
            MetricConfig(id="multi_modal_faithfulness"),
        ],
        judge=EndpointProfile(model="fake"),
        ragas_max_concurrency=1,
    )
    run_id = await evaluate_ragas(
        source.experiment_id,
        retrieval_id,
        generation_id,
        config,
        object_store=store,
        metric_factory=factory,
    )
    assert (await crud.get_record(StageRun, run_id)).status == "partial"
    rows = await crud.find_records(MetricResult, evaluation_run_id=run_id)
    assert all(
        row.status == "completed" for row in rows if row.metric_id == "exact_match"
    )
    assert sum(row.status == "failed" for row in rows) == 1
    aggregates = await crud.find_records(
        MetricAggregate, evaluation_run_id=run_id, group_by="dataset"
    )
    multimodal = next(
        row for row in aggregates if row.metric_id == "multi_modal_faithfulness"
    )
    assert (
        multimodal.expected_count,
        multimodal.scored_count,
        multimodal.failed_count,
    ) == (4, 3, 1)
    before_reads, before_judges = store.calls, len(judge_calls)
    assert (
        await evaluate_ragas(
            source.experiment_id,
            retrieval_id,
            generation_id,
            config,
            resume_run_id=run_id,
            object_store=store,
            metric_factory=factory,
        )
        == run_id
    )
    assert (await crud.get_record(StageRun, run_id)).status == "completed"
    assert store.calls - before_reads == 2
    assert len(judge_calls) - before_judges == 1


async def test_exact_match_never_reads_image_assets(pipeline_source):
    source = pipeline_source
    retrieval_id = await saved_retrieval(source)
    generation_id = await run_generation(
        retrieval_id,
        GenerationConfig(
            endpoint=EndpointProfile(model="fake"),
            context=ContextConfig(representations=["original_image"], page_top_k=2),
        ),
        object_store=source.store,
        completion=completion_recorder([]),
    )

    class OfflineStore:
        calls = 0

        async def read_asset(self, asset):
            self.calls += 1
            raise AssertionError("ExactMatch does not need page assets")

    store = OfflineStore()
    run_id = await evaluate_ragas(
        source.experiment_id,
        retrieval_id,
        generation_id,
        RagasConfig(metrics=[MetricConfig(id="exact_match")]),
        object_store=store,
    )
    assert (await crud.get_record(StageRun, run_id)).status == "completed"
    assert store.calls == 0
