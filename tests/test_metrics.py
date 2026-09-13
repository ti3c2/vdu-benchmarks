"""Metric catalog and adapters exercise real local scorers without model calls."""

import inspect
from pathlib import Path

import pytest

from src.config import (
    DenseConfig,
    EndpointProfile,
    ExperimentConfig,
    MetricConfig,
    RagasConfig,
    load_config,
)
from src.evaluate.metrics import METRICS, metric_catalog
from src.evaluate.ragas import create_metrics, validate_metrics


def test_catalog_covers_pinned_ragas_exports():
    import ragas.metrics.collections as collections

    exported = {
        name
        for name in collections.__all__
        if inspect.isclass(getattr(collections, name))
    }
    documented = {spec.implementation for spec in METRICS.values()}
    assert exported <= documented | {
        "BaseMetric",
        "DistanceMeasure",
        "ContextPrecision",
        "ContextUtilization",
        "AgentGoalAccuracy",
    }
    assert metric_catalog()["ragas_version"] == "0.4.3"


@pytest.mark.parametrize(
    "metric_id",
    [
        "exact_match",
        "string_presence",
        "non_llm_string_similarity",
        "bleu_score",
        "chrf_score",
        "rouge_score",
    ],
)
async def test_classical_metric_adapters(metric_id):
    config = RagasConfig(metrics=[MetricConfig(id=metric_id)])
    entries, client = create_metrics(config)
    assert client is None
    response = "The annual report contains financial results for the entire year."
    result = await entries[0][2].ascore(response=response, reference=response)
    assert result.value == pytest.approx(1)


async def test_string_similarity_accepts_yaml_distance_name():
    entries, _ = create_metrics(
        RagasConfig(
            metrics=[
                MetricConfig(
                    id="non_llm_string_similarity",
                    parameters={"distance_measure": "levenshtein"},
                )
            ]
        )
    )
    result = await entries[0][2].ascore(response="abc", reference="abc")
    assert result.value == 1


@pytest.mark.parametrize(
    "metric_id", ["id_based_context_precision", "id_based_context_recall"]
)
async def test_legacy_id_metric_adapters(metric_id):
    from ragas import SingleTurnSample

    entries, client = create_metrics(RagasConfig(metrics=[MetricConfig(id=metric_id)]))
    assert client is None
    result = await entries[0][2].single_turn_ascore(
        SingleTurnSample(
            retrieved_context_ids=["page-a"], reference_context_ids=["page-a"]
        )
    )
    assert result == 1


def test_metric_prerequisites_and_alias_duplicates():
    with pytest.raises(ValueError, match="judge profile"):
        validate_metrics(RagasConfig(metrics=[MetricConfig(id="faithfulness")]))
    with pytest.raises(ValueError, match="embedding profile"):
        validate_metrics(RagasConfig(metrics=[MetricConfig(id="semantic_similarity")]))
    with pytest.raises(ValueError, match="Duplicate"):
        validate_metrics(
            RagasConfig(
                metrics=[
                    MetricConfig(id="context_precision"),
                    MetricConfig(id="context_precision_with_reference"),
                ],
                judge=EndpointProfile(model="fake"),
            )
        )
    with pytest.raises(ValueError, match="conversation"):
        validate_metrics(RagasConfig(metrics=[MetricConfig(id="tool_call_accuracy")]))


def test_example_experiments_validate():
    paths = sorted((Path(__file__).parents[1] / "configs").glob("*.yaml"))
    assert len(paths) >= 4
    for path in paths:
        assert load_config(path, ExperimentConfig).name


def test_image_model_requires_explicit_wire_adapter():
    with pytest.raises(ValueError, match="vllm"):
        DenseConfig(
            endpoint=EndpointProfile(model="image"),
            space_id="image-space",
            modality="image",
        )


async def test_evaluator_vllm_instructions_applied_once_sync_and_async(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    from src.evaluate.embeddings import VLLMEmbedding

    profile = DenseConfig(
        endpoint=EndpointProfile(model="fake", max_retries=1),
        space_id="fixture",
        adapter="vllm",
        document_instruction="DOC: ",
    )
    model = VLLMEmbedding(profile)
    response = SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0])])
    model._sync_client.post = Mock(return_value=response)
    model._async_client.post = AsyncMock(return_value=response)
    monkeypatch.setattr(
        "src.evaluate.embeddings.create_dense_model", lambda config: model
    )
    try:
        entries, _ = create_metrics(
            RagasConfig(
                metrics=[MetricConfig(id="semantic_similarity")], embeddings=profile
            )
        )
        metric = entries[0][2]
        assert (
            await metric.ascore(response="answer", reference="reference")
        ).value == 1
        await metric.embeddings.aembed_text("async")
        sync_content = [
            call.kwargs["body"]["messages"][0]["content"][0]["text"]
            for call in model._sync_client.post.call_args_list
        ]
        async_content = [
            call.kwargs["body"]["messages"][0]["content"][0]["text"]
            for call in model._async_client.post.call_args_list
        ]
        assert sync_content == ["DOC: reference", "DOC: answer"]
        assert async_content == ["DOC: async"]
    finally:
        await model.aclose()
