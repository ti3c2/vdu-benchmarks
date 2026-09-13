"""Full orchestration against real stores, with cached OCR and local model doubles."""

import os
from uuid import UUID

import pytest
from llama_index.core.embeddings import MockEmbedding
from qdrant_client import AsyncQdrantClient
from test_generation_evaluation import completion_recorder
from test_generation_evaluation import pipeline_source as pipeline_source
from test_storage import storage_db as storage_db

from src.config import (
    ChunkConfig,
    ContextConfig,
    DenseConfig,
    EmbeddingConfig,
    EndpointProfile,
    ExperimentConfig,
    GenerationConfig,
    IRConfig,
    MetricConfig,
    RagasConfig,
    RetrievalConfig,
)
from src.evaluate import vectorize
from src.evaluate.generation import run_generation
from src.orchestrate import pipeline
from src.settings import get_settings
from src.stor_rel import crud
from src.stor_rel.schema import (
    EmbeddingRun,
    Experiment,
    ExperimentQuery,
    ExperimentRun,
    StageRun,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("VDU_INTEGRATION") != "1",
        reason="requires PostgreSQL, MinIO, and Qdrant",
    ),
]


async def test_complete_experiments_reuse_stages_and_compare(
    pipeline_source, monkeypatch
):
    source = pipeline_source
    embedding_calls = []
    generation_calls = []

    async def fake_encode(model, config, values, role):
        embedding_calls.extend((role, value) for value in values)
        return [[1.0, 0.0] for _ in values]

    async def generate(retrieval_run_id, config):
        return await run_generation(
            retrieval_run_id,
            config,
            object_store=source.store,
            completion=completion_recorder(generation_calls),
        )

    async def unexpected_ocr(*args, **kwargs):
        raise AssertionError("Experiments must reuse cached OCR")

    async def skip_preflight(*args, **kwargs):
        return None

    monkeypatch.setattr(
        vectorize, "create_dense_model", lambda *args: MockEmbedding(embed_dim=2)
    )
    monkeypatch.setattr(vectorize, "encode_dense", fake_encode)
    monkeypatch.setattr(vectorize, "verify_embedding_endpoint", skip_preflight)
    monkeypatch.setattr(pipeline, "run_generation", generate)
    monkeypatch.setattr(pipeline, "preprocess_pages", unexpected_ocr)
    config = ExperimentConfig(
        name="first complete fixture",
        dataset_id=source.dataset_id,
        chunking=ChunkConfig(max_chars=600),
        embeddings=EmbeddingConfig(
            dense=DenseConfig(
                endpoint=EndpointProfile(model="fixture-dense"),
                dimensions=2,
                space_id="fixture-space",
            ),
            batch_size=3,
        ),
        retrieval=RetrievalConfig(mode="dense", page_top_k=4),
        generation=GenerationConfig(
            endpoint=EndpointProfile(model="fixture-generation"),
            context=ContextConfig(
                representations=["ocr", "original_image"], page_top_k=2
            ),
        ),
        ir=IRConfig(cutoffs=[1, 2, 4]),
        ragas=RagasConfig(
            metrics=[MetricConfig(id="exact_match"), MetricConfig(id="string_presence")]
        ),
        reuse={
            "representations": source.preprocess_id,
            "chunks": source.chunks_id,
        },
    )
    client = AsyncQdrantClient(url=get_settings().qdrant_url)
    try:
        first_id = await pipeline.run_experiment(config)
        first_runs = {
            association.role: association.run_id
            for association in await crud.find_records(
                ExperimentRun, experiment_id=first_id
            )
        }
        expected_roles = {
            "representations",
            "chunks",
            "query_embeddings",
            "corpus_embeddings",
            "retrieval",
            "generation",
            "ir",
            "ragas",
        }
        assert set(first_runs) == expected_roles
        assert (await crud.get_record(Experiment, first_id)).status == "completed"
        assert len(generation_calls) == len(source.queries)
        assert sum(role == "query" for role, _ in embedding_calls) == len(
            source.queries
        )
        assert sum(role == "corpus" for role, _ in embedding_calls) == len(
            source.chunks
        )
        calls_after_first = (len(embedding_calls), len(generation_calls))
        saved = await crud.get_record(Experiment, first_id)
        assert (
            UUID(saved.config["generation"]["context"]["representation_run_id"])
            == source.preprocess_id
        )

        second_config = config.model_copy(deep=True)
        second_config.name = "second complete fixture"
        second_id = await pipeline.run_experiment(second_config)
        second_runs = {
            association.role: association.run_id
            for association in await crud.find_records(
                ExperimentRun, experiment_id=second_id
            )
        }
        assert first_id != second_id
        assert (len(embedding_calls), len(generation_calls)) == calls_after_first
        for role in expected_roles - {"ir", "ragas"}:
            assert first_runs[role] == second_runs[role]
        for role in ("ir", "ragas"):
            assert first_runs[role] != second_runs[role]
        for experiment_id in (first_id, second_id):
            selection = await crud.find_records(
                ExperimentQuery, experiment_id=experiment_id
            )
            assert {item.query_id for item in selection} == {
                q.id for q in source.queries
            }
        for run_id in second_runs.values():
            assert (await crud.get_record(StageRun, run_id)).status == "completed"

        comparison = await pipeline.compare_experiments([first_id, second_id])
        assert comparison["compatible"], comparison["compatibility_reasons"]
        assert comparison["configuration_differences"] == {}
        first_metrics, second_metrics = [
            [
                {
                    key: value
                    for key, value in metric.items()
                    if key != "evaluation_run_id"
                }
                for metric in row["metrics"]
            ]
            for row in comparison["experiments"]
        ]
        metric_order = lambda metric: (
            metric["framework"],
            metric["metric_id"],
            metric["group_by"],
            metric["group_value"],
        )
        assert sorted(first_metrics, key=metric_order) == sorted(
            second_metrics, key=metric_order
        )
        assert {metric["framework"] for metric in first_metrics} == {"ir", "ragas"}
        assert all(metric["failed_count"] == 0 for metric in first_metrics)
        assert all(
            metric["value"] == 1.0
            for metric in first_metrics
            if metric["framework"] == "ragas"
        )
    finally:
        for embedding in await crud.find_records(
            EmbeddingRun, dataset_id=source.dataset_id
        ):
            if await client.collection_exists(embedding.collection_name):
                await client.delete_collection(embedding.collection_name)
        await client.close()
