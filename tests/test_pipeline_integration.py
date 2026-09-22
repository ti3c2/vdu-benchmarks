"""Full orchestration against real stores, with cached OCR and local model doubles."""

import json
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
    SparseConfig,
)
from src.etls.text_transforms import remove_tables
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
    MetricAggregate,
    PageRepresentation,
    StageRun,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("VDU_INTEGRATION") != "1",
        reason="requires PostgreSQL, MinIO, and Qdrant",
    ),
]


@pytest.mark.parametrize("mode", ["dense", "sparse"])
async def test_page_table_ablation_end_to_end(
    pipeline_source, monkeypatch, tmp_path, mode
):
    monkeypatch.chdir(tmp_path)
    source = pipeline_source
    calls = []

    async def fake_encode(model, config, values, role):
        calls.extend((role, value) for value in values)
        return [[1.0, 0.0] for _ in values]

    async def skip_preflight(*args, **kwargs):
        return None

    monkeypatch.setattr(vectorize, "encode_dense", fake_encode)
    monkeypatch.setattr(vectorize, "verify_embedding_endpoint", skip_preflight)
    monkeypatch.setattr(
        vectorize, "create_dense_model", lambda *args: MockEmbedding(embed_dim=2)
    )
    representations = await crud.find_records(
        PageRepresentation, run_id=source.preprocess_id
    )
    # A table-only page and a page larger than the chunk limit exercise both edges.
    await crud.update_record(
        PageRepresentation,
        representations[0].id,
        text="| Report | Value |\n| --- | --- |\n| Revenue | 42 |",
    )
    await crud.update_record(
        PageRepresentation,
        representations[1].id,
        text="Report prose. " * 500
        + "\n\n<table><tr><td>Secret cell</td></tr></table>",
    )
    originals = {
        str(row.corpus_id): row.text
        for row in await crud.find_records(
            PageRepresentation, run_id=source.preprocess_id
        )
    }
    config = ExperimentConfig(
        name=f"page-{mode}",
        dataset_id=source.dataset_id,
        corpus_unit="page",
        embeddings=EmbeddingConfig(
            dense=DenseConfig(
                endpoint=EndpointProfile(model="fixture-dense"),
                dimensions=2,
                space_id="fixture-space",
            )
            if mode == "dense"
            else None,
            sparse=SparseConfig() if mode == "sparse" else None,
        ),
        retrieval=RetrievalConfig(mode=mode, page_top_k=4),
        ir=IRConfig(cutoffs=[1, 4]),
        reuse={"representations": source.preprocess_id},
    )
    client = AsyncQdrantClient(url=get_settings().qdrant_url)
    variants = []
    try:
        for include_tables in (True, False):
            config.page_include_tables = include_tables
            experiment_id = await pipeline.run_experiment(config)
            runs = {
                row.role: row.run_id
                for row in await crud.find_records(
                    ExperimentRun, experiment_id=experiment_id
                )
            }
            variants.append(runs)
            assert "chunks" not in runs
            embedding = await crud.get_record(EmbeddingRun, runs["corpus_embeddings"])
            expected = {
                page_id: text if include_tables else remove_tables(text)
                for page_id, text in originals.items()
                if include_tables or remove_tables(text).strip()
            }
            assert embedding.unit_kind == "page"
            assert embedding.profiles["include_tables"] == include_tables
            assert (
                embedding.point_count == len(expected) == (4 if include_tables else 3)
            )
            points, _ = await client.scroll(
                embedding.collection_name, limit=10, with_payload=True
            )
            assert {
                str(point.id): json.loads(point.payload["_node_content"])["text"]
                for point in points
            } == expected
            metrics = await crud.find_records(
                MetricAggregate, evaluation_run_id=runs["ir"]
            )
            assert metrics and all(metric.failed_count == 0 for metric in metrics)
            assert (
                await crud.get_record(Experiment, experiment_id)
            ).status == "completed"
            assert next(
                (tmp_path / "data/experiments").glob(f"*_{experiment_id}_results.json")
            )

        assert variants[0]["query_embeddings"] == variants[1]["query_embeddings"]
        assert variants[0]["corpus_embeddings"] != variants[1]["corpus_embeddings"]
        assert {
            str(row.corpus_id): row.text
            for row in await crud.find_records(
                PageRepresentation, run_id=source.preprocess_id
            )
        } == originals
        if mode == "dense":
            assert sum(role == "query" for role, _ in calls) == len(source.queries)
            assert [value for role, value in calls if role == "corpus"] == [
                text
                for include_tables in (True, False)
                for _, text in sorted(
                    (page_id, text if include_tables else remove_tables(text))
                    for page_id, text in originals.items()
                    if include_tables or remove_tables(text).strip()
                )
            ]

        config.reuse["corpus_embeddings"] = variants[0]["corpus_embeddings"]
        with pytest.raises(
            RuntimeError, match="Reused corpus_embeddings configuration differs"
        ):
            await pipeline.run_experiment(config)
    finally:
        for embedding in await crud.find_records(
            EmbeddingRun, dataset_id=source.dataset_id
        ):
            if await client.collection_exists(embedding.collection_name):
                await client.delete_collection(embedding.collection_name)
        await client.close()


async def test_complete_experiments_reuse_stages_and_compare(
    pipeline_source, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
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
        report_path = next(
            (tmp_path / "data/experiments").glob(f"*_{first_id}_results.json")
        )
        report = json.loads(report_path.read_text())
        assert report["status"] == "completed"
        assert report["queries"] and report["queries"][0]["metrics"]
        assert report["queries"][0]["answer_text"] is not None
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
