import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest

from src.config import (
    DenseConfig,
    EmbeddingConfig,
    EndpointProfile,
    ExperimentConfig,
    PageEmbeddingConfig,
    RetrievalConfig,
    SparseConfig,
)
from src.orchestrate import pipeline
from src.stor_rel.schema import (
    Dataset,
    Experiment,
    ExperimentQuery,
    ExperimentRun,
    Query,
    RunDependency,
    RunItem,
    StageRun,
)


@pytest.fixture
def memory(monkeypatch):
    rows = {}

    async def save(model, **values):
        values.setdefault("id", uuid4())
        record = SimpleNamespace(**values)
        rows.setdefault(model, {})[record.id] = record
        return record

    async def get(model, id):
        try:
            return rows[model][id]
        except KeyError as exc:
            raise ValueError("missing record") from exc

    async def find(model, **filters):
        return [
            record
            for record in rows.get(model, {}).values()
            if all(
                getattr(record, key, None) == value for key, value in filters.items()
            )
        ]

    async def update(model, id, **values):
        record = await get(model, id)
        for key, value in values.items():
            setattr(record, key, value)
        return record

    async def validate(id, kind=None, dataset_id=None):
        record = await get(StageRun, id)
        if (
            record.status != "completed"
            or (kind and kind != record.kind)
            or (dataset_id and dataset_id != record.dataset_id)
        ):
            raise ValueError("invalid run")
        return record

    async def attach(experiment_id, run_id, role):
        existing = await find(ExperimentRun, experiment_id=experiment_id, role=role)
        if existing:
            assert existing[0].run_id == run_id
            return existing[0]
        experiment = await get(Experiment, experiment_id)
        return await save(
            ExperimentRun,
            experiment_id=experiment_id,
            run_id=run_id,
            role=role,
            dataset_id=experiment.dataset_id,
        )

    for name, function in (
        ("save_record", save),
        ("get_record", get),
        ("find_records", find),
        ("update_record", update),
        ("validate_run", validate),
        ("attach_experiment_run", attach),
    ):
        monkeypatch.setattr(pipeline.crud, name, function)
    return SimpleNamespace(rows=rows, save=save, get=get, find=find)


async def fixture_config(memory):
    dataset = await memory.save(Dataset, status="completed")
    for original_id, level in (("b", 0), ("a", 1), ("a", 0)):
        await memory.save(
            Query,
            dataset_id=dataset.id,
            original_id=original_id,
            rephrase_level=level,
            language="en",
        )
    return ExperimentConfig(
        name="test",
        dataset_id=dataset.id,
        corpus_unit="page",
        embeddings=EmbeddingConfig(
            dense=DenseConfig(
                endpoint=EndpointProfile(model="fixture"),
                space_id="shared",
                modality="image",
                adapter="vllm",
            )
        ),
        retrieval=RetrievalConfig(page_top_k=2),
        ir=None,
    )


async def produced_run(memory, dataset_id, kind, config, status="completed"):
    run = await memory.save(
        StageRun,
        dataset_id=dataset_id,
        kind=kind,
        config=config.model_dump(mode="json"),
        status=status,
        error="fixture failure" if status == "failed" else None,
    )
    observer = pipeline.crud.run_observer.get()
    if observer:
        await observer(run)
    return run.id


@pytest.mark.parametrize("export_fails", [False, True])
async def test_experiment_freezes_stable_cohort_and_parallel_embeddings(
    memory, monkeypatch, export_fails
):
    config = await fixture_config(memory)
    config.queries.limit = 2
    events = []
    query_started = asyncio.Event()
    corpus_started = asyncio.Event()

    async def queries(dataset_id, embedding, query_ids):
        query_started.set()
        await corpus_started.wait()
        selected = [await memory.get(Query, id) for id in query_ids]
        assert [(query.original_id, query.rephrase_level) for query in selected] == [
            ("a", 0),
            ("a", 1),
        ]
        events.append("queries")
        return await produced_run(memory, dataset_id, "embed_queries", embedding)

    async def pages(dataset_id, embedding, representation_run_id):
        assert representation_run_id is None
        corpus_started.set()
        await query_started.wait()
        events.append("pages")
        return await produced_run(memory, dataset_id, "embed_pages", embedding)

    async def retrieval(query_embedding, corpus_embedding, retrieval_config):
        assert len(events) == 2
        return await produced_run(
            memory, config.dataset_id, "retrieval", retrieval_config
        )

    exported = []

    async def export(ids):
        assert (await memory.get(Experiment, ids[0])).status == "completed"
        exported.extend(ids)
        if export_fails:
            raise OSError("disk full")

    monkeypatch.setattr(pipeline, "vectorize_queries", queries)
    monkeypatch.setattr(pipeline, "vectorize_pages", pages)
    monkeypatch.setattr(pipeline, "run_retrieval", retrieval)
    monkeypatch.setattr(pipeline, "export_experiments", export)
    if export_fails:
        with pytest.raises(RuntimeError, match="completed, but results export failed"):
            await pipeline.run_experiment(config)
        experiment_id = exported[0]
    else:
        experiment_id = await pipeline.run_experiment(config)
    assert exported == [experiment_id]
    assert (await memory.get(Experiment, experiment_id)).status == "completed"
    assert len(await memory.find(ExperimentQuery, experiment_id=experiment_id)) == 2
    assert {
        record.role
        for record in await memory.find(ExperimentRun, experiment_id=experiment_id)
    } == {"query_embeddings", "corpus_embeddings", "retrieval"}


async def test_failed_branch_stays_attached_and_sibling_finishes(memory, monkeypatch):
    config = await fixture_config(memory)
    sibling_finished = []

    async def queries(dataset_id, embedding, query_ids):
        await produced_run(
            memory, dataset_id, "embed_queries", embedding, status="failed"
        )
        raise ValueError("query service unavailable")

    async def pages(dataset_id, embedding, representation_run_id):
        await asyncio.sleep(0)
        result = await produced_run(memory, dataset_id, "embed_pages", embedding)
        sibling_finished.append(result)
        return result

    monkeypatch.setattr(pipeline, "vectorize_queries", queries)
    monkeypatch.setattr(pipeline, "vectorize_pages", pages)
    with pytest.raises(RuntimeError, match="query service unavailable"):
        await pipeline.run_experiment(config)
    experiment = next(iter(memory.rows[Experiment].values()))
    assert experiment.status == "failed" and sibling_finished
    assert {
        record.role
        for record in await memory.find(ExperimentRun, experiment_id=experiment.id)
    } == {"query_embeddings", "corpus_embeddings"}


async def test_reuse_rejects_changed_configuration_before_models(memory):
    config = await fixture_config(memory)
    existing = await memory.save(
        StageRun,
        dataset_id=config.dataset_id,
        kind="embed_queries",
        config={"different": "encoder"},
        status="completed",
    )
    config.reuse["query_embeddings"] = existing.id
    with pytest.raises(RuntimeError, match="configuration differs"):
        await pipeline.run_experiment(config)
    experiment = next(iter(memory.rows[Experiment].values()))
    assert experiment.status == "failed"
    assert len(await memory.find(ExperimentRun, experiment_id=experiment.id)) == 1


async def test_query_resume_uses_frozen_selection(memory, monkeypatch):
    config = await fixture_config(memory)
    selected = [uuid4(), uuid4()]
    source = await memory.save(
        StageRun,
        dataset_id=config.dataset_id,
        kind="embed_queries",
        config=config.embeddings.model_dump(mode="json"),
        status="failed",
        provenance={"selection": list(map(str, selected))},
    )

    async def resume(dataset_id, embedding, query_ids, resume_run_id):
        assert dataset_id == config.dataset_id
        assert query_ids == selected and resume_run_id == source.id
        source.status = "completed"
        return source.id

    monkeypatch.setattr(pipeline, "vectorize_queries", resume)
    assert await pipeline.run_resume(source.id) == source.id


async def test_page_resume_retains_table_policy(memory, monkeypatch):
    dataset_id, representation_id = uuid4(), uuid4()
    source = await memory.save(
        StageRun,
        dataset_id=dataset_id,
        kind="embed_pages",
        config=PageEmbeddingConfig(
            sparse=SparseConfig(), include_tables=False
        ).model_dump(mode="json"),
        status="failed",
    )
    await memory.save(
        RunDependency,
        run_id=source.id,
        role="representations",
        input_run_id=representation_id,
    )

    async def resume(dataset_id, embedding, representation_run_id, resume_run_id):
        assert dataset_id == source.dataset_id
        assert representation_run_id == representation_id
        assert resume_run_id == source.id
        assert embedding.include_tables is False
        source.status = "completed"
        return source.id

    monkeypatch.setattr(pipeline, "vectorize_pages", resume)
    assert await pipeline.run_resume(source.id) == source.id


def test_table_exclusion_rejects_chunk_and_image_experiments():
    with pytest.raises(ValueError, match="requires corpus_unit: page"):
        ExperimentConfig(
            name="invalid",
            dataset_id=uuid4(),
            embeddings=EmbeddingConfig(sparse=SparseConfig()),
            page_include_tables=False,
        )
    with pytest.raises(ValueError, match="requires text page embeddings"):
        PageEmbeddingConfig(
            dense=DenseConfig(
                endpoint=EndpointProfile(model="image"),
                space_id="image",
                modality="image",
                adapter="vllm",
            ),
            include_tables=False,
        )


async def test_comparison_reports_incompatible_cohorts(memory):
    dataset_id = uuid4()
    first = await memory.save(
        Experiment, dataset_id=dataset_id, name="first", status="completed", config={}
    )
    second = await memory.save(
        Experiment, dataset_id=dataset_id, name="second", status="completed", config={}
    )
    await memory.save(ExperimentQuery, experiment_id=first.id, query_id=uuid4())
    await memory.save(ExperimentQuery, experiment_id=second.id, query_id=uuid4())
    comparison = await pipeline.compare_experiments([first.id, second.id])
    assert comparison["compatible"] is False
    assert any(
        "query cohorts differ" in reason
        for reason in comparison["compatibility_reasons"]
    )


async def test_list_datasets_includes_datetimes_snapshot_fields_and_metadata(memory):
    created_at = datetime(2026, 9, 14, 9, 0, tzinfo=UTC)
    completed = await memory.save(
        Dataset,
        source="fixture/source",
        subset="reports",
        split="test",
        revision="main",
        fingerprint="abc123",
        status="completed",
        metadata_json={"documents": 1},
        created_at=created_at,
        updated_at=created_at,
    )
    await memory.save(
        Dataset,
        source="fixture/source",
        subset="reports",
        split="train",
        revision="main",
        fingerprint="def456",
        status="failed",
        metadata_json={},
        created_at=created_at,
        updated_at=created_at,
    )

    listing = await pipeline.list_datasets(status="completed", source="fixture/source")

    assert listing == {
        "datasets": [
            {
                "dataset_id": str(completed.id),
                "source": "fixture/source",
                "subset": "reports",
                "split": "test",
                "revision": "main",
                "fingerprint": "abc123",
                "status": "completed",
                "created_at": created_at.isoformat(),
                "updated_at": created_at.isoformat(),
                "metadata": {"documents": 1},
            }
        ]
    }


async def test_list_experiments_includes_datetimes_configs_and_runs(memory):
    dataset_id = uuid4()
    created_at = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)
    updated_at = datetime(2026, 9, 13, 12, 30, tzinfo=UTC)
    experiment = await memory.save(
        Experiment,
        dataset_id=dataset_id,
        name="dense",
        status="completed",
        config={"retrieval": {"mode": "dense"}},
        created_at=created_at,
        updated_at=updated_at,
    )
    await memory.save(
        Experiment,
        dataset_id=dataset_id,
        name="failed",
        status="failed",
        config={},
        created_at=created_at,
        updated_at=updated_at,
    )
    await memory.save(
        Experiment,
        dataset_id=uuid4(),
        name="other",
        status="completed",
        config={},
        created_at=created_at,
        updated_at=updated_at,
    )
    await memory.save(
        ExperimentQuery,
        dataset_id=dataset_id,
        experiment_id=experiment.id,
        query_id=uuid4(),
    )
    run = await memory.save(
        StageRun,
        dataset_id=dataset_id,
        kind="retrieval",
        status="completed",
        started_at=created_at,
        finished_at=updated_at,
    )
    await memory.save(
        ExperimentRun,
        dataset_id=dataset_id,
        experiment_id=experiment.id,
        run_id=run.id,
        role="retrieval",
    )

    listing = await pipeline.list_experiments(dataset_id=dataset_id, status="completed")

    assert listing == {
        "experiments": [
            {
                "experiment_id": str(experiment.id),
                "name": "dense",
                "dataset_id": str(dataset_id),
                "status": "completed",
                "created_at": created_at.isoformat(),
                "updated_at": updated_at.isoformat(),
                "query_count": 1,
                "runs": {
                    "retrieval": {
                        "run_id": str(run.id),
                        "kind": "retrieval",
                        "status": "completed",
                        "started_at": created_at.isoformat(),
                        "finished_at": updated_at.isoformat(),
                    }
                },
                "config": {"retrieval": {"mode": "dense"}},
            }
        ]
    }


async def test_reused_query_cohort_mismatch_is_rejected(memory):
    config = await fixture_config(memory)
    existing = await memory.save(
        StageRun,
        dataset_id=config.dataset_id,
        kind="embed_queries",
        config=config.embeddings.model_dump(mode="json"),
        status="completed",
    )
    await memory.save(RunItem, run_id=existing.id, query_id=uuid4(), status="completed")
    config.reuse["query_embeddings"] = existing.id
    with pytest.raises(RuntimeError, match="different query cohort"):
        await pipeline.run_experiment(config)


async def test_ragas_context_requirements_are_checked_before_models(memory):
    from src.config import ContextConfig, GenerationConfig, MetricConfig, RagasConfig

    config = await fixture_config(memory)
    config.generation = GenerationConfig(
        endpoint=EndpointProfile(model="mock"),
        context=ContextConfig(page_top_k=2, representations=["image"]),
    )
    config.ragas = RagasConfig(
        judge=EndpointProfile(model="judge"), metrics=[MetricConfig(id="faithfulness")]
    )
    with pytest.raises(ValueError, match="requires text generation context"):
        await pipeline.run_experiment(config)
    assert not memory.rows.get(Experiment)


async def test_show_evaluation_run_includes_metric_failures_and_skips(memory):
    from src.stor_rel.schema import MetricResult

    run = await memory.save(
        StageRun,
        dataset_id=uuid4(),
        kind="evaluate_ragas",
        status="partial",
        config={},
        provenance={},
        metadata_json={},
        expected_count=3,
        completed_count=1,
        failed_count=1,
        error=None,
    )
    query_id = uuid4()
    for status, metric, error, reason in (
        ("failed", "faithfulness", "judge timed out", None),
        ("skipped", "exact_match", None, "Reference answer is missing"),
        ("completed", "string_presence", None, None),
    ):
        await memory.save(
            MetricResult,
            evaluation_run_id=run.id,
            query_id=query_id,
            metric_id=metric,
            status=status,
            error=error,
            reason=reason,
        )
    details = await pipeline.show_run(run.id)
    assert details["failed_items"] == []
    assert details["failed_metrics"] == [
        {
            "query_id": str(query_id),
            "metric_id": "faithfulness",
            "error": "judge timed out",
            "reason": None,
        }
    ]
    assert details["skipped_metrics"] == [
        {
            "query_id": str(query_id),
            "metric_id": "exact_match",
            "error": None,
            "reason": "Reference answer is missing",
        }
    ]


async def test_comparison_discovers_standalone_evaluations_and_config_variants(memory):
    from src.stor_rel.schema import EvaluationRun, MetricAggregate

    dataset_id, query_id = uuid4(), uuid4()
    experiments = [
        await memory.save(
            Experiment,
            dataset_id=dataset_id,
            name=name,
            status="completed",
            config={"ir": None},
        )
        for name in ("first", "second")
    ]
    for experiment in experiments:
        await memory.save(
            ExperimentQuery, experiment_id=experiment.id, query_id=query_id
        )

    async def standalone(experiment, cutoffs):
        run = await memory.save(
            StageRun,
            dataset_id=dataset_id,
            kind="evaluate_ir",
            status="completed",
            config={
                "experiment_id": str(experiment.id),
                "cutoffs": cutoffs,
                "relevance_threshold": 1,
            },
            provenance={
                "implementation_sha256": "fixture-v1",
                "selection": [str(query_id)],
            },
        )
        await memory.save(
            EvaluationRun,
            id=run.id,
            experiment_id=experiment.id,
            framework="ir_measures",
            retrieval_run_id=uuid4(),
            generation_run_id=None,
        )
        await memory.save(
            MetricAggregate,
            evaluation_run_id=run.id,
            metric_id=f"P@{max(cutoffs)}",
            group_by="dataset",
            group_value=str(dataset_id),
            value=1.0,
            expected_count=1,
            scored_count=1,
            skipped_count=0,
            failed_count=0,
        )
        return run.id

    first_run = await standalone(experiments[0], [1])
    second_run = await standalone(experiments[1], [1])
    assert not memory.rows.get(ExperimentRun)
    comparison = await pipeline.compare_experiments([item.id for item in experiments])
    assert comparison["compatible"]
    first, second = comparison["experiments"]
    assert first["metrics"][0]["evaluation_run_id"] == str(first_run)
    assert second["metrics"][0]["evaluation_run_id"] == str(second_run)
    assert (
        first["metrics"][0]["evaluation_key"] == second["metrics"][0]["evaluation_key"]
    )
    assert (
        first["evaluations"][0]["provenance"]["implementation_sha256"] == "fixture-v1"
    )
    assert "experiment_id" not in first["evaluations"][0]["config"]

    variant_run = await standalone(experiments[0], [1, 2])
    comparison = await pipeline.compare_experiments([item.id for item in experiments])
    assert not comparison["compatible"]
    assert any(
        "evaluator configurations differ" in reason
        for reason in comparison["compatibility_reasons"]
    )
    first = comparison["experiments"][0]
    assert len(first["evaluations"]) == len(first["evaluator_configs"]) == 2
    assert {metric["evaluation_run_id"] for metric in first["metrics"]} == {
        str(first_run),
        str(variant_run),
    }
    assert len({metric["evaluation_key"] for metric in first["metrics"]}) == 2
