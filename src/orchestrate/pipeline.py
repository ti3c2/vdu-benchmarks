"""Compose independently resumable stages and freeze experiment comparisons."""

import asyncio
import hashlib
import json
from uuid import UUID

from sqlalchemy import delete, select

from src.config import (
    ChunkConfig,
    EmbeddingConfig,
    ExperimentConfig,
    GenerationConfig,
    IRConfig,
    PreprocessConfig,
    RagasConfig,
    RetrievalConfig,
)
from src.etls.chunks import build_chunks
from src.etls.load_datasets import ingest_dataset
from src.etls.process_images import preprocess_pages
from src.evaluate.generation import run_generation
from src.evaluate.ir import evaluate_ir
from src.evaluate.metrics import resolve_metric
from src.evaluate.ragas import evaluate_ragas, validate_metrics
from src.evaluate.retrieval import run_retrieval
from src.evaluate.vector_store import create_vector_store
from src.evaluate.vectorize import vectorize_chunks, vectorize_pages, vectorize_queries
from src.stor_rel import crud
from src.stor_rel.entry import get_db
from src.stor_rel.schema import (
    Chunk,
    Dataset,
    EmbeddingRun,
    EvaluationRun,
    Experiment,
    ExperimentQuery,
    ExperimentRun,
    Generation,
    GenerationContext,
    GenerationRun,
    MetricAggregate,
    MetricResult,
    PageRepresentation,
    Query,
    Retrieval,
    RetrievalHit,
    RetrievalRun,
    RunDependency,
    RunItem,
    StageRun,
)


class IncompleteRunError(ValueError):
    def __init__(self, run):
        self.run_id = run.id
        self.status = run.status
        super().__init__(
            f"Run {run.id} ended {run.status}: {run.error or 'inspect run show for item failures'}"
        )


async def require_completed(run_id):
    run = await crud.get_record(StageRun, run_id)
    if run.status != "completed":
        raise IncompleteRunError(run)
    return run


def _same_artifact_config(saved: dict, wanted) -> bool:
    return crud.identity_config(saved) == crud.identity_config(wanted)


async def prepare_dataset(
    dataset_id: UUID, preprocess: PreprocessConfig, chunking: ChunkConfig | None = None
):
    representation_id = await preprocess_pages(dataset_id, preprocess)
    await require_completed(representation_id)
    chunk_id = await build_chunks(representation_id, chunking or ChunkConfig())
    await require_completed(chunk_id)
    return {
        "dataset_id": dataset_id,
        "representations": representation_id,
        "chunks": chunk_id,
    }


async def _experiment_stage(experiment, role, operation):
    async def attach(run):
        await crud.attach_experiment_run(experiment.id, run.id, role)

    token = crud.run_observer.set(attach)
    try:
        run_id = await operation
    finally:
        crud.run_observer.reset(token)
    await crud.attach_experiment_run(experiment.id, run_id, role)
    await require_completed(run_id)
    return run_id


async def run_experiment(config: ExperimentConfig) -> UUID:
    config = config.model_copy(deep=True)
    dataset = await crud.get_record(Dataset, config.dataset_id)
    if dataset.status != "completed":
        raise ValueError("Experiment requires a completed dataset ingestion")
    validate_metrics(config.ragas)
    if config.generation:
        context_kinds = set(config.generation.context.representations)
        image_kinds = {"image", "original_image"}
        for metric in config.ragas.metrics:
            spec = resolve_metric(metric.id)
            if (
                spec.modality == "text"
                and "retrieved_contexts" in spec.fields
                and context_kinds <= image_kinds
            ):
                raise ValueError(f"{spec.id} requires text generation context")
    if (
        config.corpus_unit == "chunk"
        and config.embeddings.dense
        and config.embeddings.dense.modality != "text"
    ):
        raise ValueError("Image embeddings require corpus_unit: page")
    if config.retrieval.mode in {"dense", "hybrid"} and config.embeddings.dense is None:
        raise ValueError("Dense or hybrid retrieval requires dense embeddings")
    if (
        config.retrieval.mode in {"sparse", "hybrid"}
        and config.embeddings.sparse is None
    ):
        raise ValueError("Sparse or hybrid retrieval requires sparse embeddings")
    allowed = {
        "representations",
        "chunks",
        "query_embeddings",
        "corpus_embeddings",
        "retrieval",
        "generation",
    }
    if set(config.reuse) - allowed:
        raise ValueError(f"Unknown reuse roles: {sorted(set(config.reuse) - allowed)}")
    if "generation" in config.reuse and config.generation is None:
        raise ValueError("Reusing generation requires its generation configuration")
    queries = sorted(
        [
            query
            for query in await crud.find_records(Query, dataset_id=dataset.id)
            if query.rephrase_level in config.queries.rephrase_levels
            and (
                config.queries.languages is None
                or query.language in config.queries.languages
            )
        ],
        key=lambda query: (query.original_id, query.rephrase_level),
    )
    if config.queries.limit is not None:
        queries = queries[: config.queries.limit]
    query_ids = [query.id for query in queries]
    if not query_ids:
        raise ValueError("Experiment query selection is empty")
    experiment = await crud.save_record(
        Experiment,
        dataset_id=dataset.id,
        name=config.name,
        config=config.model_dump(mode="json"),
        status="running",
    )
    for query_id in query_ids:
        await crud.save_record(
            ExperimentQuery,
            dataset_id=dataset.id,
            experiment_id=experiment.id,
            query_id=query_id,
        )
    chosen = dict(config.reuse)
    try:
        # A downstream reuse ID pins its own ancestors; explicit conflicts are errors.
        pending = list(chosen)
        checked = set()
        while pending:
            role = pending.pop()
            if role in checked:
                continue
            checked.add(role)
            await crud.attach_experiment_run(experiment.id, chosen[role], role)
            run = await crud.validate_run(chosen[role], dataset_id=dataset.id)
            dependency_roles = {
                "generation": {"retrieval": "retrieval"},
                "retrieval": {
                    "queries": "query_embeddings",
                    "corpus": "corpus_embeddings",
                },
                "corpus_embeddings": {
                    "chunks": "chunks",
                    "representations": "representations",
                },
                "chunks": {"representations": "representations"},
            }.get(role, {})
            for dependency in await crud.find_records(RunDependency, run_id=run.id):
                upstream = dependency_roles.get(dependency.role)
                if upstream is None:
                    continue
                if upstream in chosen and chosen[upstream] != dependency.input_run_id:
                    raise ValueError(
                        f"Reuse role {role} pins a different {upstream} run"
                    )
                chosen[upstream] = dependency.input_run_id
                pending.append(upstream)

        if (
            config.generation
            and "generation" in chosen
            and config.generation.context.representation_run_id is None
        ):
            saved = await crud.get_record(StageRun, chosen["generation"])
            saved_generation = GenerationConfig.model_validate(saved.config)
            config.generation.context.representation_run_id = (
                saved_generation.context.representation_run_id
            )
        if config.generation and config.generation.context.representation_run_id:
            await crud.validate_run(
                config.generation.context.representation_run_id, dataset_id=dataset.id
            )

        expected = {
            "representations": ("preprocess", config.preprocess),
            "chunks": ("chunks", config.chunking),
            "query_embeddings": ("embed_queries", config.embeddings),
            "corpus_embeddings": (
                "embed_chunks" if config.corpus_unit == "chunk" else "embed_pages",
                config.embeddings,
            ),
            "retrieval": ("retrieval", config.retrieval),
        }
        for role, (kind, wanted_config) in expected.items():
            if role not in chosen:
                continue
            run = await crud.validate_run(
                chosen[role], kind=kind, dataset_id=dataset.id
            )
            if wanted_config is not None and not _same_artifact_config(
                run.config, wanted_config.model_dump(mode="json")
            ):
                raise ValueError(
                    f"Reused {role} configuration differs from the experiment"
                )
            if role == "query_embeddings":
                items = await crud.find_records(
                    RunItem, run_id=run.id, status="completed"
                )
                if {item.query_id for item in items} != set(query_ids):
                    raise ValueError(
                        "Reused query embeddings have a different query cohort"
                    )
                info = await crud.get_record(EmbeddingRun, run.id)
                if info.role != "query" or info.point_count != len(query_ids):
                    raise ValueError("Reused query embedding coverage is incomplete")

        needs_text_vectors = (
            config.corpus_unit == "chunk"
            or config.embeddings.sparse is not None
            or (
                config.embeddings.dense is not None
                and config.embeddings.dense.modality == "text"
            )
        )
        needs_context = bool(
            config.generation
            and any(
                kind not in {"image", "original_image"}
                for kind in config.generation.context.representations
            )
            and config.generation.context.representation_run_id is None
        )
        if (needs_text_vectors or needs_context) and "representations" not in chosen:
            if config.preprocess is None:
                raise ValueError(
                    "Text retrieval or OCR generation context requires preprocess configuration or reuse.representations"
                )
            chosen["representations"] = await _experiment_stage(
                experiment,
                "representations",
                preprocess_pages(dataset.id, config.preprocess),
            )
        if config.corpus_unit == "chunk" and "chunks" not in chosen:
            chosen["chunks"] = await _experiment_stage(
                experiment,
                "chunks",
                build_chunks(chosen["representations"], config.chunking),
            )
        if config.generation and needs_context:
            config.generation.context.representation_run_id = chosen["representations"]
        if config.generation and "generation" in chosen:
            run = await crud.validate_run(
                chosen["generation"], kind="generation", dataset_id=dataset.id
            )
            if not _same_artifact_config(
                run.config, config.generation.model_dump(mode="json")
            ):
                raise ValueError(
                    "Reused generation configuration differs from the experiment"
                )
        await crud.update_record(
            Experiment, experiment.id, config=config.model_dump(mode="json")
        )

        operations = []
        operation_roles = []
        if "query_embeddings" not in chosen:
            operation_roles.append("query_embeddings")
            operations.append(
                _experiment_stage(
                    experiment,
                    "query_embeddings",
                    vectorize_queries(
                        dataset.id, config.embeddings, query_ids=query_ids
                    ),
                )
            )
        if "corpus_embeddings" not in chosen:
            operation_roles.append("corpus_embeddings")
            operation = (
                vectorize_chunks(chosen["chunks"], config.embeddings)
                if config.corpus_unit == "chunk"
                else vectorize_pages(
                    dataset.id,
                    config.embeddings,
                    representation_run_id=chosen.get("representations"),
                )
            )
            operations.append(
                _experiment_stage(experiment, "corpus_embeddings", operation)
            )
        # Settle both branches so one failure cannot abandon a running sibling.
        results = await asyncio.gather(*operations, return_exceptions=True)
        errors = []
        for role, result in zip(operation_roles, results, strict=True):
            if isinstance(result, BaseException):
                errors.append(result)
            else:
                chosen[role] = result
        if errors:
            raise errors[0]
        if "retrieval" not in chosen:
            chosen["retrieval"] = await _experiment_stage(
                experiment,
                "retrieval",
                run_retrieval(
                    chosen["query_embeddings"],
                    chosen["corpus_embeddings"],
                    config.retrieval,
                ),
            )
        if config.ir:
            await _experiment_stage(
                experiment,
                "ir",
                evaluate_ir(experiment.id, chosen["retrieval"], config.ir),
            )
        if config.generation and "generation" not in chosen:
            chosen["generation"] = await _experiment_stage(
                experiment,
                "generation",
                run_generation(chosen["retrieval"], config.generation),
            )
        if config.ragas.metrics:
            await _experiment_stage(
                experiment,
                "ragas",
                evaluate_ragas(
                    experiment.id,
                    chosen["retrieval"],
                    chosen.get("generation"),
                    config.ragas,
                ),
            )
        await crud.update_record(Experiment, experiment.id, status="completed")
        return experiment.id
    except BaseException as exc:
        await crud.update_record(
            Experiment,
            experiment.id,
            status="partial" if isinstance(exc, IncompleteRunError) else "failed",
        )
        if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)):
            raise
        raise RuntimeError(f"Experiment {experiment.id} failed: {exc}") from exc


async def run_suite(configs: list[ExperimentConfig]) -> list[UUID]:
    if not configs:
        raise ValueError("An experiment suite must contain at least one configuration")
    return [await run_experiment(config) for config in configs]


async def list_datasets(status: str | None = None, source: str | None = None):
    filters = {}
    if status is not None:
        filters["status"] = status
    if source is not None:
        filters["source"] = source

    rows = []
    for dataset in await crud.find_records(Dataset, **filters):
        created_at = getattr(dataset, "created_at", None)
        updated_at = getattr(dataset, "updated_at", None)
        rows.append(
            {
                "dataset_id": str(dataset.id),
                "source": dataset.source,
                "subset": dataset.subset,
                "split": dataset.split,
                "revision": dataset.revision,
                "fingerprint": dataset.fingerprint,
                "status": dataset.status,
                "created_at": created_at.isoformat() if created_at else None,
                "updated_at": updated_at.isoformat() if updated_at else None,
                "metadata": getattr(dataset, "metadata_json", {}),
            }
        )
    return {"datasets": rows}


async def list_experiments(dataset_id: UUID | None = None, status: str | None = None):
    """List all experiments unless filters are supplied."""
    filters = {}
    if dataset_id is not None:
        filters["dataset_id"] = UUID(str(dataset_id))
    if status is not None:
        filters["status"] = status

    rows = []
    for experiment in await crud.find_records(Experiment, **filters):
        selections = await crud.find_records(
            ExperimentQuery, experiment_id=experiment.id
        )
        associations = await crud.find_records(
            ExperimentRun, experiment_id=experiment.id
        )
        runs = {}
        for association in associations:
            run = await crud.get_record(StageRun, association.run_id)
            started_at = getattr(run, "started_at", None)
            finished_at = getattr(run, "finished_at", None)
            runs[association.role] = {
                "run_id": str(run.id),
                "kind": run.kind,
                "status": run.status,
                "started_at": started_at.isoformat() if started_at else None,
                "finished_at": finished_at.isoformat() if finished_at else None,
            }

        created_at = getattr(experiment, "created_at", None)
        updated_at = getattr(experiment, "updated_at", None)
        rows.append(
            {
                "experiment_id": str(experiment.id),
                "name": experiment.name,
                "dataset_id": str(experiment.dataset_id),
                "status": experiment.status,
                "created_at": created_at.isoformat() if created_at else None,
                "updated_at": updated_at.isoformat() if updated_at else None,
                "query_count": len(selections),
                "runs": runs,
                "config": experiment.config,
            }
        )
    return {"experiments": rows}


async def discard_experiment(
    experiment_id: UUID, *, include_completed: bool = False
) -> dict:
    """Delete an experiment and progress rows that are not shared elsewhere."""
    experiment = await crud.get_record(Experiment, experiment_id)
    async with get_db() as session:
        associations = list(
            (
                await session.scalars(
                    select(ExperimentRun).where(
                        ExperimentRun.experiment_id == experiment.id
                    )
                )
            ).all()
        )
        candidate_ids = {association.run_id for association in associations}
        runs = (
            list(
                (
                    await session.scalars(
                        select(StageRun).where(StageRun.id.in_(candidate_ids))
                    )
                ).all()
            )
            if candidate_ids
            else []
        )
        run_by_id = {run.id: run for run in runs}
        linked_elsewhere = (
            set(
                (
                    await session.scalars(
                        select(ExperimentRun.run_id).where(
                            ExperimentRun.run_id.in_(candidate_ids),
                            ExperimentRun.experiment_id != experiment.id,
                        )
                    )
                ).all()
            )
            if candidate_ids
            else set()
        )
        delete_ids = {
            run.id
            for run in runs
            if run.id not in linked_elsewhere
            and (include_completed or run.status != "completed")
        }
        while delete_ids:
            blocked = set(
                (
                    await session.scalars(
                        select(RunDependency.input_run_id).where(
                            RunDependency.input_run_id.in_(delete_ids),
                            RunDependency.run_id.not_in(delete_ids),
                        )
                    )
                ).all()
            )
            if not blocked:
                break
            delete_ids -= blocked

        collection_names = (
            list(
                (
                    await session.scalars(
                        select(EmbeddingRun.collection_name).where(
                            EmbeddingRun.id.in_(delete_ids)
                        )
                    )
                ).all()
            )
            if delete_ids
            else []
        )
        run_ids = list(delete_ids)
        if run_ids:
            retrieval_ids = (
                select(Retrieval.id).where(Retrieval.run_id.in_(run_ids)).subquery()
            )
            generation_ids = (
                select(Generation.id).where(Generation.run_id.in_(run_ids)).subquery()
            )
            await session.execute(
                delete(MetricAggregate).where(
                    MetricAggregate.evaluation_run_id.in_(run_ids)
                )
            )
            await session.execute(
                delete(MetricResult).where(MetricResult.evaluation_run_id.in_(run_ids))
            )
            await session.execute(
                delete(EvaluationRun).where(EvaluationRun.id.in_(run_ids))
            )
            await session.execute(
                delete(GenerationContext).where(
                    GenerationContext.generation_id.in_(select(generation_ids.c.id))
                )
            )
            await session.execute(
                delete(Generation).where(Generation.run_id.in_(run_ids))
            )
            await session.execute(
                delete(GenerationRun).where(GenerationRun.id.in_(run_ids))
            )
            await session.execute(
                delete(RetrievalHit).where(
                    RetrievalHit.retrieval_id.in_(select(retrieval_ids.c.id))
                )
            )
            await session.execute(
                delete(Retrieval).where(Retrieval.run_id.in_(run_ids))
            )
            await session.execute(
                delete(RetrievalRun).where(RetrievalRun.id.in_(run_ids))
            )
            await session.execute(
                delete(EmbeddingRun).where(EmbeddingRun.id.in_(run_ids))
            )
            await session.execute(delete(Chunk).where(Chunk.run_id.in_(run_ids)))
            await session.execute(
                delete(PageRepresentation).where(PageRepresentation.run_id.in_(run_ids))
            )
            await session.execute(delete(RunItem).where(RunItem.run_id.in_(run_ids)))
            await session.execute(
                delete(RunDependency).where(RunDependency.run_id.in_(run_ids))
            )
            await session.execute(
                delete(RunDependency).where(RunDependency.input_run_id.in_(run_ids))
            )
            await session.execute(
                delete(ExperimentRun).where(ExperimentRun.run_id.in_(run_ids))
            )
            await session.execute(delete(StageRun).where(StageRun.id.in_(run_ids)))

        await session.execute(
            delete(ExperimentRun).where(ExperimentRun.experiment_id == experiment.id)
        )
        await session.execute(
            delete(ExperimentQuery).where(
                ExperimentQuery.experiment_id == experiment.id
            )
        )
        await session.execute(delete(Experiment).where(Experiment.id == experiment.id))

    deleted_collections = []
    collection_errors = {}
    for name in collection_names:
        store = create_vector_store(name)
        try:
            if await store.delete_collection():
                deleted_collections.append(name)
        except Exception as exc:
            collection_errors[name] = str(exc)
        finally:
            await store.close()

    preserved = []
    for run_id, run in sorted(run_by_id.items(), key=lambda item: str(item[0])):
        if run_id in delete_ids:
            continue
        reason = "completed"
        if run_id in linked_elsewhere:
            reason = "shared with another experiment"
        elif include_completed:
            reason = "referenced by preserved downstream run"
        preserved.append(
            {
                "run_id": str(run_id),
                "kind": run.kind,
                "status": run.status,
                "reason": reason,
            }
        )

    return {
        "experiment_id": str(experiment.id),
        "status": "discarded",
        "deleted_runs": [str(run_id) for run_id in sorted(delete_ids, key=str)],
        "preserved_runs": preserved,
        "deleted_collections": deleted_collections,
        "collection_errors": collection_errors,
    }


async def compare_experiments(experiment_ids: list[UUID]):
    if len(experiment_ids) < 2 or len(set(experiment_ids)) != len(experiment_ids):
        raise ValueError("Select at least two distinct experiments")
    rows = []
    for experiment_id in experiment_ids:
        experiment = await crud.get_record(Experiment, experiment_id)
        selections = await crud.find_records(
            ExperimentQuery, experiment_id=experiment.id
        )
        associations = await crud.find_records(
            ExperimentRun, experiment_id=experiment.id
        )
        evaluator_configs = {}
        evaluator_provenance = {}
        evaluations = []
        metrics = []
        runs = {
            association.role: str(association.run_id) for association in associations
        }
        for evaluation in await crud.find_records(
            EvaluationRun, experiment_id=experiment.id
        ):
            run = await crud.get_record(StageRun, evaluation.id)
            framework = (
                "ir" if evaluation.framework == "ir_measures" else evaluation.framework
            )
            evaluation_config = {
                key: value
                for key, value in run.config.items()
                if key != "experiment_id"
            }
            evaluation_identity_config = crud.identity_config(evaluation_config)
            config_hash = hashlib.sha256(
                json.dumps(
                    evaluation_identity_config, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
            evaluation_key = f"{framework}:{config_hash}"
            evaluator_configs[evaluation_key] = evaluation_identity_config
            provenance = {
                key: value
                for key, value in run.provenance.items()
                if key != "selection"
            }
            provenance_json = json.dumps(
                provenance, sort_keys=True, separators=(",", ":")
            )
            evaluator_provenance.setdefault(evaluation_key, set()).add(provenance_json)
            evaluations.append(
                {
                    "evaluation_run_id": str(run.id),
                    "evaluation_key": evaluation_key,
                    "framework": framework,
                    "status": run.status,
                    "config": evaluation_config,
                    "identity_config": evaluation_identity_config,
                    "provenance": run.provenance,
                    "retrieval_run_id": str(evaluation.retrieval_run_id),
                    "generation_run_id": str(evaluation.generation_run_id)
                    if evaluation.generation_run_id
                    else None,
                }
            )
            for metric in await crud.find_records(
                MetricAggregate, evaluation_run_id=run.id
            ):
                metrics.append(
                    {
                        "evaluation_run_id": str(run.id),
                        "evaluation_key": evaluation_key,
                        "framework": framework,
                        "metric_id": metric.metric_id,
                        "group_by": metric.group_by,
                        "group_value": metric.group_value,
                        "value": metric.value,
                        "expected_count": metric.expected_count,
                        "scored_count": metric.scored_count,
                        "skipped_count": metric.skipped_count,
                        "failed_count": metric.failed_count,
                    }
                )
        rows.append(
            {
                "experiment_id": str(experiment.id),
                "name": experiment.name,
                "dataset_id": str(experiment.dataset_id),
                "status": experiment.status,
                "query_ids": sorted(str(item.query_id) for item in selections),
                "config": experiment.config,
                "evaluator_configs": evaluator_configs,
                "evaluator_provenance": {
                    key: [json.loads(item) for item in sorted(values)]
                    for key, values in evaluator_provenance.items()
                },
                "evaluations": evaluations,
                "runs": runs,
                "metrics": metrics,
            }
        )
    reasons = []
    first = rows[0]
    for row in rows[1:]:
        for field, reason in (
            ("dataset_id", "dataset snapshots differ"),
            ("query_ids", "query cohorts differ"),
            ("evaluator_configs", "evaluator configurations differ"),
            ("evaluator_provenance", "evaluator implementation provenance differs"),
        ):
            if row[field] != first[field]:
                reasons.append(f"{row['experiment_id']}: {reason}")
    if any(row["status"] != "completed" for row in rows):
        reasons.append("One or more experiments are incomplete; inspect coverage")
    if any(
        evaluation["status"] != "completed"
        for row in rows
        for evaluation in row["evaluations"]
    ):
        reasons.append("One or more evaluation runs are incomplete; inspect coverage")
    differences = {}
    for key in set().union(*(row["config"] for row in rows)) - {"name", "reuse"}:
        values = {row["experiment_id"]: row["config"].get(key) for row in rows}
        identity_values = {
            experiment_id: crud.identity_config(value)
            for experiment_id, value in values.items()
        }
        if any(
            value != next(iter(identity_values.values()))
            for value in identity_values.values()
        ):
            differences[key] = values
    return {
        "compatible": not reasons,
        "compatibility_reasons": reasons,
        "configuration_differences": differences,
        "experiments": rows,
    }


async def show_run(run_id: UUID):
    run = await crud.get_record(StageRun, run_id)
    dependencies = await crud.find_records(RunDependency, run_id=run.id)
    failures = await crud.find_records(RunItem, run_id=run.id, status="failed")
    metric_diagnostics = {}
    if run.kind in {"evaluate_ir", "evaluate_ragas"}:
        for status in ("failed", "skipped"):
            results = await crud.find_records(
                MetricResult, evaluation_run_id=run.id, status=status
            )
            metric_diagnostics[f"{status}_metrics"] = [
                {
                    "query_id": str(result.query_id),
                    "metric_id": result.metric_id,
                    "error": result.error,
                    "reason": result.reason,
                }
                for result in results
            ]
    return {
        **metric_diagnostics,
        "run_id": str(run.id),
        "dataset_id": str(run.dataset_id),
        "kind": run.kind,
        "status": run.status,
        "config": run.config,
        "provenance": run.provenance,
        "metadata": run.metadata_json,
        "expected_count": run.expected_count,
        "completed_count": run.completed_count,
        "failed_count": run.failed_count,
        "error": run.error,
        "dependencies": {item.role: str(item.input_run_id) for item in dependencies},
        "failed_items": [
            {
                "id": str(item.id),
                "query_id": str(item.query_id) if item.query_id else None,
                "corpus_id": str(item.corpus_id) if item.corpus_id else None,
                "chunk_id": str(item.chunk_id) if item.chunk_id else None,
                "attempts": item.attempts,
                "error": item.error,
            }
            for item in failures
        ],
    }


async def run_resume(run_id: UUID) -> UUID:
    run = await crud.get_record(StageRun, run_id)
    if run.status == "completed":
        return run.id
    dependencies = {
        item.role: item.input_run_id
        for item in await crud.find_records(RunDependency, run_id=run.id)
    }
    config = dict(run.config)
    kwargs = {"resume_run_id": run.id}
    if run.kind == "ingest":
        await ingest_dataset(
            config["source"],
            revision=config.get("revision"),
            split=config.get("split", "test"),
            subset=config.get("subset"),
            **kwargs,
        )
        result = run.id
    elif run.kind == "preprocess":
        result = await preprocess_pages(
            run.dataset_id, PreprocessConfig.model_validate(config), **kwargs
        )
    elif run.kind == "chunks":
        result = await build_chunks(
            dependencies["representations"],
            ChunkConfig.model_validate(config),
            **kwargs,
        )
    elif run.kind == "embed_queries":
        selected = run.provenance.get("selection")
        if selected is None:
            experiments = await crud.find_records(ExperimentRun, run_id=run.id)
            if experiments:
                selected = [
                    str(item.query_id)
                    for item in await crud.find_records(
                        ExperimentQuery, experiment_id=experiments[0].experiment_id
                    )
                ]
        if selected is None:
            run_item_selection = [
                str(item.query_id)
                for item in await crud.find_records(RunItem, run_id=run.id)
                if item.query_id is not None
            ]
            if run_item_selection:
                selected = run_item_selection
        result = await vectorize_queries(
            run.dataset_id,
            EmbeddingConfig.model_validate(config),
            query_ids=[UUID(value) for value in selected]
            if selected is not None
            else None,
            **kwargs,
        )
    elif run.kind == "embed_chunks":
        result = await vectorize_chunks(
            dependencies["chunks"], EmbeddingConfig.model_validate(config), **kwargs
        )
    elif run.kind == "embed_pages":
        result = await vectorize_pages(
            run.dataset_id,
            EmbeddingConfig.model_validate(config),
            representation_run_id=dependencies.get("representations"),
            **kwargs,
        )
    elif run.kind == "retrieval":
        result = await run_retrieval(
            dependencies["queries"],
            dependencies["corpus"],
            RetrievalConfig.model_validate(config),
            **kwargs,
        )
    elif run.kind == "generation":
        result = await run_generation(
            dependencies["retrieval"], GenerationConfig.model_validate(config), **kwargs
        )
    elif run.kind in {"evaluate_ir", "evaluate_ragas"}:
        experiment_id = UUID(config.pop("experiment_id"))
        if run.kind == "evaluate_ir":
            result = await evaluate_ir(
                experiment_id,
                dependencies["retrieval"],
                IRConfig.model_validate(config),
                **kwargs,
            )
        else:
            result = await evaluate_ragas(
                experiment_id,
                dependencies["retrieval"],
                dependencies.get("generation"),
                RagasConfig.model_validate(config),
                **kwargs,
            )
    else:
        raise ValueError(f"Unsupported stage kind {run.kind}")
    await require_completed(result)
    return result
