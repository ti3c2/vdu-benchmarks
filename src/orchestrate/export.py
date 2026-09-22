"""Readable, query-level snapshots of persisted experiment results."""

import json
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import UUID

from sqlalchemy import func, select

from src.stor_rel.entry import get_db
from src.stor_rel.schema import (
    Chunk,
    Corpus,
    Doc,
    EvaluationRun,
    Experiment,
    ExperimentQuery,
    ExperimentRun,
    Generation,
    MetricResult,
    Qrel,
    Query,
    Retrieval,
    RetrievalHit,
    StageRun,
)


async def build_experiment_report(experiment_id: UUID) -> dict:
    """Read saved results without running retrieval, generation, or evaluation."""
    async with get_db() as session:
        # Keep queries, hits, and metrics consistent during an active experiment.
        await session.connection(
            execution_options={"isolation_level": "REPEATABLE READ"}
        )
        experiment = await session.get(Experiment, UUID(str(experiment_id)))
        if experiment is None:
            raise ValueError(f"Experiment {experiment_id} does not exist")
        selected_ids = select(ExperimentQuery.query_id).where(
            ExperimentQuery.experiment_id == experiment.id
        )
        queries = (
            await session.scalars(
                select(Query)
                .where(Query.id.in_(selected_ids))
                .order_by(Query.original_id, Query.rephrase_level, Query.id)
            )
        ).all()
        attached = (
            await session.execute(
                select(ExperimentRun.role, StageRun)
                .join(StageRun, StageRun.id == ExperimentRun.run_id)
                .where(ExperimentRun.experiment_id == experiment.id)
                .order_by(ExperimentRun.role)
            )
        ).all()
        evaluations = (
            await session.execute(
                select(EvaluationRun, StageRun)
                .join(StageRun, StageRun.id == EvaluationRun.id)
                .where(EvaluationRun.experiment_id == experiment.id)
                .order_by(StageRun.created_at, StageRun.id)
            )
        ).all()
        runs = {role: run for role, run in attached}
        retrieval_run_ids = {
            evaluation.retrieval_run_id for evaluation, _ in evaluations
        }
        generation_run_ids = {
            evaluation.generation_run_id
            for evaluation, _ in evaluations
            if evaluation.generation_run_id is not None
        }
        if "retrieval" in runs:
            retrieval_run_ids.add(runs["retrieval"].id)
        if "generation" in runs:
            generation_run_ids.add(runs["generation"].id)

        records = {
            query.id: {
                "query_id": str(query.id),
                "query_original_id": query.original_id,
                "query_text": query.query,
                "language": query.language,
                "rephrase_level": query.rephrase_level,
                "rephrase_of_id": str(query.rephrase_of_id)
                if query.rephrase_of_id
                else None,
                "reference_doc_ids": [],
                "reference_corpus_ids": [],
                "reference_answer": None,
                "reference_answers": [],
                "references": [],
                "answer_text": None,
                "answer_generation_run_id": None,
                "retrievals": [],
                "generations": [],
                "metrics": [],
            }
            for query in queries
        }
        references = await session.execute(
            select(Query.id, Qrel, Corpus, Doc)
            .select_from(Query)
            .join(
                Qrel,
                (Qrel.query_id == func.coalesce(Query.rephrase_of_id, Query.id))
                & (Qrel.dataset_id == Query.dataset_id),
            )
            .join(Corpus, Corpus.id == Qrel.corpus_id)
            .join(Doc, Doc.id == Corpus.doc_id)
            .where(Query.id.in_(selected_ids), Qrel.score > 0)
            .order_by(Query.id, Corpus.original_id, Corpus.id)
        )
        for query_id, qrel, page, doc in references:
            record = records[query_id]
            record["references"].append(
                {
                    "doc_id": str(doc.id),
                    "doc_original_id": doc.original_id,
                    "corpus_id": str(page.id),
                    "corpus_original_id": page.original_id,
                    "relevance": qrel.score,
                    "answer": qrel.answer,
                }
            )
            if str(doc.id) not in record["reference_doc_ids"]:
                record["reference_doc_ids"].append(str(doc.id))
            record["reference_corpus_ids"].append(str(page.id))
            if (
                qrel.answer
                and qrel.answer.strip()
                and qrel.answer not in record["reference_answers"]
            ):
                record["reference_answers"].append(qrel.answer)
        for record in records.values():
            if len(record["reference_answers"]) == 1:
                record["reference_answer"] = record["reference_answers"][0]

        retrievals = (
            await session.scalars(
                select(Retrieval).where(
                    Retrieval.run_id.in_(retrieval_run_ids),
                    Retrieval.query_id.in_(selected_ids),
                )
            )
        ).all()
        retrieval_by_query_run = {
            (retrieval.query_id, retrieval.run_id): retrieval
            for retrieval in retrievals
        }
        exported_retrievals = {}
        for query_id, record in records.items():
            for run_id in sorted(retrieval_run_ids, key=str):
                retrieval = retrieval_by_query_run.get((query_id, run_id))
                result = {
                    "retrieval_run_id": str(run_id),
                    "retrieval_id": str(retrieval.id) if retrieval else None,
                    "status": retrieval.status if retrieval else "missing",
                    "error": retrieval.error if retrieval else None,
                    "latency_ms": retrieval.latency_ms if retrieval else None,
                    "retrieved_doc_ids": [],
                    "retrieved_corpus_ids": [],
                    "hits": [],
                }
                record["retrievals"].append(result)
                if retrieval:
                    exported_retrievals[retrieval.id] = result
        hits = await session.execute(
            select(RetrievalHit, Corpus, Doc, Chunk.text)
            .join(Retrieval, Retrieval.id == RetrievalHit.retrieval_id)
            .join(Corpus, Corpus.id == RetrievalHit.corpus_id)
            .join(Doc, Doc.id == Corpus.doc_id)
            .outerjoin(Chunk, Chunk.id == RetrievalHit.chunk_id)
            .where(
                Retrieval.run_id.in_(retrieval_run_ids),
                Retrieval.query_id.in_(selected_ids),
            )
            .order_by(RetrievalHit.retrieval_id, RetrievalHit.rank)
        )
        for hit, page, doc, chunk_text in hits:
            result = exported_retrievals[hit.retrieval_id]
            if str(doc.id) not in result["retrieved_doc_ids"]:
                result["retrieved_doc_ids"].append(str(doc.id))
            result["retrieved_corpus_ids"].append(str(page.id))
            result["hits"].append(
                {
                    "rank": hit.rank,
                    "score": hit.score,
                    "doc_id": str(doc.id),
                    "doc_original_id": doc.original_id,
                    "corpus_id": str(page.id),
                    "corpus_original_id": page.original_id,
                    "chunk_id": str(hit.chunk_id) if hit.chunk_id else None,
                    "chunk_text": chunk_text,
                }
            )

        generations = await session.scalars(
            select(Generation)
            .where(
                Generation.run_id.in_(generation_run_ids),
                Generation.query_id.in_(selected_ids),
            )
            .order_by(Generation.created_at, Generation.id)
        )
        for generation in generations:
            records[generation.query_id]["generations"].append(
                {
                    "generation_id": str(generation.id),
                    "generation_run_id": str(generation.run_id),
                    "retrieval_id": str(generation.retrieval_id),
                    "answer_text": generation.response,
                    "status": generation.status,
                    "error": generation.error,
                    "latency_ms": generation.latency_ms,
                }
            )
        for record in records.values():
            answers = record["generations"]
            if "generation" in runs:
                answers = [
                    answer
                    for answer in answers
                    if answer["generation_run_id"] == str(runs["generation"].id)
                ]
            if len(answers) == 1:
                record["answer_text"] = answers[0]["answer_text"]
                record["answer_generation_run_id"] = answers[0]["generation_run_id"]

        metrics = await session.execute(
            select(MetricResult, EvaluationRun.framework)
            .join(EvaluationRun, EvaluationRun.id == MetricResult.evaluation_run_id)
            .where(
                EvaluationRun.experiment_id == experiment.id,
                MetricResult.query_id.in_(selected_ids),
            )
            .order_by(MetricResult.evaluation_run_id, MetricResult.metric_id)
        )
        for metric, framework in metrics:
            records[metric.query_id]["metrics"].append(
                {
                    "evaluation_run_id": str(metric.evaluation_run_id),
                    "framework": framework,
                    "metric_id": metric.metric_id,
                    "value": metric.value,
                    "raw_value": metric.raw_value,
                    "status": metric.status,
                    "reason": metric.reason,
                    "error": metric.error,
                }
            )

        return {
            "experiment_id": str(experiment.id),
            "name": experiment.name,
            "dataset_id": str(experiment.dataset_id),
            "status": experiment.status,
            "created_at": experiment.created_at.isoformat(),
            "updated_at": experiment.updated_at.isoformat(),
            "exported_at": datetime.now(UTC).isoformat(),
            "config": experiment.config,
            "runs": {role: str(run.id) for role, run in attached},
            "evaluations": [
                {
                    "evaluation_run_id": str(evaluation.id),
                    "framework": evaluation.framework,
                    "retrieval_run_id": str(evaluation.retrieval_run_id),
                    "generation_run_id": str(evaluation.generation_run_id)
                    if evaluation.generation_run_id
                    else None,
                    "status": run.status,
                    "config": run.config,
                    "provenance": run.provenance,
                }
                for evaluation, run in evaluations
            ],
            "query_count": len(records),
            "queries": list(records.values()),
        }


async def export_experiments(
    experiment_ids: list[UUID] | None = None,
    *,
    dataset_id: UUID | None = None,
    output_dir: Path = Path("data/experiments"),
) -> dict:
    """Save one JSON file per experiment; no IDs selects all matching experiments."""
    statement = select(Experiment).order_by(Experiment.created_at, Experiment.id)
    if experiment_ids is not None:
        experiment_ids = [UUID(str(id)) for id in experiment_ids]
        if not experiment_ids or len(set(experiment_ids)) != len(experiment_ids):
            raise ValueError("Select at least one experiment, without duplicate IDs")
        statement = statement.where(Experiment.id.in_(experiment_ids))
    if dataset_id is not None:
        statement = statement.where(Experiment.dataset_id == UUID(str(dataset_id)))
    async with get_db() as session:
        experiments = (await session.scalars(statement)).all()
    if experiment_ids is not None:
        missing = set(experiment_ids) - {experiment.id for experiment in experiments}
        if missing:
            raise ValueError(
                "Experiments do not exist or do not match the dataset: "
                + ", ".join(sorted(map(str, missing)))
            )

    exports = []
    for experiment in experiments:
        report = await build_experiment_report(experiment.id)
        rendered = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
        timestamp = experiment.created_at.astimezone(UTC).strftime("%Y%m%d-%H%M%S")
        path = Path(output_dir) / f"{timestamp}_{experiment.id}_results.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        # Replace only after the full snapshot is written, preserving prior exports.
        temporary_path = None
        try:
            with NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(rendered + "\n")
            temporary_path.replace(path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        exports.append(
            {
                "experiment_id": str(experiment.id),
                "path": str(path),
                "query_count": report["query_count"],
            }
        )
    return {"exports": exports}
