"""Page-level retrieval metrics with explicit coverage and persisted ranks."""

import math
from collections import defaultdict
from statistics import mean
from uuid import UUID

import ir_measures

from src.config import IRConfig
from src.stor_rel.crud import (
    find_records,
    finish_run,
    get_effective_qrels,
    get_record,
    start_run,
    upsert_record,
    validate_run,
)
from src.stor_rel.schema import (
    EvaluationRun,
    Experiment,
    ExperimentQuery,
    MetricAggregate,
    MetricResult,
    Query,
    Retrieval,
    RetrievalHit,
)


def ir_metric_definitions(config: IRConfig):
    return [
        metric
        for k in config.cutoffs
        for metric in (
            ir_measures.P(rel=config.relevance_threshold, judged_only=False) @ k,
            ir_measures.R(rel=config.relevance_threshold, judged_only=False) @ k,
            ir_measures.RR(rel=config.relevance_threshold, judged_only=False) @ k,
            ir_measures.nDCG(dcg="log2", judged_only=False) @ k,
        )
    ]


def calculate_ir(qrels: list[dict], rankings: list[dict], config: IRConfig):
    """Rankings must contain unique page IDs and their already-decided ranks."""
    converted_qrels = []
    for row in qrels:
        score = float(row["score"])
        if not math.isfinite(score) or score < 0 or score != int(score):
            raise ValueError(
                "ir-measures requires nonnegative integer relevance labels"
            )
        converted_qrels.append(
            ir_measures.Qrel(str(row["query_id"]), str(row["corpus_id"]), int(score))
        )
    seen = set()
    converted_run = []
    for row in rankings:
        key = (str(row["query_id"]), str(row["corpus_id"]))
        if key in seen:
            raise ValueError("Rankings contain duplicate corpus pages for a query")
        seen.add(key)
        converted_run.append(ir_measures.ScoredDoc(*key, -float(row["rank"])))
    return [
        (result.query_id, str(result.measure), float(result.value))
        for result in ir_measures.iter_calc(
            ir_metric_definitions(config), converted_qrels, converted_run
        )
    ]


async def persist_aggregates(evaluation_run_id: UUID, queries: list[Query]):
    rows = await find_records(MetricResult, evaluation_run_id=evaluation_run_id)
    query_map = {q.id: q for q in queries}
    groups = defaultdict(list)
    for row in rows:
        query = query_map[row.query_id]
        for group_by, group_value in [
            ("dataset", str(query.dataset_id)),
            ("language", query.language or "unknown"),
            ("rephrase_level", str(query.rephrase_level)),
        ]:
            groups[(row.metric_id, group_by, group_value)].append(row)
    for (metric_id, group_by, group_value), items in groups.items():
        values = [
            r.value
            for r in items
            if r.status == "completed"
            and r.value is not None
            and math.isfinite(r.value)
        ]
        await upsert_record(
            MetricAggregate,
            {
                "evaluation_run_id": evaluation_run_id,
                "metric_id": metric_id,
                "group_by": group_by,
                "group_value": group_value,
            },
            {
                "dataset_id": items[0].dataset_id,
                "value": mean(values) if values else None,
                "expected_count": len(items),
                "scored_count": len(values),
                "skipped_count": sum(r.status == "skipped" for r in items),
                "failed_count": sum(r.status == "failed" for r in items),
            },
        )


async def evaluate_ir(
    experiment_id: UUID,
    retrieval_run_id: UUID,
    config: IRConfig | None = None,
    resume_run_id: UUID | None = None,
) -> UUID:
    config = config or IRConfig()
    experiment = await get_record(Experiment, experiment_id)
    source = await validate_run(
        retrieval_run_id, kind="retrieval", dataset_id=experiment.dataset_id
    )
    if max(config.cutoffs) > source.config.get("page_top_k", 20):
        raise ValueError("Requested metric cutoff exceeds saved retrieval depth")
    selections = await find_records(ExperimentQuery, experiment_id=experiment_id)
    query_ids = [row.query_id for row in selections]
    if not query_ids:
        raise ValueError("Experiment has no selected queries")
    retrievals = await find_records(Retrieval, run_id=retrieval_run_id)
    by_query = {r.query_id: r for r in retrievals}
    if set(by_query) != set(query_ids):
        raise ValueError("Retrieval query cohort does not match experiment")
    run = await start_run(
        experiment.dataset_id,
        "evaluate_ir",
        {**config.model_dump(mode="json"), "experiment_id": str(experiment_id)},
        inputs={"retrieval": retrieval_run_id},
        selection=sorted(map(str, query_ids)),
        resume_run_id=resume_run_id,
    )
    if run.status == "completed":
        return run.id
    await upsert_record(
        EvaluationRun,
        {"id": run.id},
        {
            "dataset_id": run.dataset_id,
            "experiment_id": experiment_id,
            "retrieval_run_id": retrieval_run_id,
            "framework": "ir_measures",
        },
    )
    try:
        qrels = await get_effective_qrels(experiment.dataset_id, query_ids)
        judged = {
            UUID(str(q["query_id"]))
            for q in qrels
            if q["score"] >= config.relevance_threshold
        }
        valid = {
            qid
            for qid in query_ids
            if by_query[qid].status == "completed" and qid in judged
        }
        rankings = []
        for qid in valid:
            for hit in await find_records(RetrievalHit, retrieval_id=by_query[qid].id):
                rankings.append(
                    {"query_id": qid, "corpus_id": hit.corpus_id, "rank": hit.rank}
                )
        scores = {
            (UUID(qid), metric): value
            for qid, metric, value in calculate_ir(
                [q for q in qrels if UUID(str(q["query_id"])) in valid],
                rankings,
                config,
            )
        }
        failed = 0
        scored = 0
        for qid in query_ids:
            for metric in ir_metric_definitions(config):
                status, value, reason = (
                    "completed",
                    scores.get((qid, str(metric)), 0.0),
                    None,
                )
                if by_query[qid].status != "completed":
                    status, value, reason = "failed", None, "Retrieval failed"
                    failed += 1
                elif qid not in judged:
                    status, value, reason = (
                        "skipped",
                        None,
                        "No relevant judgments for query",
                    )
                else:
                    scored += 1
                await upsert_record(
                    MetricResult,
                    {
                        "evaluation_run_id": run.id,
                        "query_id": qid,
                        "metric_id": str(metric),
                    },
                    {
                        "dataset_id": run.dataset_id,
                        "status": status,
                        "value": value,
                        "reason": reason,
                        "error": reason if status == "failed" else None,
                    },
                )
        queries = [await get_record(Query, qid) for qid in query_ids]
        await persist_aggregates(run.id, queries)
        await finish_run(
            run.id,
            status="partial" if failed else "completed",
            expected_count=len(query_ids) * len(ir_metric_definitions(config)),
            completed_count=scored,
            failed_count=failed,
        )
    except Exception as exc:
        await finish_run(run.id, status="failed", error=str(exc))
        raise
    return run.id
