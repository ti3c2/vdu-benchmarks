"""Export real relational joins using an isolated PostgreSQL schema."""

import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import text
from test_storage import source_records
from test_storage import storage_db as storage_db

from src.orchestrate.export import build_experiment_report, export_experiments
from src.stor_rel import crud
from src.stor_rel.entry import get_db
from src.stor_rel.schema import (
    Chunk,
    Corpus,
    EmbeddingRun,
    EvaluationRun,
    Experiment,
    ExperimentQuery,
    Generation,
    GenerationRun,
    MetricResult,
    PageRepresentation,
    Qrel,
    Query,
    Retrieval,
    RetrievalHit,
    RetrievalRun,
    StageRun,
)

pytestmark = pytest.mark.integration


@pytest.fixture
async def export_source(storage_db):
    dataset, doc, page, base = await source_records()
    second_page = await crud.save_record(
        Corpus,
        dataset_id=dataset.id,
        doc_id=doc.id,
        asset_id=page.asset_id,
        original_id="page-2",
    )
    rephrase = await crud.save_record(
        Query,
        dataset_id=dataset.id,
        original_id=base.original_id,
        query="Rephrased question?",
        rephrase_of_id=base.id,
        rephrase_level=1,
    )
    missing, unselected = [
        await crud.save_record(
            Query, dataset_id=dataset.id, original_id=name, query=name
        )
        for name in ("missing retrieval", "outside experiment")
    ]
    await crud.save_record(
        Qrel,
        dataset_id=dataset.id,
        query_id=base.id,
        corpus_id=page.id,
        score=1,
        answer="42",
    )
    zero_qrel = await crud.save_record(
        Qrel,
        dataset_id=dataset.id,
        query_id=base.id,
        corpus_id=second_page.id,
        score=0,
        answer="not a reference",
    )
    experiment = await crud.save_record(
        Experiment, dataset_id=dataset.id, name="export fixture", status="partial"
    )
    for query in (base, rephrase, missing):
        await crud.save_record(
            ExperimentQuery,
            dataset_id=dataset.id,
            experiment_id=experiment.id,
            query_id=query.id,
        )

    async def stage(kind):
        return await crud.save_record(
            StageRun,
            dataset_id=dataset.id,
            kind=kind,
            fingerprint=uuid4().hex,
            status="completed",
        )

    preprocess = await stage("preprocess")
    representation = await crud.save_record(
        PageRepresentation,
        dataset_id=dataset.id,
        run_id=preprocess.id,
        corpus_id=page.id,
        kind="ocr",
        text="Revenue: 42. Caf\u00e9.",
    )
    chunks = await stage("chunks")
    chunk = await crud.save_record(
        Chunk,
        dataset_id=dataset.id,
        run_id=chunks.id,
        corpus_id=page.id,
        representation_id=representation.id,
        ordinal=0,
        text=representation.text,
        content_hash="a" * 64,
    )
    embeddings = []
    for role, unit_kind in (("query", "query"), ("corpus", "chunk")):
        run = await stage(f"embed_{role}")
        embeddings.append(run.id)
        await crud.save_record(
            EmbeddingRun,
            id=run.id,
            dataset_id=dataset.id,
            role=role,
            unit_kind=unit_kind,
            collection_name=uuid4().hex,
        )
    retrieval_run = await stage("retrieval")
    await crud.save_record(
        RetrievalRun,
        id=retrieval_run.id,
        dataset_id=dataset.id,
        query_embedding_run_id=embeddings[0],
        corpus_embedding_run_id=embeddings[1],
    )
    await crud.attach_experiment_run(experiment.id, retrieval_run.id, "retrieval")
    retrievals = {}
    for query, status in (
        (base, "completed"),
        (rephrase, "failed"),
        (unselected, "completed"),
    ):
        retrievals[query.id] = await crud.save_record(
            Retrieval,
            dataset_id=dataset.id,
            run_id=retrieval_run.id,
            query_id=query.id,
            status=status,
            error="retrieval failed" if status == "failed" else None,
        )
    for rank, corpus, matched_chunk in ((2, page, chunk), (1, second_page, None)):
        await crud.save_record(
            RetrievalHit,
            dataset_id=dataset.id,
            retrieval_id=retrievals[base.id].id,
            corpus_id=corpus.id,
            chunk_id=matched_chunk.id if matched_chunk else None,
            point_id=matched_chunk.id if matched_chunk else corpus.id,
            rank=rank,
            score=1 / rank,
        )
    generation_run = await stage("generation")
    await crud.save_record(
        GenerationRun,
        id=generation_run.id,
        dataset_id=dataset.id,
        retrieval_run_id=retrieval_run.id,
    )
    await crud.attach_experiment_run(experiment.id, generation_run.id, "generation")
    await crud.save_record(
        Generation,
        dataset_id=dataset.id,
        run_id=generation_run.id,
        retrieval_id=retrievals[base.id].id,
        query_id=base.id,
        response="The answer is 42.",
        status="completed",
    )
    evaluations = []
    for value in (0.63, 0.5):
        evaluation = await stage("evaluate_ir")
        evaluations.append(evaluation)
        await crud.save_record(
            EvaluationRun,
            id=evaluation.id,
            dataset_id=dataset.id,
            experiment_id=experiment.id,
            retrieval_run_id=retrieval_run.id,
            generation_run_id=generation_run.id,
            framework="ir_measures",
        )
        for query in (base, unselected):
            await crud.save_record(
                MetricResult,
                dataset_id=dataset.id,
                evaluation_run_id=evaluation.id,
                query_id=query.id,
                metric_id="nDCG@5",
                value=value,
                status="completed",
            )
    for query, status in ((rephrase, "failed"), (missing, "skipped")):
        await crud.save_record(
            MetricResult,
            dataset_id=dataset.id,
            evaluation_run_id=evaluations[0].id,
            query_id=query.id,
            metric_id="nDCG@5",
            status=status,
            reason="No usable retrieval",
            error="retrieval failed" if status == "failed" else None,
        )
    await crud.save_record(
        MetricResult,
        dataset_id=dataset.id,
        evaluation_run_id=evaluations[1].id,
        query_id=base.id,
        metric_id="category",
        status="completed",
        raw_value={"label": "supported"},
    )
    return SimpleNamespace(**locals())


async def test_report_preserves_rank_cohort_references_and_metric_variants(
    export_source,
):
    source = export_source
    report = await build_experiment_report(source.experiment.id)
    assert report["query_count"] == 3
    records = {record["query_id"]: record for record in report["queries"]}
    assert str(source.unselected.id) not in records
    base = records[str(source.base.id)]
    assert base["reference_doc_ids"] == [str(source.doc.id)]
    assert base["reference_corpus_ids"] == [str(source.page.id)]
    assert base["reference_answer"] == "42"
    assert base["answer_text"] == "The answer is 42."
    hits = base["retrievals"][0]["hits"]
    assert [hit["rank"] for hit in hits] == [1, 2]
    assert hits[0]["chunk_id"] is None and hits[0]["chunk_text"] is None
    assert hits[1]["chunk_text"] == source.chunk.text
    assert hits[1]["chunk_id"] == str(source.chunk.id)
    assert base["retrievals"][0]["retrieved_doc_ids"] == [str(source.doc.id)]
    metrics = [m for m in base["metrics"] if m["metric_id"] == "nDCG@5"]
    assert {m["value"] for m in metrics} == {0.63, 0.5}
    assert len({m["evaluation_run_id"] for m in metrics}) == 2
    category = next(m for m in base["metrics"] if m["metric_id"] == "category")
    assert category["value"] is None and category["raw_value"] == {"label": "supported"}
    rephrase = records[str(source.rephrase.id)]
    assert rephrase["references"] == base["references"]
    assert rephrase["retrievals"][0]["status"] == "failed"
    assert rephrase["answer_text"] is None
    assert rephrase["metrics"][0]["value"] is None
    missing = records[str(source.missing.id)]
    assert missing["retrievals"][0]["status"] == "missing"
    assert missing["retrievals"][0]["hits"] == []
    assert missing["reference_answer"] is None
    assert missing["metrics"][0]["status"] == "skipped"

    await crud.update_record(
        Retrieval,
        source.retrievals[source.rephrase.id].id,
        status="completed",
        error=None,
    )
    updated = await build_experiment_report(source.experiment.id)
    empty = next(
        q for q in updated["queries"] if q["query_id"] == str(source.rephrase.id)
    )
    assert empty["retrievals"][0]["status"] == "completed"
    assert empty["retrievals"][0]["hits"] == []


async def test_standalone_evaluations_preserve_their_retrieval_and_answer(
    export_source,
):
    source = export_source
    retrieval_run = await source.stage("retrieval")
    await crud.save_record(
        RetrievalRun,
        id=retrieval_run.id,
        dataset_id=source.dataset.id,
        query_embedding_run_id=source.embeddings[0],
        corpus_embedding_run_id=source.embeddings[1],
    )
    retrieval = await crud.save_record(
        Retrieval,
        dataset_id=source.dataset.id,
        run_id=retrieval_run.id,
        query_id=source.base.id,
        status="completed",
    )
    generation_run = await source.stage("generation")
    await crud.save_record(
        GenerationRun,
        id=generation_run.id,
        dataset_id=source.dataset.id,
        retrieval_run_id=retrieval_run.id,
    )
    await crud.save_record(
        Generation,
        dataset_id=source.dataset.id,
        run_id=generation_run.id,
        retrieval_id=retrieval.id,
        query_id=source.base.id,
        response="Alternative answer",
        status="completed",
    )
    await crud.update_record(
        EvaluationRun,
        source.evaluations[1].id,
        retrieval_run_id=retrieval_run.id,
        generation_run_id=generation_run.id,
    )
    report = await build_experiment_report(source.experiment.id)
    base = next(q for q in report["queries"] if q["query_id"] == str(source.base.id))
    assert base["answer_text"] == "The answer is 42."
    assert {g["answer_text"] for g in base["generations"]} == {
        "The answer is 42.",
        "Alternative answer",
    }
    assert {r["retrieval_run_id"] for r in base["retrievals"]} == {
        str(source.retrieval_run.id),
        str(retrieval_run.id),
    }
    evaluation = next(
        e
        for e in report["evaluations"]
        if e["evaluation_run_id"] == str(source.evaluations[1].id)
    )
    assert evaluation["retrieval_run_id"] == str(retrieval_run.id)
    assert evaluation["generation_run_id"] == str(generation_run.id)


async def test_export_all_filters_refresh_and_missing_ids(export_source, tmp_path):
    source = export_source
    other_dataset, _, _, _ = await source_records("other")
    other = await crud.save_record(
        Experiment, dataset_id=other_dataset.id, name="no results yet"
    )
    result = await export_experiments(output_dir=tmp_path)
    assert len(result["exports"]) == 2
    empty_path = next(
        Path(item["path"])
        for item in result["exports"]
        if item["experiment_id"] == str(other.id)
    )
    assert json.loads(empty_path.read_text())["queries"] == []
    result = await export_experiments(dataset_id=source.dataset.id, output_dir=tmp_path)
    assert len(result["exports"]) == 1
    path = Path(result["exports"][0]["path"])
    saved = path.read_text(encoding="utf-8")
    assert saved.startswith('{\n  "experiment_id":')
    assert "Caf\u00e9" in saved
    assert json.loads(saved)["query_count"] == 3
    await crud.update_record(Qrel, source.zero_qrel.id, score=1, answer="43")
    await export_experiments([source.experiment.id], output_dir=tmp_path)
    refreshed = json.loads(path.read_text())
    base = next(q for q in refreshed["queries"] if q["query_id"] == str(source.base.id))
    assert base["reference_answer"] is None
    assert set(base["reference_answers"]) == {"42", "43"}
    assert len(list(tmp_path.glob("*_results.json"))) == 2
    with pytest.raises(ValueError, match="do not exist"):
        await export_experiments([uuid4()], output_dir=tmp_path)
    with pytest.raises(ValueError, match="do not match"):
        await export_experiments(
            [other.id], dataset_id=source.dataset.id, output_dir=tmp_path
        )


async def test_sql_joins_reference_chunks_in_query_and_rank_order(export_source):
    source = export_source
    await crud.update_record(Qrel, source.zero_qrel.id, score=1, answer="43")
    await crud.attach_experiment_run(source.experiment.id, source.chunks.id, "chunks")
    second_chunk = await crud.save_record(
        Chunk,
        dataset_id=source.dataset.id,
        run_id=source.chunks.id,
        corpus_id=source.page.id,
        representation_id=source.representation.id,
        ordinal=1,
        text="Another reference chunk",
        content_hash="b" * 64,
    )
    unrelated_run = await source.stage("chunks")
    await crud.save_record(
        Chunk,
        dataset_id=source.dataset.id,
        run_id=unrelated_run.id,
        corpus_id=source.page.id,
        representation_id=source.representation.id,
        ordinal=0,
        text="Chunk from another run",
        content_hash="c" * 64,
    )
    sql = (
        Path(__file__).resolve().parents[1] / "docs/sql/experiment_results.sql"
    ).read_text()
    async with get_db() as session:
        rows = (
            (await session.execute(text(sql), {"experiment_id": source.experiment.id}))
            .mappings()
            .all()
        )
    assert len(rows) == 10
    assert all(
        not isinstance(value, (list, dict)) for row in rows for value in row.values()
    )
    base_rows = [row for row in rows if row["query_id"] == source.base.id]
    assert [row["rank"] for row in base_rows] == [1, 1, 1, 2, 2, 2]
    assert all(row["retrieved_chunk_text"] is None for row in base_rows[:3])
    assert all(
        row["retrieved_chunk_text"] == source.chunk.text for row in base_rows[3:]
    )
    assert [
        row["reference_chunk_id"]
        for row in base_rows
        if row["reference_corpus_id"] == source.page.id
    ] == [source.chunk.id, second_chunk.id, source.chunk.id, second_chunk.id]
    assert {row["reference_chunk_text"] for row in base_rows} == {
        None,
        source.chunk.text,
        second_chunk.text,
    }
    assert base_rows[0]["answer_text"] == "The answer is 42."
    assert {row["reference_answer"] for row in base_rows} == {"42", "43"}
    rephrases = [row for row in rows if row["query_id"] == source.rephrase.id]
    assert {row["reference_answer"] for row in rephrases} == {"42", "43"}
    assert all(row["rank"] is None for row in rephrases)
    missing = next(row for row in rows if row["query_id"] == source.missing.id)
    assert missing["rank"] is None and missing["answer_text"] is None
