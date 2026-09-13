"""Thin ID-based commands; stdout is JSON, progress and diagnostics use stderr."""

import asyncio
import csv
import io
import json
from pathlib import Path
from uuid import UUID

import typer
import yaml

from src.config import (
    ChunkConfig,
    EmbeddingConfig,
    ExperimentConfig,
    GenerationConfig,
    IRConfig,
    PreprocessConfig,
    RagasConfig,
    RetrievalConfig,
    load_config,
)
from src.etls.chunks import build_chunks
from src.etls.load_datasets import ingest_dataset
from src.etls.process_images import preprocess_pages
from src.evaluate.generation import run_generation
from src.evaluate.ir import evaluate_ir
from src.evaluate.metrics import metric_catalog
from src.evaluate.ragas import evaluate_ragas
from src.evaluate.retrieval import run_retrieval
from src.evaluate.vectorize import vectorize_chunks, vectorize_pages, vectorize_queries
from src.orchestrate.pipeline import (
    compare_experiments,
    discard_experiment,
    require_completed,
    run_experiment,
    run_resume,
    run_suite,
    show_run,
)
from src.stor_rel.crud import get_record
from src.stor_rel.entry import dispose_engine
from src.stor_rel.schema import Dataset

app = typer.Typer(
    no_args_is_help=True,
    help="Visual document retrieval benchmarks with durable stage IDs.",
)
dataset_app = typer.Typer(no_args_is_help=True)
preprocess_app = typer.Typer(no_args_is_help=True)
chunks_app = typer.Typer(no_args_is_help=True)
vectorize_app = typer.Typer(no_args_is_help=True)
retrieve_app = typer.Typer(no_args_is_help=True)
generate_app = typer.Typer(no_args_is_help=True)
evaluate_app = typer.Typer(no_args_is_help=True)
experiment_app = typer.Typer(no_args_is_help=True)
suite_app = typer.Typer(no_args_is_help=True)
run_app = typer.Typer(no_args_is_help=True)
metrics_app = typer.Typer(no_args_is_help=True)
db_app = typer.Typer(no_args_is_help=True)
for name, group in (
    ("dataset", dataset_app),
    ("preprocess", preprocess_app),
    ("chunks", chunks_app),
    ("vectorize", vectorize_app),
    ("retrieve", retrieve_app),
    ("generate", generate_app),
    ("evaluate", evaluate_app),
    ("experiment", experiment_app),
    ("suite", suite_app),
    ("run", run_app),
    ("metrics", metrics_app),
    ("db", db_app),
):
    app.add_typer(group, name=name)


def _load(path, model):
    try:
        return load_config(path, model)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        typer.echo(json.dumps({"error": str(exc)}))
        raise typer.Exit(2) from exc


def _execute(operation, kind="result", output_format="json"):
    async def execute():
        try:
            result = await operation
            if kind == "run":
                run = await require_completed(result)
                return {
                    "run_id": str(run.id),
                    "status": run.status,
                    "kind": run.kind,
                    "dataset_id": str(run.dataset_id),
                }
            if kind == "dataset":
                dataset = await get_record(Dataset, result)
                if dataset.status != "completed":
                    raise ValueError(f"Dataset {dataset.id} ended {dataset.status}")
                return {"dataset_id": str(dataset.id), "status": dataset.status}
            if kind == "experiment":
                return {"experiment_id": str(result), "status": "completed"}
            if kind == "suite":
                return {
                    "experiment_ids": [str(value) for value in result],
                    "status": "completed",
                }
            return result
        finally:
            await dispose_engine()

    try:
        result = asyncio.run(execute())
        if output_format == "csv":
            stream = io.StringIO()
            fields = [
                "experiment_id",
                "name",
                "dataset_id",
                "status",
                "compatible",
                "framework",
                "evaluation_run_id",
                "evaluation_key",
                "metric_id",
                "group_by",
                "group_value",
                "value",
                "expected_count",
                "scored_count",
                "skipped_count",
                "failed_count",
            ]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for experiment in result["experiments"]:
                common = {key: experiment[key] for key in fields[:4]}
                for metric in experiment["metrics"]:
                    writer.writerow(
                        {**common, "compatible": result["compatible"], **metric}
                    )
            typer.echo(stream.getvalue().rstrip())
            if result["compatibility_reasons"]:
                typer.echo(
                    json.dumps(
                        {"compatibility_reasons": result["compatibility_reasons"]}
                    ),
                    err=True,
                )
        else:
            typer.echo(json.dumps(result, default=str, ensure_ascii=False))
    except Exception as exc:
        details = {"error": str(exc)}
        if hasattr(exc, "run_id"):
            details.update(run_id=str(exc.run_id), status=exc.status)
        typer.echo(json.dumps(details, ensure_ascii=False))
        raise typer.Exit(1) from exc


@dataset_app.command("ingest")
def dataset_ingest(
    source: str = typer.Option(...),
    revision: str | None = typer.Option(None),
    split: str = typer.Option("test"),
    subset: str | None = typer.Option(None),
    resume_run_id: UUID | None = typer.Option(None),
):
    """Ingest a source snapshot and normalize IDs and query rephrases."""
    _execute(
        ingest_dataset(
            source,
            revision=revision,
            split=split,
            subset=subset,
            resume_run_id=resume_run_id,
        ),
        "dataset",
    )


@preprocess_app.command("run")
def preprocess_run(
    dataset_id: UUID = typer.Option(...),
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    resume_run_id: UUID | None = typer.Option(None),
):
    _execute(
        preprocess_pages(
            dataset_id, _load(config, PreprocessConfig), resume_run_id=resume_run_id
        ),
        "run",
    )


@chunks_app.command("build")
def chunks_build(
    representation_run_id: UUID = typer.Option(...),
    config: Path | None = typer.Option(None, exists=True, dir_okay=False),
    resume_run_id: UUID | None = typer.Option(None),
):
    _execute(
        build_chunks(
            representation_run_id,
            _load(config, ChunkConfig) if config else ChunkConfig(),
            resume_run_id=resume_run_id,
        ),
        "run",
    )


@vectorize_app.command("queries")
def vectorize_queries_command(
    dataset_id: UUID = typer.Option(...),
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    query_id: list[UUID] | None = typer.Option(
        None, help="Repeat to select query UUIDs; omit to encode all queries."
    ),
    resume_run_id: UUID | None = typer.Option(None),
):
    _execute(
        vectorize_queries(
            dataset_id,
            _load(config, EmbeddingConfig),
            query_ids=query_id or None,
            resume_run_id=resume_run_id,
        ),
        "run",
    )


@vectorize_app.command("chunks")
def vectorize_chunks_command(
    chunk_run_id: UUID = typer.Option(...),
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    resume_run_id: UUID | None = typer.Option(None),
):
    _execute(
        vectorize_chunks(
            chunk_run_id, _load(config, EmbeddingConfig), resume_run_id=resume_run_id
        ),
        "run",
    )


@vectorize_app.command("pages")
def vectorize_pages_command(
    dataset_id: UUID = typer.Option(...),
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    representation_run_id: UUID | None = typer.Option(None),
    resume_run_id: UUID | None = typer.Option(None),
):
    _execute(
        vectorize_pages(
            dataset_id,
            _load(config, EmbeddingConfig),
            representation_run_id=representation_run_id,
            resume_run_id=resume_run_id,
        ),
        "run",
    )


@retrieve_app.command("run")
def retrieve_run(
    query_embedding_run_id: UUID = typer.Option(...),
    corpus_embedding_run_id: UUID = typer.Option(...),
    config: Path | None = typer.Option(None, exists=True, dir_okay=False),
    resume_run_id: UUID | None = typer.Option(None),
):
    _execute(
        run_retrieval(
            query_embedding_run_id,
            corpus_embedding_run_id,
            _load(config, RetrievalConfig) if config else RetrievalConfig(),
            resume_run_id=resume_run_id,
        ),
        "run",
    )


@generate_app.command("run")
def generate_run(
    retrieval_run_id: UUID = typer.Option(...),
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    resume_run_id: UUID | None = typer.Option(None),
):
    _execute(
        run_generation(
            retrieval_run_id,
            _load(config, GenerationConfig),
            resume_run_id=resume_run_id,
        ),
        "run",
    )


@evaluate_app.command("ir")
def evaluate_ir_command(
    experiment_id: UUID = typer.Option(...),
    retrieval_run_id: UUID = typer.Option(...),
    config: Path | None = typer.Option(None, exists=True, dir_okay=False),
    resume_run_id: UUID | None = typer.Option(None),
):
    _execute(
        evaluate_ir(
            experiment_id,
            retrieval_run_id,
            _load(config, IRConfig) if config else IRConfig(),
            resume_run_id=resume_run_id,
        ),
        "run",
    )


@evaluate_app.command("ragas")
def evaluate_ragas_command(
    experiment_id: UUID = typer.Option(...),
    retrieval_run_id: UUID = typer.Option(...),
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    generation_run_id: UUID | None = typer.Option(None),
    resume_run_id: UUID | None = typer.Option(None),
):
    _execute(
        evaluate_ragas(
            experiment_id,
            retrieval_run_id,
            generation_run_id,
            _load(config, RagasConfig),
            resume_run_id=resume_run_id,
        ),
        "run",
    )


@experiment_app.command("run")
def experiment_run(config: Path = typer.Option(..., exists=True, dir_okay=False)):
    _execute(run_experiment(_load(config, ExperimentConfig)), "experiment")


@experiment_app.command("discard")
def experiment_discard(
    experiment_id: UUID = typer.Option(...),
    include_completed: bool = typer.Option(
        False,
        help="Also delete completed runs that are exclusive to this experiment.",
    ),
):
    _execute(discard_experiment(experiment_id, include_completed=include_completed))


@experiment_app.command("compare")
def experiment_compare(
    experiment_id: list[UUID] = typer.Option(
        ..., help="Repeat for each experiment UUID."
    ),
    format: str = typer.Option("json", help="json or csv"),
):
    if format not in {"json", "csv"}:
        raise typer.BadParameter("format must be json or csv")
    _execute(compare_experiments(experiment_id), output_format=format)


@suite_app.command("run")
def suite_run(config: Path = typer.Option(..., exists=True, dir_okay=False)):
    """Run a YAML list of complete experiment configurations in order."""
    try:
        values = yaml.safe_load(config.read_text())
        if not isinstance(values, list):
            raise ValueError("Suite configuration must be a YAML list")
        configs = [ExperimentConfig.model_validate(value) for value in values]
    except (OSError, ValueError, yaml.YAMLError) as exc:
        typer.echo(json.dumps({"error": str(exc)}))
        raise typer.Exit(2) from exc
    _execute(run_suite(configs), "suite")


@run_app.command("show")
def run_show(run_id: UUID = typer.Option(...)):
    _execute(show_run(run_id))


@run_app.command("resume")
def run_resume_command(run_id: UUID = typer.Option(...)):
    _execute(run_resume(run_id), "run")


@metrics_app.command("list")
def metrics_list():
    typer.echo(json.dumps(metric_catalog(), ensure_ascii=False))


@db_app.command("migrate")
def db_migrate():
    """Apply all Alembic PostgreSQL migrations."""
    from alembic import command
    from alembic.config import Config

    try:
        command.upgrade(
            Config(str(Path(__file__).resolve().parents[1] / "alembic.ini")), "head"
        )
    except Exception as exc:
        typer.echo(json.dumps({"error": str(exc)}))
        raise typer.Exit(1) from exc
    typer.echo(json.dumps({"status": "completed", "revision": "head"}))


if __name__ == "__main__":
    app()
