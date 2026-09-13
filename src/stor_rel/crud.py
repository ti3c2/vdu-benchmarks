"""Small transactional operations shared by pipeline stages."""

import hashlib
import importlib.metadata
import json
import subprocess
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert

from .entry import get_db
from .schema import (
    Dataset,
    Experiment,
    ExperimentRun,
    Qrel,
    Query,
    RunDependency,
    StageRun,
)


def canonical_config(value):
    """Make configurations deterministic and exclude credentials from persistence."""
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        return {
            str(key): canonical_config(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key).lower()
            not in {"api_key", "password", "secret", "access_token", "authorization"}
            and not str(key).lower().endswith(("_api_key", "_password", "_secret"))
        }
    if isinstance(value, (list, tuple)):
        return [canonical_config(item) for item in value]
    if isinstance(value, (UUID, Path, datetime)):
        return str(value)
    return value


def implementation_provenance():
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    paths = [
        *sorted((root / "src").rglob("*.py")),
        root / "pyproject.toml",
        root / "uv.lock",
    ]
    for path in paths:
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    provenance = {"implementation_sha256": digest.hexdigest(), "packages": {}}
    for package in (
        "sqlalchemy",
        "llama-index-core",
        "qdrant-client",
        "ir-measures",
        "ragas",
        "fastembed",
        "openai",
    ):
        try:
            provenance["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        provenance["git_revision"] = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            .decode()
            .strip()
        )
        diff = subprocess.check_output(
            ["git", "diff", "HEAD", "--", "src", "pyproject.toml", "uv.lock"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        provenance["dirty_worktree_sha256"] = hashlib.sha256(diff).hexdigest()
    except (OSError, subprocess.SubprocessError):
        provenance["git_revision"] = None
    return provenance


async def get_record(model, id):
    async with get_db() as session:
        record = await session.get(model, UUID(str(id)))
        if record is None:
            raise ValueError(f"{model.__name__} {id} does not exist")
        return record


async def find_records(model, **filters):
    async with get_db() as session:
        return list(
            (
                await session.scalars(
                    select(model)
                    .filter_by(**filters)
                    .order_by(model.created_at, model.id)
                )
            ).all()
        )


async def save_record(model, **values):
    async with get_db() as session:
        record = model(**values)
        session.add(record)
        await session.flush()
        return record


async def update_record(model, id, **values):
    async with get_db() as session:
        record = await session.get(model, UUID(str(id)))
        if record is None:
            raise ValueError(f"{model.__name__} {id} does not exist")
        columns = set(model.__table__.columns.keys())
        for key, value in values.items():
            if key not in columns or key == "id":
                raise ValueError(f"Cannot update {model.__name__}.{key}")
            setattr(record, key, value)
        await session.flush()
        return record


async def upsert_record(model, keys: dict, values: dict):
    """An atomic PostgreSQL upsert; keys must identify one declared unique constraint."""
    if not keys:
        raise ValueError("Upsert requires a nonempty unique key")
    if keys.keys() & values.keys():
        raise ValueError("Upsert keys must not also appear in values")
    async with get_db() as session:
        statement = insert(model).values(**keys, **values)
        if values:
            statement = statement.on_conflict_do_update(
                index_elements=list(keys), set_={**values, "updated_at": func.now()}
            )
        else:
            statement = statement.on_conflict_do_nothing(index_elements=list(keys))
        record = (await session.scalars(statement.returning(model))).one_or_none()
        if record is None:
            record = (await session.scalars(select(model).filter_by(**keys))).one()
        return record


run_observer: ContextVar = ContextVar("stage_run_observer", default=None)


async def start_run(
    dataset_id,
    kind,
    config,
    inputs: dict[str, UUID] | None = None,
    selection: list[str] | None = None,
    resume_run_id: UUID | None = None,
):
    run = await _start_run(dataset_id, kind, config, inputs, selection, resume_run_id)
    observer = run_observer.get()
    if observer is not None:
        await observer(run)
    return run


async def _start_run(
    dataset_id,
    kind,
    config,
    inputs: dict[str, UUID] | None = None,
    selection: list[str] | None = None,
    resume_run_id: UUID | None = None,
):
    dataset_id = UUID(str(dataset_id))
    config = canonical_config(config)
    inputs = {role: UUID(str(run_id)) for role, run_id in (inputs or {}).items()}
    provenance = implementation_provenance()
    identity = {
        "dataset_id": str(dataset_id),
        "kind": kind,
        "config": config,
        "inputs": {role: str(run_id) for role, run_id in sorted(inputs.items())},
        "selection": sorted(set(map(str, selection)))
        if selection is not None
        else None,
        "provenance": provenance,
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    async with get_db() as session:
        if await session.get(Dataset, dataset_id) is None:
            raise ValueError(f"Dataset {dataset_id} does not exist")
        for role, input_id in inputs.items():
            dependency = await session.get(StageRun, input_id)
            if dependency is None or dependency.dataset_id != dataset_id:
                raise ValueError(
                    f"Input {role} is missing or belongs to another dataset"
                )
            if dependency.status != "completed":
                raise ValueError(
                    f"Input {role} ({input_id}) is {dependency.status}, expected completed"
                )
        # Serialize creators of the same run without locking around external work.
        lock_key = int.from_bytes(bytes.fromhex(fingerprint[:16]), "big", signed=True)
        await session.execute(
            text("SELECT pg_advisory_xact_lock(:key)"), {"key": lock_key}
        )
        if resume_run_id is not None:
            record = await session.get(StageRun, UUID(str(resume_run_id)))
            if record is None:
                raise ValueError(f"Run {resume_run_id} does not exist")
            if (
                record.dataset_id != dataset_id
                or record.kind != kind
                or record.fingerprint != fingerprint
            ):
                raise ValueError(
                    "Resume requires identical dataset, stage, inputs, selection, configuration, and implementation"
                )
        else:
            record = (
                await session.scalars(
                    select(StageRun).filter_by(
                        dataset_id=dataset_id, kind=kind, fingerprint=fingerprint
                    )
                )
            ).one_or_none()
        if record is not None:
            if record.status == "completed":
                return record
            if record.status == "running" and resume_run_id is None:
                raise ValueError(
                    f"Run {record.id} is already running; use its explicit resume ID after stopping the previous worker"
                )
            record.status = "running"
            record.error = None
            record.started_at = datetime.now(UTC)
            record.finished_at = None
        else:
            record = StageRun(
                dataset_id=dataset_id,
                kind=kind,
                config=config,
                fingerprint=fingerprint,
                provenance=provenance,
                status="running",
                started_at=datetime.now(UTC),
            )
            session.add(record)
            await session.flush()
            session.add_all(
                [
                    RunDependency(
                        dataset_id=dataset_id,
                        run_id=record.id,
                        input_run_id=input_id,
                        role=role,
                    )
                    for role, input_id in inputs.items()
                ]
            )
        await session.flush()
        return record


async def finish_run(run_id, status="completed", **counts):
    if status not in {"completed", "partial", "failed"}:
        raise ValueError("A finished run must be completed, partial, or failed")
    if set(counts) - {"expected_count", "completed_count", "failed_count", "error"}:
        raise ValueError("Unsupported run completion fields")
    return await update_record(
        StageRun, run_id, status=status, finished_at=datetime.now(UTC), **counts
    )


async def validate_run(run_id, kind=None, dataset_id=None):
    run = await get_record(StageRun, run_id)
    if run.status != "completed":
        raise ValueError(f"Run {run_id} is {run.status}; completed input is required")
    if kind is not None and run.kind != kind:
        raise ValueError(f"Run {run_id} has kind {run.kind}, expected {kind}")
    if dataset_id is not None and run.dataset_id != UUID(str(dataset_id)):
        raise ValueError(f"Run {run_id} belongs to another dataset")
    return run


async def get_effective_qrels(dataset_id, query_ids: list[UUID]):
    dataset_id = UUID(str(dataset_id))
    query_ids = list({UUID(str(query_id)) for query_id in query_ids})
    if not query_ids:
        return []
    async with get_db() as session:
        queries = list(
            (
                await session.scalars(
                    select(Query).where(
                        Query.dataset_id == dataset_id, Query.id.in_(query_ids)
                    )
                )
            ).all()
        )
        if len(queries) != len(query_ids):
            raise ValueError(
                "Selected queries are missing or belong to another dataset"
            )
        rows = await session.execute(
            select(Query.id.label("query_id"), Qrel.corpus_id, Qrel.answer, Qrel.score)
            .join(
                Qrel,
                (Qrel.query_id == func.coalesce(Query.rephrase_of_id, Query.id))
                & (Qrel.dataset_id == Query.dataset_id),
            )
            .where(Query.dataset_id == dataset_id, Query.id.in_(query_ids))
            .order_by(Query.id, Qrel.corpus_id)
        )
        return [dict(row) for row in rows.mappings()]


async def attach_experiment_run(experiment_id, run_id, role):
    async with get_db() as session:
        experiment = await session.get(Experiment, UUID(str(experiment_id)))
        run = await session.get(StageRun, UUID(str(run_id)))
        if experiment is None or run is None:
            raise ValueError("Experiment or run does not exist")
        if experiment.dataset_id != run.dataset_id:
            raise ValueError("Experiment and run must belong to the same dataset")
        statement = (
            insert(ExperimentRun)
            .values(
                dataset_id=experiment.dataset_id,
                experiment_id=experiment.id,
                run_id=run.id,
                role=role,
            )
            .on_conflict_do_nothing(index_elements=["experiment_id", "role"])
            .returning(ExperimentRun)
        )
        record = (await session.scalars(statement)).one_or_none()
        if record is None:
            record = (
                await session.scalars(
                    select(ExperimentRun).filter_by(
                        experiment_id=experiment.id, role=role
                    )
                )
            ).one()
            if record.run_id != run.id:
                raise ValueError(
                    f"Experiment role {role} is already pinned to another run"
                )
        return record
