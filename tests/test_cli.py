import json
from types import SimpleNamespace
from uuid import uuid4

from typer.testing import CliRunner

from src import cli
from src.orchestrate.pipeline import IncompleteRunError

runner = CliRunner()


def test_cli_stage_help_and_metric_catalog():
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0 and "experiment" in result.stdout
    result = runner.invoke(cli.app, ["vectorize", "queries", "--help"])
    assert result.exit_code == 0 and "query-id" in result.stdout
    result = runner.invoke(cli.app, ["metrics", "list"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["metrics"]


def test_cli_closes_database_engine_each_invocation(monkeypatch):
    run_id = uuid4()
    disposed = []

    async def resume(id):
        assert id == run_id
        return id

    async def complete(id):
        return SimpleNamespace(
            id=id, status="completed", kind="retrieval", dataset_id=uuid4()
        )

    async def dispose():
        disposed.append(True)

    monkeypatch.setattr(cli, "run_resume", resume)
    monkeypatch.setattr(cli, "require_completed", complete)
    monkeypatch.setattr(cli, "dispose_engine", dispose)
    for _ in range(2):
        result = runner.invoke(cli.app, ["run", "resume", "--run-id", str(run_id)])
        assert result.exit_code == 0
        assert json.loads(result.stdout)["run_id"] == str(run_id)
    assert len(disposed) == 2


def test_cli_partial_run_returns_nonzero_and_usable_id(monkeypatch):
    run_id = uuid4()

    async def resume(id):
        raise IncompleteRunError(
            SimpleNamespace(id=id, status="partial", error="one query failed")
        )

    async def dispose():
        pass

    monkeypatch.setattr(cli, "run_resume", resume)
    monkeypatch.setattr(cli, "dispose_engine", dispose)
    result = runner.invoke(cli.app, ["run", "resume", "--run-id", str(run_id)])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["run_id"] == str(run_id)
    assert json.loads(result.stdout)["status"] == "partial"


def test_cli_experiment_discard(monkeypatch):
    experiment_id = uuid4()

    async def discard(id, *, include_completed=False):
        assert id == experiment_id
        assert include_completed is True
        return {"experiment_id": str(id), "status": "discarded"}

    async def dispose():
        pass

    monkeypatch.setattr(cli, "discard_experiment", discard)
    monkeypatch.setattr(cli, "dispose_engine", dispose)
    result = runner.invoke(
        cli.app,
        [
            "experiment",
            "discard",
            "--experiment-id",
            str(experiment_id),
            "--include-completed",
        ],
    )
    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "experiment_id": str(experiment_id),
        "status": "discarded",
    }


def test_invalid_config_fails_before_stage_execution(tmp_path):
    config = tmp_path / "invalid.yaml"
    config.write_text("dense: null\nsparse: null\n")
    result = runner.invoke(
        cli.app,
        ["vectorize", "queries", "--dataset-id", str(uuid4()), "--config", str(config)],
    )
    assert result.exit_code == 2
    assert "Select at least one" in json.loads(result.stdout)["error"]


def test_comparison_csv_identifies_each_evaluation_variant(monkeypatch):
    import csv
    import io

    experiment_ids = [uuid4(), uuid4()]
    evaluation_ids = [uuid4(), uuid4()]

    async def compare(ids):
        assert ids == experiment_ids
        return {
            "compatible": False,
            "compatibility_reasons": ["evaluator configurations differ"],
            "experiments": [
                {
                    "experiment_id": str(experiment_ids[0]),
                    "name": "fixture",
                    "dataset_id": str(uuid4()),
                    "status": "completed",
                    "metrics": [
                        {
                            "framework": "ir",
                            "evaluation_run_id": str(id),
                            "evaluation_key": f"ir:variant-{index}",
                            "metric_id": "P@1",
                            "value": 1.0,
                        }
                        for index, id in enumerate(evaluation_ids)
                    ],
                }
            ],
        }

    async def dispose():
        pass

    monkeypatch.setattr(cli, "compare_experiments", compare)
    monkeypatch.setattr(cli, "dispose_engine", dispose)
    result = runner.invoke(
        cli.app,
        [
            "experiment",
            "compare",
            "--experiment-id",
            str(experiment_ids[0]),
            "--experiment-id",
            str(experiment_ids[1]),
            "--format",
            "csv",
        ],
    )
    assert result.exit_code == 0
    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    assert {row["evaluation_run_id"] for row in rows} == set(map(str, evaluation_ids))
    assert {row["evaluation_key"] for row in rows} == {"ir:variant-0", "ir:variant-1"}
