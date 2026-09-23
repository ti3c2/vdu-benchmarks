import csv
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
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


def test_cli_dataset_list(monkeypatch):
    dataset_id = uuid4()
    created_at = datetime(2026, 9, 14, 9, 0, tzinfo=UTC).isoformat()

    async def list_existing(status=None, source=None):
        assert status == "completed"
        assert source == "fixture/source"
        return {
            "datasets": [
                {
                    "dataset_id": str(dataset_id),
                    "source": source,
                    "subset": "",
                    "split": "test",
                    "revision": "main",
                    "fingerprint": "abc123",
                    "status": status,
                    "created_at": created_at,
                    "updated_at": created_at,
                    "metadata": {"documents": 1},
                }
            ]
        }

    async def dispose():
        pass

    monkeypatch.setattr(cli, "list_datasets", list_existing)
    monkeypatch.setattr(cli, "dispose_engine", dispose)
    result = runner.invoke(
        cli.app,
        [
            "dataset",
            "list",
            "--status",
            "completed",
            "--source",
            "fixture/source",
        ],
    )
    assert result.exit_code == 0
    assert result.stdout.startswith("{\n  ")
    payload = json.loads(result.stdout)
    assert payload["datasets"][0]["dataset_id"] == str(dataset_id)
    assert payload["datasets"][0]["metadata"] == {"documents": 1}


def test_cli_experiment_list(monkeypatch):
    listed_dataset_id = uuid4()
    experiment_id = uuid4()
    created_at = datetime(2026, 9, 13, 12, 0, tzinfo=UTC).isoformat()

    async def list_existing(dataset_id=None, status=None):
        assert dataset_id is None
        assert status == "completed"
        return {
            "experiments": [
                {
                    "experiment_id": str(experiment_id),
                    "name": "dense",
                    "dataset_id": str(listed_dataset_id),
                    "status": "completed",
                    "created_at": created_at,
                    "updated_at": created_at,
                    "query_count": 2,
                    "runs": {},
                    "config": {"retrieval": {"mode": "dense"}},
                }
            ]
        }

    async def dispose():
        pass

    monkeypatch.setattr(cli, "list_experiments", list_existing)
    monkeypatch.setattr(cli, "dispose_engine", dispose)
    result = runner.invoke(
        cli.app,
        [
            "experiment",
            "list",
            "--status",
            "completed",
        ],
    )
    assert result.exit_code == 0
    assert result.stdout.startswith("{\n  ")
    payload = json.loads(result.stdout)
    assert payload["experiments"][0]["experiment_id"] == str(experiment_id)
    assert payload["experiments"][0]["config"]["retrieval"]["mode"] == "dense"


@pytest.mark.parametrize("select_ids", [False, True])
def test_cli_experiment_export(monkeypatch, tmp_path, select_ids):
    ids = [uuid4(), uuid4()]
    dataset_id = uuid4()
    disposed = []

    async def export(experiment_ids, *, dataset_id=None, output_dir):
        assert experiment_ids == (ids if select_ids else None)
        assert dataset_id == expected_dataset_id
        assert output_dir == (tmp_path if select_ids else Path("data/experiments"))
        return {"exports": [{"path": "results.json", "query_count": 2}]}

    async def dispose():
        disposed.append(True)

    expected_dataset_id = dataset_id if select_ids else None
    monkeypatch.setattr(cli, "export_experiments", export)
    monkeypatch.setattr(cli, "dispose_engine", dispose)
    args = ["experiment", "export"]
    if select_ids:
        for id in ids:
            args.extend(["--experiment-id", str(id)])
        args.extend(["--dataset-id", str(dataset_id), "--output-dir", str(tmp_path)])
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["exports"][0]["query_count"] == 2
    assert disposed == [True]


def test_invalid_config_fails_before_stage_execution(tmp_path):
    config = tmp_path / "invalid.yaml"
    config.write_text("dense: null\nsparse: null\n")
    result = runner.invoke(
        cli.app,
        ["vectorize", "queries", "--dataset-id", str(uuid4()), "--config", str(config)],
    )
    assert result.exit_code == 2
    assert "Select at least one" in json.loads(result.stdout)["error"]


def test_comparison_csv_identifies_each_evaluation_variant(monkeypatch, tmp_path):
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
    monkeypatch.chdir(tmp_path)
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
            "--layout",
            "long",
        ],
    )
    assert result.exit_code == 0
    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    assert {row["evaluation_run_id"] for row in rows} == set(map(str, evaluation_ids))
    assert {row["evaluation_key"] for row in rows} == {"ir:variant-0", "ir:variant-1"}
    saved = list((tmp_path / "data" / "experiments").glob("*_comparison_long.csv"))
    assert len(saved) == 1
    assert saved[0].read_text().rstrip() == result.stdout.rstrip()
    assert "Saved comparison to data/experiments/" in result.stderr


@pytest.fixture
def comparison_result():
    dataset_id = str(uuid4())
    experiments = []
    for name, scores in (
        ("BM25", {"P@20": 0.04, "nDCG@5": 0.57, "P@5": 0.14, "P@1": 0.0}),
        ("Dense", {"nDCG@5": None, "P@5": 0.10, "P@1": 0.3}),
    ):
        evaluation_id = str(uuid4())
        metrics = [
            {
                "framework": "ir",
                "evaluation_run_id": evaluation_id,
                "evaluation_key": "ir:baseline",
                "metric_id": metric_id,
                "group_by": "dataset",
                "group_value": dataset_id,
                "value": value,
            }
            for metric_id, value in scores.items()
        ]
        metrics.append(
            {**metrics[-1], "group_by": "language", "group_value": "en", "value": 99}
        )
        experiments.append(
            {
                "experiment_id": str(uuid4()),
                "name": name,
                "dataset_id": dataset_id,
                "status": "completed",
                "metrics": metrics,
            }
        )
    return {
        "compatible": False,
        "compatibility_reasons": ["evaluator configurations differ"],
        "experiments": experiments,
    }


@pytest.mark.parametrize("output_format", [None, "json"])
def test_comparison_defaults_to_wide_and_saves_selected_format(
    monkeypatch, tmp_path, comparison_result, output_format
):
    async def compare(ids):
        assert list(map(str, ids)) == [
            e["experiment_id"] for e in comparison_result["experiments"]
        ]
        return comparison_result

    async def dispose():
        pass

    monkeypatch.setattr(cli, "compare_experiments", compare)
    monkeypatch.setattr(cli, "dispose_engine", dispose)
    monkeypatch.chdir(tmp_path)
    args = ["experiment", "compare"]
    for experiment in comparison_result["experiments"]:
        args.extend(["--experiment-id", experiment["experiment_id"]])
    if output_format:
        args.extend(["--format", output_format])
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    if output_format == "json":
        rows = json.loads(result.stdout)
        assert rows[0] == {"metric": "P@1", "BM25": 0.0, "Dense": 0.3}
        assert rows[2]["Dense"] is None and rows[3]["Dense"] is None
    else:
        rows = list(csv.DictReader(io.StringIO(result.stdout)))
        assert list(rows[0]) == ["metric", "BM25", "Dense"]
        assert rows[0] == {"metric": "P@1", "BM25": "0.0", "Dense": "0.3"}
        assert rows[2]["Dense"] == "" and rows[3]["Dense"] == ""
    assert [row["metric"] for row in rows] == ["P@1", "P@5", "nDCG@5", "P@20"]
    assert "evaluator configurations differ" in result.stderr
    saved = list(
        (tmp_path / "data/experiments").glob(
            f"*_comparison_wide.{output_format or 'csv'}"
        )
    )
    assert len(saved) == 1
    assert saved[0].read_text().rstrip() == result.stdout.rstrip()


def test_long_json_preserves_original_comparison(
    monkeypatch, tmp_path, comparison_result
):
    async def compare(ids):
        return comparison_result

    async def dispose():
        pass

    monkeypatch.setattr(cli, "compare_experiments", compare)
    monkeypatch.setattr(cli, "dispose_engine", dispose)
    monkeypatch.chdir(tmp_path)
    args = ["experiment", "compare", "--format", "json", "--layout", "long"]
    for experiment in comparison_result["experiments"]:
        args.extend(["--experiment-id", experiment["experiment_id"]])
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == comparison_result
    saved = next((tmp_path / "data/experiments").glob("*_comparison_long.json"))
    assert json.loads(saved.read_text()) == comparison_result


@pytest.mark.parametrize("names", [("same", "same"), ("metric", "Dense")])
def test_wide_comparison_disambiguates_experiment_names(comparison_result, names):
    for experiment, name in zip(comparison_result["experiments"], names, strict=True):
        experiment["name"] = name
    fields, rows = cli._compare_wide(comparison_result)
    assert fields[0] == "metric"
    assert len(set(fields)) == 3
    assert comparison_result["experiments"][0]["experiment_id"] in fields[1]
    assert rows[0][fields[1]] == 0.0 and rows[0][fields[2]] == 0.3


def test_wide_comparison_keeps_variants_and_rejects_ambiguous_runs(comparison_result):
    metrics = comparison_result["experiments"][0]["metrics"]
    baseline = next(metric for metric in metrics if metric["metric_id"] == "nDCG@5")
    variant = {
        **baseline,
        "evaluation_key": "ir:variant",
        "evaluation_run_id": str(uuid4()),
        "value": 0.8,
    }
    metrics.append(variant)
    _, rows = cli._compare_wide(comparison_result)
    scores = {row["metric"]: row for row in rows}
    assert scores["nDCG@5 [ir:baseline]"]["BM25"] == 0.57
    assert scores["nDCG@5 [ir:variant]"] == {
        "metric": "nDCG@5 [ir:variant]",
        "BM25": 0.8,
        "Dense": None,
    }
    metrics.append({**variant, "evaluation_run_id": str(uuid4()), "value": 0.9})
    with pytest.raises(ValueError, match="Use --layout long"):
        cli._compare_wide(comparison_result)
