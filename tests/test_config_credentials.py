import dotenv

from src.config import EndpointProfile


def test_process_credentials_override_dotenv(monkeypatch):
    monkeypatch.setenv("TEST_MODEL_KEY", "process-value")
    monkeypatch.setattr(
        dotenv, "dotenv_values", lambda path: {"TEST_MODEL_KEY": "dotenv-value"}
    )
    profile = EndpointProfile(model="fixture", api_key_env="TEST_MODEL_KEY")
    assert profile.resolve_api_key() == "process-value"
    assert "process-value" not in profile.model_dump_json()


def test_dotenv_credentials_are_resolved_but_never_serialized(monkeypatch):
    monkeypatch.delenv("TEST_MODEL_KEY", raising=False)
    monkeypatch.setattr(
        dotenv, "dotenv_values", lambda path: {"TEST_MODEL_KEY": "dotenv-value"}
    )
    profile = EndpointProfile(model="fixture", api_key_env="TEST_MODEL_KEY")
    assert profile.resolve_api_key() == "dotenv-value"
    assert "dotenv-value" not in profile.model_dump_json()
