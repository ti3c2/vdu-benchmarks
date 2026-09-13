from types import SimpleNamespace

import pytest

from src.config import EndpointProfile
from src.evaluate.model_endpoints import verify_model_endpoint


class FakeModels:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    async def list(self):
        if self.error:
            raise self.error
        return self.result


class FakeClient:
    def __init__(self, *, models):
        self.models = models
        self.closed = False

    async def close(self):
        self.closed = True


def fake_client_factory(models):
    clients = []

    def create(**kwargs):
        client = FakeClient(models=models)
        clients.append((client, kwargs))
        return client

    return create, clients


async def test_model_endpoint_preflight_accepts_listed_model():
    create, clients = fake_client_factory(
        FakeModels(
            result=SimpleNamespace(
                data=[SimpleNamespace(id="first"), SimpleNamespace(id="required")]
            )
        )
    )
    await verify_model_endpoint(
        EndpointProfile(model="required", base_url="https://example.test/v1"),
        purpose="embedding",
        client_factory=create,
    )
    assert clients[0][0].closed
    assert clients[0][1]["base_url"] == "https://example.test/v1"


async def test_model_endpoint_preflight_rejects_missing_model():
    create, _ = fake_client_factory(
        FakeModels(result=SimpleNamespace(data=[SimpleNamespace(id="other")]))
    )
    with pytest.raises(ValueError, match="does not list required model"):
        await verify_model_endpoint(
            EndpointProfile(model="required", base_url="https://example.test/v1"),
            purpose="generation",
            client_factory=create,
        )


async def test_model_endpoint_preflight_reports_unreachable_models_endpoint():
    create, _ = fake_client_factory(FakeModels(error=RuntimeError("offline")))
    with pytest.raises(ValueError, match="did not return /models"):
        await verify_model_endpoint(
            EndpointProfile(
                model="required",
                base_url="https://example.test/v1",
                max_retries=1,
            ),
            purpose="OCR",
            client_factory=create,
        )
