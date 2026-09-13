from types import SimpleNamespace

import pytest

from src.config import DenseConfig, EndpointProfile
from src.evaluate.model_endpoints import (
    verify_embedding_endpoint,
    verify_model_endpoint,
)


class FakeModels:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    async def list(self):
        if self.error:
            raise self.error
        return self.result


class FakeClient:
    def __init__(self, *, models=None, embeddings=None, post=None):
        self.models = models
        self.embeddings = embeddings
        self.post = post
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


class FakeEmbeddings:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.result


class FakePost:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    async def __call__(self, path, **kwargs):
        self.calls.append((path, kwargs))
        if self.error:
            raise self.error
        return self.result


def fake_embedding_client_factory(*, embeddings=None, post=None):
    clients = []

    def create(**kwargs):
        client = FakeClient(embeddings=embeddings, post=post)
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
        purpose="generation",
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


async def test_embedding_preflight_sends_sample_text_to_embeddings_endpoint():
    embeddings = FakeEmbeddings(
        result=SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2])])
    )
    create, clients = fake_embedding_client_factory(embeddings=embeddings)
    await verify_embedding_endpoint(
        DenseConfig(
            endpoint=EndpointProfile(model="embedding-model"),
            space_id="space",
            dimensions=2,
            query_instruction="query: ",
        ),
        client_factory=create,
    )
    assert clients[0][0].closed
    assert embeddings.calls == [
        {
            "model": "embedding-model",
            "input": ["query: preflight embedding check"],
            "encoding_format": "float",
            "dimensions": 2,
        }
    ]


async def test_embedding_preflight_supports_vllm_messages_extension():
    post = FakePost(
        result=SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2])])
    )
    create, _ = fake_embedding_client_factory(post=post)
    await verify_embedding_endpoint(
        DenseConfig(
            endpoint=EndpointProfile(model="vllm-embedding"),
            space_id="space",
            dimensions=2,
            adapter="vllm",
            modality="image",
            query_instruction="query: ",
        ),
        client_factory=create,
    )
    assert post.calls[0][0] == "/embeddings"
    body = post.calls[0][1]["body"]
    assert body["model"] == "vllm-embedding"
    assert body["messages"][0]["content"][0]["text"] == (
        "query: preflight embedding check"
    )


async def test_embedding_preflight_rejects_invalid_test_embedding():
    embeddings = FakeEmbeddings(
        result=SimpleNamespace(data=[SimpleNamespace(embedding=[0.1])])
    )
    create, _ = fake_embedding_client_factory(embeddings=embeddings)
    with pytest.raises(ValueError, match="returned 1 dimensions"):
        await verify_embedding_endpoint(
            DenseConfig(
                endpoint=EndpointProfile(model="embedding-model"),
                space_id="space",
                dimensions=2,
            ),
            client_factory=create,
        )
