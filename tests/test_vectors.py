import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from llama_index.core.schema import TextNode
from qdrant_client import AsyncQdrantClient, models

from src.config import DenseConfig, EndpointProfile
from src.evaluate.embeddings import (
    BM25Encoder,
    VLLMEmbedding,
    create_dense_model,
    encode_dense,
    validate_dense,
    validate_sparse,
)
from src.evaluate.vector_store import create_vector_store


def test_custom_model_and_instructions():
    config = DenseConfig(
        endpoint=EndpointProfile(model="custom/embedding"), space_id="example"
    )
    model = create_dense_model(config)
    assert (
        model.model_name
        == model._query_engine
        == model._text_engine
        == "custom/embedding"
    )


@pytest.mark.parametrize("vector", [[], [float("nan")], [float("inf")], [[1.0, 2.0]]])
def test_dense_vectors_reject_invalid_output(vector):
    with pytest.raises(ValueError):
        validate_dense(vector)


def test_dense_and_sparse_validation():
    with pytest.raises(ValueError, match="dimensions"):
        validate_dense([1.0], 2)
    assert validate_sparse([2, 0], [1.0, 0.5]).indices == [0, 2]
    for indices, values in [
        ([1, 1], [1.0, 2.0]),
        ([1], []),
        ([-1], [1.0]),
        ([2**32], [1.0]),
        ([1], [float("nan")]),
    ]:
        with pytest.raises(ValueError):
            validate_sparse(indices, values)


def test_bm25_uses_distinct_document_and_query_paths():
    encoder = BM25Encoder.__new__(BM25Encoder)
    encoder.model = SimpleNamespace(
        embed=lambda texts: [SimpleNamespace(indices=[1], values=[2.0]) for _ in texts],
        query_embed=lambda texts: [
            SimpleNamespace(indices=[1], values=[1.0]) for _ in texts
        ],
    )
    assert encoder.document_callback(["hello"])[1] == [[2.0]]
    assert encoder.query_callback(["hello"])[1] == [[1.0]]


def test_vllm_image_request_contains_actual_image_and_instruction():
    config = DenseConfig(
        endpoint=EndpointProfile(
            model="vision", extra_body={"truncate_prompt_tokens": -1}
        ),
        space_id="vision",
        modality="image",
        adapter="vllm",
        document_instruction="Describe.",
        dimensions=2,
    )
    model = VLLMEmbedding(config)
    model._async_client.post = AsyncMock(
        return_value=SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2])])
    )

    async def exercise():
        assert await model.aget_image_embedding("data:image/png;base64,AAAA") == [
            0.1,
            0.2,
        ]
        body = model._async_client.post.call_args.kwargs["body"]
        assert body["messages"][0]["content"][0]["image_url"]["url"].startswith(
            "data:image/png"
        )
        assert body["messages"][0]["content"][1]["text"] == "Describe."
        assert body["dimensions"] == 2
        assert "input" not in body
        await model.aclose()

    asyncio.run(exercise())


def test_dense_encoding_preserves_order_and_query_instruction():
    config = DenseConfig(
        endpoint=EndpointProfile(model="fake"),
        space_id="x",
        query_instruction="query: ",
        dimensions=2,
    )
    model = SimpleNamespace(
        aget_query_embedding=AsyncMock(side_effect=[[1.0, 0.0], [0.0, 1.0]])
    )
    output = asyncio.run(encode_dense(model, config, ["first", "second"], "query"))
    assert output == [[1.0, 0.0], [0.0, 1.0]]
    assert [call.args[0] for call in model.aget_query_embedding.await_args_list] == [
        "query: first",
        "query: second",
    ]


def test_sparse_only_persistence_and_query_idf():
    async def exercise():
        client = AsyncQdrantClient(":memory:")
        store = create_vector_store("sparse_queries", aclient=client)
        await store.ensure_collection(dimensions=None, sparse=True, role="query")
        point_id = str(uuid4())
        node = TextNode(id_=point_id, text="query")
        await store.async_add(
            [node], sparse_vectors=[models.SparseVector(indices=[1], values=[1.0])]
        )
        vectors = await store.load_vectors([point_id])
        assert set(vectors[point_id]) == {"sparse"}
        info = await client.get_collection("sparse_queries")
        assert info.config.params.sparse_vectors["sparse"].modifier is None
        with pytest.raises(ValueError, match="IDF"):
            await store.ensure_collection(dimensions=None, sparse=True, role="corpus")
        await client.close()

    asyncio.run(exercise())


@pytest.mark.skipif(
    os.environ.get("VDU_INTEGRATION") != "1", reason="requires real Qdrant"
)
def test_native_qdrant_hybrid_groups_unique_pages():
    async def exercise():
        client = AsyncQdrantClient(
            url=os.environ.get("QDRANT_URL", "http://localhost:6333")
        )
        collection = f"vdu_test_{uuid4().hex}"
        store = create_vector_store(collection, aclient=client)
        pages = [str(uuid4()), str(uuid4())]
        dataset = str(uuid4())
        try:
            await store.ensure_collection(dimensions=2, sparse=True, role="corpus")
            nodes = [
                TextNode(
                    id_=str(uuid4()),
                    text="report",
                    embedding=vector,
                    metadata={"corpus_id": page, "dataset_id": dataset},
                )
                for page, vector in [
                    (pages[0], [1.0, 0.0]),
                    (pages[0], [0.9, 0.1]),
                    (pages[1], [0.0, 1.0]),
                ]
            ]
            await store.async_add(
                nodes,
                sparse_vectors=[
                    models.SparseVector(indices=[1], values=[value])
                    for value in [1.0, 0.9, 0.5]
                ],
            )
            groups = await store.retrieve_pages(
                {
                    "dense": [1.0, 0.0],
                    "sparse": models.SparseVector(indices=[1], values=[1.0]),
                },
                mode="hybrid",
                page_top_k=2,
                prefetch_limit=3,
            )
            assert len(groups) == 2
            assert {str(group.id) for group in groups} == set(pages)
            assert all(len(group.hits) == 1 for group in groups)
            info = await client.get_collection(collection)
            assert (
                info.config.params.sparse_vectors["sparse"].modifier
                == models.Modifier.IDF
            )
        finally:
            await client.delete_collection(collection)
            await client.close()

    asyncio.run(exercise())


def test_llamaindex_preserves_precomputed_vectors_without_encoder_calls():
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.core.embeddings import MockEmbedding

    calls = []

    class GuardEmbedding(MockEmbedding):
        async def _aget_text_embeddings(self, texts):
            calls.extend(texts)
            raise AssertionError("Precomputed nodes must not be embedded again")

    async def exercise():
        client = AsyncQdrantClient(":memory:")
        store = create_vector_store("dense_precomputed", aclient=client)
        await store.ensure_collection(dimensions=2, sparse=False, role="query")
        index = VectorStoreIndex(
            nodes=[],
            storage_context=StorageContext.from_defaults(vector_store=store),
            embed_model=GuardEmbedding(embed_dim=2),
        )
        node = TextNode(
            id_=str(uuid4()), text="precomputed query", embedding=[1.0, 0.0]
        )
        await index.ainsert_nodes([node])
        saved = await store.load_vectors([node.node_id])
        assert saved[node.node_id]["dense"] == [1.0, 0.0]
        assert calls == []
        await client.close()

    asyncio.run(exercise())
