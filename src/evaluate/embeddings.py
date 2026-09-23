"""LlamaIndex dense adapters and role-aware BM25 encoding.

Image embedding follows vLLM's /embeddings ``messages`` extension. Endpoints
must return a single pooled vector; token/patch multivectors are not supported.
"""

import asyncio
import logging
import math
from typing import Any

from llama_index.core.embeddings import MultiModalEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from openai import AsyncOpenAI, OpenAI
from openai.types import CreateEmbeddingResponse
from pydantic import PrivateAttr
from qdrant_client import models
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_fixed

from src.config import DenseConfig, SparseConfig

logger = logging.getLogger(__name__)


def validate_dense(vector: list[float], dimensions: int | None = None) -> list[float]:
    if not vector or any(
        not isinstance(x, (int, float)) or not math.isfinite(x) for x in vector
    ):
        raise ValueError("Dense embeddings must be nonempty finite scalar vectors")
    if dimensions is not None and len(vector) != dimensions:
        raise ValueError(f"Expected {dimensions} dimensions, received {len(vector)}")
    return [float(value) for value in vector]


def validate_sparse(indices: list[int], values: list[float]) -> models.SparseVector:
    if len(indices) != len(values) or len(indices) != len(set(indices)):
        raise ValueError(
            "Sparse indices and values must have equal lengths and unique indices"
        )
    if any(not isinstance(i, int) or i < 0 or i > 2**32 - 1 for i in indices):
        raise ValueError("Sparse indices must be unsigned 32-bit integers")
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError("Sparse values must be finite scalars")
    ordered = sorted(zip(indices, values))
    return models.SparseVector(
        indices=[i for i, _ in ordered], values=[float(v) for _, v in ordered]
    )


class VLLMEmbedding(MultiModalEmbedding):
    """The narrow LlamaIndex adapter for vLLM image/text chat embeddings."""

    _config: DenseConfig = PrivateAttr()
    _sync_client: OpenAI = PrivateAttr()
    _async_client: AsyncOpenAI = PrivateAttr()

    def __init__(self, config: DenseConfig, batch_size: int = 64):
        super().__init__(model_name=config.endpoint.model, embed_batch_size=batch_size)
        self._config = config
        kwargs = dict(
            api_key=config.endpoint.resolve_api_key(),
            base_url=config.endpoint.base_url,
            timeout=config.endpoint.timeout_seconds,
            max_retries=0,
        )
        self._sync_client = OpenAI(**kwargs)
        self._async_client = AsyncOpenAI(**kwargs)

    def _body(self, content: list[dict[str, Any]]) -> dict[str, Any]:
        body = dict(self._config.endpoint.extra_body)
        body.update(
            model=self.model_name,
            encoding_format="float",
            messages=[{"role": "user", "content": content}],
        )
        if self._config.dimensions is not None:
            body["dimensions"] = self._config.dimensions
        return body

    def _request(self, content: list[dict[str, Any]]) -> list[float]:
        endpoint = self._config.endpoint

        @retry(
            stop=stop_after_attempt(endpoint.max_retries),
            wait=wait_fixed(endpoint.retry_wait_seconds),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        def request():
            response = self._sync_client.post(
                "/embeddings", cast_to=CreateEmbeddingResponse, body=self._body(content)
            )
            if len(response.data) != 1:
                raise ValueError(
                    "Expected one pooled embedding for each multimodal input"
                )
            return validate_dense(response.data[0].embedding, self._config.dimensions)

        return request()

    async def _arequest(self, content: list[dict[str, Any]]) -> list[float]:
        endpoint = self._config.endpoint

        @retry(
            stop=stop_after_attempt(endpoint.max_retries),
            wait=wait_fixed(endpoint.retry_wait_seconds),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        async def request():
            response = await self._async_client.post(
                "/embeddings", cast_to=CreateEmbeddingResponse, body=self._body(content)
            )
            if len(response.data) != 1:
                raise ValueError(
                    "Expected one pooled embedding for each multimodal input"
                )
            return validate_dense(response.data[0].embedding, self._config.dimensions)

        return await request()

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._request(
            [{"type": "text", "text": self._config.query_instruction + query}]
        )

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return await self._arequest(
            [{"type": "text", "text": self._config.query_instruction + query}]
        )

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._request(
            [{"type": "text", "text": self._config.document_instruction + text}]
        )

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return await self._arequest(
            [{"type": "text", "text": self._config.document_instruction + text}]
        )

    def _get_image_embedding(self, img_file_path: str) -> list[float]:
        return self._request(
            [
                {"type": "image_url", "image_url": {"url": img_file_path}},
                {"type": "text", "text": self._config.document_instruction},
            ]
        )

    async def _aget_image_embedding(self, img_file_path: str) -> list[float]:
        return await self._arequest(
            [
                {"type": "image_url", "image_url": {"url": img_file_path}},
                {"type": "text", "text": self._config.document_instruction},
            ]
        )

    async def aclose(self):
        self._sync_client.close()
        await self._async_client.close()


def create_dense_model(config: DenseConfig, batch_size: int = 64):
    if config.adapter == "vllm":
        return VLLMEmbedding(config, batch_size)
    if config.modality == "image":
        raise ValueError("Image embeddings require the vllm adapter")
    return OpenAIEmbedding(
        # model_name overrides both engines without constraining custom model names.
        model_name=config.endpoint.model,
        api_base=config.endpoint.base_url,
        api_key=config.endpoint.resolve_api_key(),
        dimensions=config.dimensions,
        embed_batch_size=batch_size,
        num_workers=config.endpoint.concurrency,
        timeout=config.endpoint.timeout_seconds,
        max_retries=0,
        additional_kwargs={"extra_body": config.endpoint.extra_body},
    )


async def encode_dense(
    model, config: DenseConfig, inputs: list[str], role: str
) -> list[list[float]]:
    """Encode strings/data URIs through LlamaIndex, preserving input order."""
    semaphore = asyncio.Semaphore(config.endpoint.concurrency)

    async def encode_one(value):
        async with semaphore:
            if role == "query":
                text = (
                    value
                    if isinstance(model, VLLMEmbedding)
                    else config.query_instruction + value
                )
                return await model.aget_query_embedding(text)
            if config.modality == "image":
                return await model.aget_image_embedding(value)
            text = (
                value
                if isinstance(model, VLLMEmbedding)
                else config.document_instruction + value
            )
            return await model.aget_text_embedding(text)

    @retry(
        stop=stop_after_attempt(config.endpoint.max_retries),
        wait=wait_fixed(config.endpoint.retry_wait_seconds),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def encode_text_batch():
        return await model.aget_text_embedding_batch(
            [config.document_instruction + value for value in inputs],
            show_progress=False,
        )

    if (
        role != "query"
        and config.modality == "text"
        and not isinstance(model, VLLMEmbedding)
    ):
        vectors = await encode_text_batch()
    else:
        # VLLMEmbedding has its own Tenacity policy; ordinary query calls need one.
        operation = encode_one
        if not isinstance(model, VLLMEmbedding):
            operation = retry(
                stop=stop_after_attempt(config.endpoint.max_retries),
                wait=wait_fixed(config.endpoint.retry_wait_seconds),
                before_sleep=before_sleep_log(logger, logging.WARNING),
                reraise=True,
            )(encode_one)
        vectors = await asyncio.gather(*(operation(value) for value in inputs))
    if len(vectors) != len(inputs):
        raise ValueError("Embedding endpoint returned the wrong number of vectors")
    dimensions = config.dimensions or (len(vectors[0]) if vectors else None)
    return [validate_dense(vector, dimensions) for vector in vectors]


class BM25Encoder:
    def __init__(self, config: SparseConfig):
        from fastembed import SparseTextEmbedding

        self.model = SparseTextEmbedding(
            model_name=config.model,
            language=config.language,
            k=config.k,
            b=config.b,
            avg_len=config.avg_len,
            token_max_length=config.token_max_length,
            disable_stemmer=config.disable_stemmer,
        )

    def encode(self, texts: list[str], role: str) -> list[models.SparseVector]:
        # Query term weights differ from document BM25 length normalization.
        output = (
            self.model.query_embed(texts)
            if role == "query"
            else self.model.embed(texts)
        )
        vectors = [
            validate_sparse(
                [int(i) for i in item.indices], [float(v) for v in item.values]
            )
            for item in output
        ]
        if len(vectors) != len(texts):
            raise ValueError("Sparse encoder returned the wrong number of vectors")
        return vectors

    def document_callback(self, texts):
        vectors = self.encode(texts, "corpus")
        return [v.indices for v in vectors], [v.values for v in vectors]

    def query_callback(self, texts):
        vectors = self.encode(texts, "query")
        return [v.indices for v in vectors], [v.values for v in vectors]
