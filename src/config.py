"""Validated, serializable pipeline contracts. Credentials are environment references."""

import os
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EndpointProfile(Config):
    model: str
    base_url: str = "http://localhost:8000/v1"
    api_key_env: str = "OPENAI_API_KEY"
    timeout_seconds: float = Field(default=600, gt=0)
    max_retries: int = Field(default=3, ge=1)
    retry_wait_seconds: float = Field(default=2, ge=0)
    concurrency: int = Field(default=8, ge=1)
    extra_body: dict[str, Any] = Field(default_factory=dict)

    def resolve_api_key(self) -> str:
        """Resolve credentials without adding them to serialized configuration."""
        from dotenv import dotenv_values

        value = os.environ.get(self.api_key_env)
        if value is None:
            value = dotenv_values(Path(__file__).resolve().parents[1] / ".env").get(
                self.api_key_env
            )
        return value or "unused"


class PreprocessConfig(Config):
    endpoint: EndpointProfile
    prompt: str = "Convert the document to markdown."
    page_limit: int | None = Field(default=None, gt=0)


class ChunkConfig(Config):
    max_chars: int = Field(default=4000, ge=1)
    overlap_chars: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def check_overlap(self):
        if self.overlap_chars >= self.max_chars:
            raise ValueError("overlap_chars must be smaller than max_chars")
        return self


class DenseConfig(Config):
    endpoint: EndpointProfile
    space_id: str = Field(min_length=1)
    dimensions: int | None = Field(default=None, gt=0)
    modality: Literal["text", "image"] = "text"
    adapter: Literal["openai", "vllm"] = "openai"
    query_instruction: str = ""
    document_instruction: str = ""
    distance: Literal["cosine", "dot", "euclid"] = "cosine"

    @model_validator(mode="after")
    def check_modality_adapter(self):
        if self.modality == "image" and self.adapter != "vllm":
            raise ValueError("Image embeddings require the vllm request adapter")
        return self


class SparseConfig(Config):
    model: Literal["Qdrant/bm25"] = "Qdrant/bm25"
    language: str = "english"
    k: float = Field(default=1.2, gt=0)
    b: float = Field(default=0.75, ge=0, le=1)
    avg_len: float = Field(default=256, gt=0)
    token_max_length: int = Field(default=40, gt=0)
    disable_stemmer: bool = False


class EmbeddingConfig(Config):
    dense: DenseConfig | None = None
    sparse: SparseConfig | None = None
    batch_size: int = Field(default=64, gt=0)

    @model_validator(mode="after")
    def check_provider(self):
        if self.dense is None and self.sparse is None:
            raise ValueError("Select at least one dense or sparse encoder")
        return self


class RetrievalConfig(Config):
    mode: Literal["dense", "sparse", "hybrid"] = "dense"
    page_top_k: int = Field(default=20, gt=0)
    prefetch_limit: int | None = Field(default=None, gt=0)


class ContextConfig(Config):
    representations: list[str] = Field(default_factory=lambda: ["ocr"], min_length=1)
    representation_run_id: UUID | None = None
    page_top_k: int = Field(default=5, gt=0)
    max_text_chars: int = Field(default=100000, gt=0)
    max_image_bytes: int = Field(default=20000000, gt=0)


class GenerationConfig(Config):
    endpoint: EndpointProfile
    context: ContextConfig = Field(default_factory=ContextConfig)
    prompt: str = "Answer the question using only the supplied document pages. If the pages do not contain the answer, say so."
    temperature: float = Field(default=0, ge=0)
    max_tokens: int = Field(default=2048, gt=0)


class IRConfig(Config):
    cutoffs: list[int] = Field(default_factory=lambda: [1, 5, 10, 20], min_length=1)
    relevance_threshold: int = Field(default=1, ge=0)

    @model_validator(mode="after")
    def check_cutoffs(self):
        if any(k < 1 for k in self.cutoffs) or len(set(self.cutoffs)) != len(
            self.cutoffs
        ):
            raise ValueError("IR cutoffs must be distinct positive integers")
        return self


class MetricConfig(Config):
    id: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class RagasConfig(Config):
    metrics: list[MetricConfig] = Field(default_factory=list)
    judge: EndpointProfile | None = None
    embeddings: DenseConfig | None = None
    ragas_max_concurrency: int = Field(default=4, gt=0)
    context: ContextConfig | None = None


class QuerySelection(Config):
    languages: list[str] | None = None
    rephrase_levels: list[int] = Field(default_factory=lambda: [0, 1, 2, 3])
    limit: int | None = Field(default=None, gt=0)


class ExperimentConfig(Config):
    name: str
    dataset_id: UUID
    queries: QuerySelection = Field(default_factory=QuerySelection)
    preprocess: PreprocessConfig | None = None
    chunking: ChunkConfig = Field(default_factory=ChunkConfig)
    embeddings: EmbeddingConfig
    corpus_unit: Literal["chunk", "page"] = "chunk"
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig | None = None
    ir: IRConfig | None = Field(default_factory=IRConfig)
    ragas: RagasConfig = Field(default_factory=RagasConfig)
    reuse: dict[str, UUID] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_pipeline(self):
        if self.ir and max(self.ir.cutoffs) > self.retrieval.page_top_k:
            raise ValueError("IR cutoffs exceed retrieval.page_top_k")
        if (
            self.generation
            and self.generation.context.page_top_k > self.retrieval.page_top_k
        ):
            raise ValueError("Generation page count exceeds retrieval.page_top_k")
        if self.ragas.metrics and not self.generation:
            # Context-only Ragas can be called independently from the CLI.
            raise ValueError("Experiment Ragas evaluation requires generation")
        return self


def load_config(path: str | Path, model: type[Config]):
    with Path(path).open(encoding="utf-8") as stream:
        return model.model_validate(yaml.safe_load(stream))
