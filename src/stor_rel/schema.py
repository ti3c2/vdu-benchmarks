"""Relational benchmark records. Vectors and binary assets live outside SQL."""

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    CheckConstraint,
    Computed,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(table_name)s_%(column_0_name)s",
            "uq": "uq_%(table_name)s_%(column_0_N_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )
    type_annotation_map = {dict: JSON().with_variant(JSONB, "postgresql")}


class Record:
    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class DatasetOwned(Record):
    dataset_id: Mapped[UUID] = mapped_column(
        ForeignKey("datasets.id", ondelete="RESTRICT"), index=True
    )


def scoped_constraints(*constraints):
    """Every child can use a composite FK to enforce dataset isolation."""
    return (UniqueConstraint("id", "dataset_id"), *constraints)


def dataset_fk(column: str, table: str):
    return ForeignKeyConstraint(
        [column, "dataset_id"], [f"{table}.id", f"{table}.dataset_id"]
    )


class Dataset(Record, Base):
    __tablename__ = "datasets"
    source: Mapped[str] = mapped_column(Text)
    subset: Mapped[str] = mapped_column(Text, default="")
    split: Mapped[str] = mapped_column(Text, default="train")
    revision: Mapped[str] = mapped_column(Text, default="")
    fingerprint: Mapped[str] = mapped_column(String(64), unique=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    metadata_json: Mapped[dict] = mapped_column(default=dict)

    __table_args__ = (UniqueConstraint("source", "subset", "split", "revision"),)


class Doc(DatasetOwned, Base):
    __tablename__ = "docs"
    original_id: Mapped[str] = mapped_column(Text)
    __table_args__ = scoped_constraints(UniqueConstraint("dataset_id", "original_id"))


class Asset(DatasetOwned, Base):
    __tablename__ = "assets"
    bucket: Mapped[str] = mapped_column(Text)
    object_key: Mapped[str] = mapped_column(Text)
    version_id: Mapped[str | None] = mapped_column(Text)
    sha256: Mapped[str] = mapped_column(String(64))
    mime_type: Mapped[str] = mapped_column(Text)
    size_bytes: Mapped[int] = mapped_column(Integer)
    __table_args__ = scoped_constraints(
        UniqueConstraint("dataset_id", "bucket", "object_key"),
        CheckConstraint("size_bytes >= 0", name="nonnegative_size"),
    )


class Corpus(DatasetOwned, Base):
    __tablename__ = "corpus"
    doc_id: Mapped[UUID] = mapped_column(Uuid, index=True)
    original_id: Mapped[str] = mapped_column(Text)
    image_filename: Mapped[str | None] = mapped_column(Text)
    asset_id: Mapped[UUID] = mapped_column(Uuid)
    __table_args__ = scoped_constraints(
        UniqueConstraint("dataset_id", "original_id"),
        dataset_fk("doc_id", "docs"),
        dataset_fk("asset_id", "assets"),
    )


class Query(DatasetOwned, Base):
    __tablename__ = "queries"
    original_id: Mapped[str] = mapped_column(Text)
    query: Mapped[str] = mapped_column(Text)
    language: Mapped[str | None] = mapped_column(Text)
    rephrase_of_id: Mapped[UUID | None] = mapped_column(Uuid, index=True)
    rephrase_level: Mapped[int] = mapped_column(Integer, default=0)
    # Zero in the parent FK prevents chains of rephrases at the database level.
    parent_level: Mapped[int | None] = mapped_column(
        Integer, Computed("CASE WHEN rephrase_of_id IS NULL THEN NULL ELSE 0 END")
    )
    __table_args__ = scoped_constraints(
        UniqueConstraint("dataset_id", "original_id", "rephrase_level"),
        UniqueConstraint("id", "dataset_id", "rephrase_level"),
        CheckConstraint(
            "(rephrase_of_id IS NULL AND rephrase_level = 0) OR (rephrase_of_id IS NOT NULL AND rephrase_level > 0)",
            name="base_or_rephrase",
        ),
        ForeignKeyConstraint(
            ["rephrase_of_id", "dataset_id", "parent_level"],
            ["queries.id", "queries.dataset_id", "queries.rephrase_level"],
        ),
    )


class Qrel(DatasetOwned, Base):
    __tablename__ = "qrels"
    query_id: Mapped[UUID] = mapped_column(Uuid, index=True)
    corpus_id: Mapped[UUID] = mapped_column(Uuid, index=True)
    answer: Mapped[str | None] = mapped_column(Text)
    score: Mapped[float] = mapped_column(Float)
    query_level: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    __table_args__ = scoped_constraints(
        UniqueConstraint("dataset_id", "query_id", "corpus_id"),
        CheckConstraint("query_level = 0", name="base_query_only"),
        ForeignKeyConstraint(
            ["query_id", "dataset_id", "query_level"],
            ["queries.id", "queries.dataset_id", "queries.rephrase_level"],
        ),
        dataset_fk("corpus_id", "corpus"),
    )


class StageRun(DatasetOwned, Base):
    __tablename__ = "stage_runs"
    kind: Mapped[str] = mapped_column(String(40), index=True)
    config: Mapped[dict] = mapped_column(default=dict)
    fingerprint: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    error: Mapped[str | None] = mapped_column(Text)
    expected_count: Mapped[int] = mapped_column(Integer, default=0)
    completed_count: Mapped[int] = mapped_column(Integer, default=0)
    failed_count: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    provenance: Mapped[dict] = mapped_column(default=dict)
    metadata_json: Mapped[dict] = mapped_column(default=dict)
    __table_args__ = scoped_constraints(
        UniqueConstraint("dataset_id", "kind", "fingerprint"),
        CheckConstraint(
            "status IN ('pending','running','completed','failed','partial')",
            name="valid_status",
        ),
        CheckConstraint(
            "expected_count >= 0 AND completed_count >= 0 AND failed_count >= 0",
            name="nonnegative_counts",
        ),
    )


class RunDependency(DatasetOwned, Base):
    __tablename__ = "run_dependencies"
    run_id: Mapped[UUID] = mapped_column(Uuid, index=True)
    input_run_id: Mapped[UUID] = mapped_column(Uuid)
    role: Mapped[str] = mapped_column(String(80))
    __table_args__ = scoped_constraints(
        UniqueConstraint("run_id", "role"),
        dataset_fk("run_id", "stage_runs"),
        dataset_fk("input_run_id", "stage_runs"),
        CheckConstraint("run_id <> input_run_id", name="no_self_dependency"),
    )


class RunItem(DatasetOwned, Base):
    __tablename__ = "run_items"
    run_id: Mapped[UUID] = mapped_column(Uuid, index=True)
    query_id: Mapped[UUID | None] = mapped_column(Uuid)
    corpus_id: Mapped[UUID | None] = mapped_column(Uuid)
    chunk_id: Mapped[UUID | None] = mapped_column(Uuid)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    error: Mapped[str | None] = mapped_column(Text)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    __table_args__ = scoped_constraints(
        dataset_fk("run_id", "stage_runs"),
        dataset_fk("query_id", "queries"),
        dataset_fk("corpus_id", "corpus"),
        dataset_fk("chunk_id", "chunks"),
        UniqueConstraint("run_id", "query_id"),
        UniqueConstraint("run_id", "corpus_id"),
        UniqueConstraint("run_id", "chunk_id"),
        CheckConstraint(
            "(CASE WHEN query_id IS NOT NULL THEN 1 ELSE 0 END + CASE WHEN corpus_id IS NOT NULL THEN 1 ELSE 0 END + CASE WHEN chunk_id IS NOT NULL THEN 1 ELSE 0 END) = 1",
            name="exactly_one_subject",
        ),
    )


class PageRepresentation(DatasetOwned, Base):
    __tablename__ = "page_representations"
    run_id: Mapped[UUID] = mapped_column(Uuid, index=True)
    corpus_id: Mapped[UUID] = mapped_column(Uuid, index=True)
    kind: Mapped[str] = mapped_column(String(80))
    text: Mapped[str | None] = mapped_column(Text)
    asset_id: Mapped[UUID | None] = mapped_column(Uuid)
    metadata_json: Mapped[dict] = mapped_column(default=dict)
    __table_args__ = scoped_constraints(
        UniqueConstraint("run_id", "corpus_id", "kind"),
        UniqueConstraint("id", "dataset_id", "corpus_id"),
        dataset_fk("run_id", "stage_runs"),
        dataset_fk("corpus_id", "corpus"),
        dataset_fk("asset_id", "assets"),
        CheckConstraint("text IS NOT NULL OR asset_id IS NOT NULL", name="has_content"),
    )


class Chunk(DatasetOwned, Base):
    __tablename__ = "chunks"
    run_id: Mapped[UUID] = mapped_column(Uuid, index=True)
    corpus_id: Mapped[UUID] = mapped_column(Uuid, index=True)
    representation_id: Mapped[UUID] = mapped_column(Uuid)
    ordinal: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(40), default="prose")
    text: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64))
    __table_args__ = scoped_constraints(
        UniqueConstraint("run_id", "representation_id", "ordinal"),
        UniqueConstraint("id", "dataset_id", "corpus_id"),
        dataset_fk("run_id", "stage_runs"),
        dataset_fk("corpus_id", "corpus"),
        ForeignKeyConstraint(
            ["representation_id", "dataset_id", "corpus_id"],
            [
                "page_representations.id",
                "page_representations.dataset_id",
                "page_representations.corpus_id",
            ],
        ),
        CheckConstraint("ordinal >= 0", name="nonnegative_ordinal"),
    )


class EmbeddingRun(DatasetOwned, Base):
    __tablename__ = "embedding_runs"
    role: Mapped[str] = mapped_column(String(20))
    collection_name: Mapped[str] = mapped_column(Text, unique=True)
    unit_kind: Mapped[str] = mapped_column(String(20))
    profiles: Mapped[dict] = mapped_column(default=dict)
    point_count: Mapped[int] = mapped_column(Integer, default=0)
    __table_args__ = scoped_constraints(
        dataset_fk("id", "stage_runs"),
        CheckConstraint("role IN ('query','corpus')", name="valid_role"),
        CheckConstraint(
            "unit_kind IN ('query','chunk','page')", name="valid_unit_kind"
        ),
    )


class Experiment(DatasetOwned, Base):
    __tablename__ = "experiments"
    name: Mapped[str] = mapped_column(Text)
    config: Mapped[dict] = mapped_column(default=dict)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    __table_args__ = scoped_constraints()


class ExperimentQuery(DatasetOwned, Base):
    __tablename__ = "experiment_queries"
    experiment_id: Mapped[UUID] = mapped_column(Uuid, index=True)
    query_id: Mapped[UUID] = mapped_column(Uuid)
    __table_args__ = scoped_constraints(
        UniqueConstraint("experiment_id", "query_id"),
        dataset_fk("experiment_id", "experiments"),
        dataset_fk("query_id", "queries"),
    )


class ExperimentRun(DatasetOwned, Base):
    __tablename__ = "experiment_runs"
    experiment_id: Mapped[UUID] = mapped_column(Uuid, index=True)
    run_id: Mapped[UUID] = mapped_column(Uuid)
    role: Mapped[str] = mapped_column(String(80))
    __table_args__ = scoped_constraints(
        UniqueConstraint("experiment_id", "role"),
        dataset_fk("experiment_id", "experiments"),
        dataset_fk("run_id", "stage_runs"),
    )


class RetrievalRun(DatasetOwned, Base):
    __tablename__ = "retrieval_runs"
    query_embedding_run_id: Mapped[UUID] = mapped_column(Uuid)
    corpus_embedding_run_id: Mapped[UUID] = mapped_column(Uuid)
    __table_args__ = scoped_constraints(
        dataset_fk("id", "stage_runs"),
        dataset_fk("query_embedding_run_id", "embedding_runs"),
        dataset_fk("corpus_embedding_run_id", "embedding_runs"),
    )


class Retrieval(DatasetOwned, Base):
    __tablename__ = "retrievals"
    run_id: Mapped[UUID] = mapped_column(Uuid, index=True)
    query_id: Mapped[UUID] = mapped_column(Uuid)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    latency_ms: Mapped[float | None] = mapped_column(Float)
    error: Mapped[str | None] = mapped_column(Text)
    __table_args__ = scoped_constraints(
        UniqueConstraint("run_id", "query_id"),
        UniqueConstraint("id", "dataset_id", "query_id"),
        dataset_fk("run_id", "retrieval_runs"),
        dataset_fk("query_id", "queries"),
    )


class RetrievalHit(DatasetOwned, Base):
    __tablename__ = "retrieval_hits"
    retrieval_id: Mapped[UUID] = mapped_column(Uuid, index=True)
    corpus_id: Mapped[UUID] = mapped_column(Uuid)
    chunk_id: Mapped[UUID | None] = mapped_column(Uuid)
    point_id: Mapped[UUID] = mapped_column(Uuid)
    rank: Mapped[int] = mapped_column(Integer)
    score: Mapped[float] = mapped_column(Float)
    __table_args__ = scoped_constraints(
        UniqueConstraint("retrieval_id", "rank"),
        UniqueConstraint("retrieval_id", "corpus_id"),
        UniqueConstraint("id", "dataset_id", "corpus_id"),
        dataset_fk("retrieval_id", "retrievals"),
        dataset_fk("corpus_id", "corpus"),
        ForeignKeyConstraint(
            ["chunk_id", "dataset_id", "corpus_id"],
            ["chunks.id", "chunks.dataset_id", "chunks.corpus_id"],
        ),
        CheckConstraint("rank > 0", name="positive_rank"),
    )


class GenerationRun(DatasetOwned, Base):
    __tablename__ = "generation_runs"
    retrieval_run_id: Mapped[UUID] = mapped_column(Uuid)
    __table_args__ = scoped_constraints(
        dataset_fk("id", "stage_runs"), dataset_fk("retrieval_run_id", "retrieval_runs")
    )


class Generation(DatasetOwned, Base):
    __tablename__ = "generations"
    run_id: Mapped[UUID] = mapped_column(Uuid, index=True)
    retrieval_id: Mapped[UUID] = mapped_column(Uuid)
    query_id: Mapped[UUID] = mapped_column(Uuid)
    response: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    error: Mapped[str | None] = mapped_column(Text)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    usage: Mapped[dict] = mapped_column(default=dict)
    request: Mapped[dict] = mapped_column(default=dict)
    __table_args__ = scoped_constraints(
        UniqueConstraint("run_id", "query_id"),
        dataset_fk("run_id", "generation_runs"),
        dataset_fk("query_id", "queries"),
        ForeignKeyConstraint(
            ["retrieval_id", "dataset_id", "query_id"],
            ["retrievals.id", "retrievals.dataset_id", "retrievals.query_id"],
        ),
    )


class GenerationContext(DatasetOwned, Base):
    __tablename__ = "generation_contexts"
    generation_id: Mapped[UUID] = mapped_column(Uuid, index=True)
    hit_id: Mapped[UUID] = mapped_column(Uuid)
    representation_id: Mapped[UUID | None] = mapped_column(Uuid)
    position: Mapped[int] = mapped_column(Integer)
    asset_id: Mapped[UUID | None] = mapped_column(Uuid)
    text: Mapped[str | None] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(String(80), default="ocr")
    __table_args__ = scoped_constraints(
        UniqueConstraint("generation_id", "position"),
        dataset_fk("generation_id", "generations"),
        dataset_fk("hit_id", "retrieval_hits"),
        dataset_fk("representation_id", "page_representations"),
        dataset_fk("asset_id", "assets"),
        CheckConstraint("position >= 0", name="nonnegative_position"),
    )


class EvaluationRun(DatasetOwned, Base):
    __tablename__ = "evaluation_runs"
    experiment_id: Mapped[UUID] = mapped_column(Uuid, index=True)
    retrieval_run_id: Mapped[UUID] = mapped_column(Uuid)
    generation_run_id: Mapped[UUID | None] = mapped_column(Uuid)
    framework: Mapped[str] = mapped_column(String(40))
    __table_args__ = scoped_constraints(
        dataset_fk("id", "stage_runs"),
        dataset_fk("experiment_id", "experiments"),
        dataset_fk("retrieval_run_id", "retrieval_runs"),
        dataset_fk("generation_run_id", "generation_runs"),
    )


class MetricResult(DatasetOwned, Base):
    __tablename__ = "metric_results"
    evaluation_run_id: Mapped[UUID] = mapped_column(Uuid, index=True)
    query_id: Mapped[UUID] = mapped_column(Uuid)
    metric_id: Mapped[str] = mapped_column(String(160))
    value: Mapped[float | None] = mapped_column(Float)
    raw_value: Mapped[dict | None] = mapped_column(
        JSON().with_variant(JSONB, "postgresql")
    )
    reason: Mapped[str | None] = mapped_column(Text)
    traces: Mapped[dict] = mapped_column(default=dict)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    error: Mapped[str | None] = mapped_column(Text)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    __table_args__ = scoped_constraints(
        UniqueConstraint("evaluation_run_id", "query_id", "metric_id"),
        dataset_fk("evaluation_run_id", "evaluation_runs"),
        dataset_fk("query_id", "queries"),
    )


class MetricAggregate(DatasetOwned, Base):
    __tablename__ = "metric_aggregates"
    evaluation_run_id: Mapped[UUID] = mapped_column(Uuid, index=True)
    metric_id: Mapped[str] = mapped_column(String(160))
    group_by: Mapped[str] = mapped_column(String(80), default="all")
    group_value: Mapped[str] = mapped_column(Text, default="all")
    value: Mapped[float | None] = mapped_column(Float)
    expected_count: Mapped[int] = mapped_column(Integer)
    scored_count: Mapped[int] = mapped_column(Integer)
    skipped_count: Mapped[int] = mapped_column(Integer)
    failed_count: Mapped[int] = mapped_column(Integer)
    __table_args__ = scoped_constraints(
        UniqueConstraint("evaluation_run_id", "metric_id", "group_by", "group_value"),
        dataset_fk("evaluation_run_id", "evaluation_runs"),
    )


Index(
    "ix_queries_dataset_language_level",
    Query.dataset_id,
    Query.language,
    Query.rephrase_level,
)
