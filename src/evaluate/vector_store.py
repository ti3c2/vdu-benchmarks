"""Small QdrantVectorStore extension for persisted vectors and native grouping.

LlamaIndex's stock store cannot save sparse-only points, ingest precomputed
query sparse vectors, load named vectors, or issue native grouped RRF queries.
Only those boundaries use the underlying Qdrant client here.
"""

from uuid import UUID

from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.utils import node_to_metadata_dict
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, models

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"
DISTANCES = {
    "cosine": models.Distance.COSINE,
    "dot": models.Distance.DOT,
    "euclid": models.Distance.EUCLID,
    "manhattan": models.Distance.MANHATTAN,
}


class BenchmarkVectorStore(QdrantVectorStore):
    async def ensure_collection(
        self,
        *,
        dimensions: int | None,
        sparse: bool,
        role: str,
        distance: str = "cosine",
    ):
        dense_config = (
            {
                DENSE_VECTOR: models.VectorParams(
                    size=dimensions, distance=DISTANCES[distance]
                )
            }
            if dimensions
            else {}
        )
        sparse_config = (
            {
                SPARSE_VECTOR: models.SparseVectorParams(
                    modifier=models.Modifier.IDF if role == "corpus" else None
                )
            }
            if sparse
            else {}
        )
        if not dense_config and not sparse_config:
            raise ValueError("A collection requires at least one vector representation")
        if await self._aclient.collection_exists(self.collection_name):
            info = await self._aclient.get_collection(self.collection_name)
            actual_dense = info.config.params.vectors
            actual_sparse = info.config.params.sparse_vectors or {}
            if (
                not isinstance(actual_dense, dict)
                or set(actual_dense) != set(dense_config)
                or set(actual_sparse) != set(sparse_config)
            ):
                raise ValueError("Existing collection has incompatible named vectors")
            if dimensions and (
                actual_dense[DENSE_VECTOR].size != dimensions
                or actual_dense[DENSE_VECTOR].distance != DISTANCES[distance]
            ):
                raise ValueError(
                    "Existing collection has incompatible dense dimensions/distance"
                )
            if (
                sparse
                and actual_sparse[SPARSE_VECTOR].modifier
                != sparse_config[SPARSE_VECTOR].modifier
            ):
                raise ValueError(
                    "Existing collection has incompatible sparse IDF weighting"
                )
        else:
            await self._aclient.create_collection(
                collection_name=self.collection_name,
                vectors_config=dense_config,
                sparse_vectors_config=sparse_config or None,
            )
            if role == "corpus":
                await self._aclient.create_payload_index(
                    self.collection_name,
                    "corpus_id",
                    models.PayloadSchemaType.KEYWORD,
                    wait=True,
                )
        self._collection_initialized = True
        self._legacy_vector_format = False

    async def async_add(
        self,
        nodes: list[BaseNode],
        sparse_vectors: list[models.SparseVector] | None = None,
        **kwargs,
    ) -> list[str]:
        if sparse_vectors is None:
            # Ordinary dense storage remains in LlamaIndex.
            return await super().async_add(nodes, **kwargs)
        if len(nodes) != len(sparse_vectors):
            raise ValueError("Every node must have one sparse vector")
        points = []
        for node, sparse in zip(nodes, sparse_vectors):
            vectors = {SPARSE_VECTOR: sparse}
            if node.embedding is not None:
                vectors[DENSE_VECTOR] = node.get_embedding()
            points.append(
                models.PointStruct(
                    id=node.node_id,
                    vector=vectors,
                    payload=node_to_metadata_dict(
                        node, remove_text=False, flat_metadata=self.flat_metadata
                    ),
                )
            )
        if points:
            await self._aclient.upsert(self.collection_name, points=points, wait=True)
        return [node.node_id for node in nodes]

    async def load_vectors(self, point_ids: list[str | UUID]) -> dict[str, dict]:
        if not point_ids:
            return {}
        records = await self._aclient.retrieve(
            self.collection_name,
            ids=[str(value) for value in point_ids],
            with_vectors=True,
            with_payload=False,
        )
        result = {}
        for record in records:
            if not isinstance(record.vector, dict):
                raise ValueError("Expected saved named vectors")
            result[str(record.id)] = record.vector
        return result

    async def collection_exists(self) -> bool:
        return await self._aclient.collection_exists(self.collection_name)

    async def delete_collection(self) -> bool:
        if not await self.collection_exists():
            return False
        await self._aclient.delete_collection(self.collection_name)
        return True

    async def dense_dimensions(self) -> int:
        info = await self._aclient.get_collection(self.collection_name)
        return info.config.params.vectors[DENSE_VECTOR].size

    async def close(self):
        await self._aclient.close()

    async def point_count(self) -> int:
        return (await self._aclient.count(self.collection_name, exact=True)).count

    async def server_version(self) -> str:
        return (await self._aclient.info()).version

    async def retrieve_pages(
        self, vectors: dict, *, mode: str, page_top_k: int, prefetch_limit: int
    ):
        required = {
            "dense": [DENSE_VECTOR],
            "sparse": [SPARSE_VECTOR],
            "hybrid": [DENSE_VECTOR, SPARSE_VECTOR],
        }
        if mode not in required:
            raise ValueError(f"Unknown retrieval mode: {mode}")
        if any(name not in vectors for name in required[mode]):
            raise ValueError(f"Saved query vectors do not support {mode} retrieval")
        if prefetch_limit < 1:
            return []
        kwargs = dict(
            collection_name=self.collection_name,
            group_by="corpus_id",
            group_size=1,
            limit=page_top_k,
            with_payload=["corpus_id", "chunk_id", "dataset_id"],
            with_vectors=False,
        )
        if mode == "hybrid":
            result = await self._aclient.query_points_groups(
                **kwargs,
                prefetch=[
                    models.Prefetch(
                        query=vectors[name], using=name, limit=prefetch_limit
                    )
                    for name in required[mode]
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
            )
        else:
            name = required[mode][0]
            result = await self._aclient.query_points_groups(
                **kwargs, query=vectors[name], using=name
            )
        return result.groups


def create_vector_store(
    collection_name: str, *, aclient: AsyncQdrantClient | None = None
) -> BenchmarkVectorStore:
    if aclient is None:
        from src.settings import get_settings

        settings = get_settings()
        aclient = AsyncQdrantClient(
            url=settings.qdrant_url, api_key=settings.qdrant_api_key or None
        )
    return BenchmarkVectorStore(
        collection_name=collection_name,
        aclient=aclient,
        dense_vector_name=DENSE_VECTOR,
        sparse_vector_name=SPARSE_VECTOR,
    )
