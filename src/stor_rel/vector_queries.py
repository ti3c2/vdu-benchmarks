"""Transactional persistence boundary for vector retrieval results."""

from uuid import UUID

from sqlalchemy import delete

from src.stor_rel.entry import get_db
from src.stor_rel.schema import Retrieval, RetrievalHit, RunItem


async def persist_ranking(
    dataset_id: UUID,
    retrieval_id: UUID,
    item_id: UUID,
    hits: list[dict],
    latency_ms: float,
):
    """Commit a complete ranking and its successful checkpoint atomically."""
    async with get_db() as db:
        record = await db.get(Retrieval, retrieval_id)
        run_item = await db.get(RunItem, item_id)
        if (
            record is None
            or run_item is None
            or record.dataset_id != dataset_id
            or run_item.dataset_id != dataset_id
            or record.run_id != run_item.run_id
            or record.query_id != run_item.query_id
        ):
            raise ValueError(
                "Ranking and run checkpoint must reference the same dataset/query/run"
            )
        await db.execute(
            delete(RetrievalHit).where(RetrievalHit.retrieval_id == retrieval_id)
        )
        for hit in hits:
            db.add(
                RetrievalHit(dataset_id=dataset_id, retrieval_id=retrieval_id, **hit)
            )
        record.status = "completed"
        record.error = None
        record.latency_ms = latency_ms
        run_item.status = "completed"
        run_item.error = None
