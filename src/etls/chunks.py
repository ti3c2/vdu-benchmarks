"""Create persisted, ordered text chunks from whole-page representations."""

import hashlib
from uuid import UUID

from src.config import ChunkConfig
from src.etls.text_transforms import split_markdown
from src.stor_rel import crud
from src.stor_rel.schema import Chunk, PageRepresentation, RunItem
from src.utils.progress import progress_bar


async def build_chunks(
    representation_run_id: UUID,
    config: ChunkConfig,
    resume_run_id: UUID | None = None,
) -> UUID:
    source = await crud.validate_run(representation_run_id, kind="preprocess")
    representations = sorted(
        await crud.find_records(PageRepresentation, run_id=source.id, kind="ocr"),
        key=lambda representation: str(representation.corpus_id),
    )
    run = await crud.start_run(
        source.dataset_id,
        "chunks",
        config.model_dump(mode="json"),
        inputs={"representations": source.id},
        resume_run_id=resume_run_id,
    )
    if run.status == "completed":
        return run.id
    items = {
        item.corpus_id: item for item in await crud.find_records(RunItem, run_id=run.id)
    }
    completed = 0
    try:
        with progress_bar(
            total=len(representations), desc="Building chunks", unit="page"
        ) as progress:
            for representation in representations:
                previous = items.get(representation.corpus_id)
                if previous and previous.status == "completed":
                    completed += 1
                    progress.update()
                    continue
                item = await crud.upsert_record(
                    RunItem,
                    {
                        "run_id": run.id,
                        "corpus_id": representation.corpus_id,
                    },
                    {
                        "dataset_id": source.dataset_id,
                        "status": "running",
                        "attempts": (previous.attempts if previous else 0) + 1,
                        "error": None,
                    },
                )
                try:
                    if not representation.text or not representation.text.strip():
                        raise ValueError("Cannot chunk an empty OCR representation")
                    for ordinal, (kind, text) in enumerate(
                        split_markdown(
                            representation.text,
                            config.max_chars,
                            config.overlap_chars,
                        )
                    ):
                        await crud.upsert_record(
                            Chunk,
                            {
                                "run_id": run.id,
                                "representation_id": representation.id,
                                "ordinal": ordinal,
                            },
                            {
                                "dataset_id": source.dataset_id,
                                "corpus_id": representation.corpus_id,
                                "kind": kind,
                                "text": text,
                                "content_hash": hashlib.sha256(
                                    text.encode()
                                ).hexdigest(),
                            },
                        )
                    await crud.update_record(
                        RunItem, item.id, status="completed", error=None
                    )
                    completed += 1
                except Exception as error:
                    await crud.update_record(
                        RunItem, item.id, status="failed", error=str(error)
                    )
                finally:
                    progress.update()
        await crud.finish_run(
            run.id,
            status="completed"
            if completed == len(representations)
            else "partial"
            if completed
            else "failed",
            expected_count=len(representations),
            completed_count=completed,
            failed_count=len(representations) - completed,
        )
    except BaseException:
        await crud.finish_run(run.id, status="failed")
        raise
    return run.id
