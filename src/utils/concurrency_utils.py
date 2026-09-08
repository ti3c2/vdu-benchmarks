import asyncio
from typing import Awaitable, List, Optional

import tqdm
from loguru import logger

from ..settings import get_settings

settings = get_settings()


async def semaphore_task(semaphore, task, pbar: Optional[tqdm.tqdm] = None):
    """Execute a task with a semaphore."""
    async with semaphore:
        result = await task
        if pbar:
            pbar.update(1)
        return result


async def execute_with_semaphore(
    tasks: List[Awaitable],
    max_concurrency: int = settings.openai_chat_completion_max_concurrency,
    show_progress: bool = True,
):
    """Execute tasks with limited concurrency using a semaphore."""
    semaphore = asyncio.Semaphore(max_concurrency)
    logger.info(f"Executing {len(tasks)} tasks with max concurrency {max_concurrency}")
    pbar = (
        tqdm.tqdm(total=len(tasks), desc="Executing tasks") if show_progress else None
    )
    results = await asyncio.gather(
        *[semaphore_task(semaphore, task, pbar) for task in tasks]
    )
    if pbar:
        pbar.close()
    return results
