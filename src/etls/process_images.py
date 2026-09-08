from datetime import datetime as dt

import pandas as pd
from loguru import logger
from PIL import Image
from pydantic import BaseModel

from ..settings import get_settings
from ..utils.concurrency_utils import execute_with_semaphore
from ..utils.image_utils import image2base64
from ..utils.openai_utils import create_chat_completion, get_openai_client
from .load_datasets import dataset_loaders

settings = get_settings()

prompts = {
    "deepseek-ocr": "Convert the document to markdown.",
}


async def process_image(
    image: Image,
    model: str = settings.openai_vlm_preprocess_model,
    prompt_id: str = "deepseek-ocr",
    prompt: str | None = None,
    timeout: float = settings.openai_timeout,
) -> str:
    client = get_openai_client(
        settings.openai_vlm_preprocess_api_key, settings.openai_vlm_preprocess_api_base
    )
    if prompt is None:
        prompt = prompts[prompt_id]
    image_base64 = image2base64(image)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    res = await create_chat_completion(
        client,
        model=model,
        messages=messages,
        timeout=timeout,
    )
    return res.choices[0].message.content


async def process_images(
    dataset: str,
    model: str = settings.openai_vlm_preprocess_model,
    prompt_id: str = "deepseek-ocr",
    prompt: str | None = None,
    max_concurrency: int = settings.openai_chat_completion_max_concurrency,
    show_progress: bool = True,
    save_results: bool = True,
    limit: int | None = None,
) -> tuple[pd.DataFrame, str]:
    datasets = dataset_loaders[dataset]()
    dataset_hf_name, data = next(iter(datasets.items()))
    logger.info(f"Loaded {dataset_hf_name} dataset")
    images = data["corpus"]["image"]
    ids = data["corpus"]["corpus-id"]
    if limit is not None:
        images = images[:limit]
        ids = ids[:limit]
    results = await execute_with_semaphore(
        [process_image(image, model, prompt_id, prompt) for image in images],
        max_concurrency=max_concurrency,
        show_progress=show_progress,
    )
    logger.info(f"Processed {len(images)} images")
    timestamp = dt.now().strftime("%Y%m%d-%H%M%S")
    dataset_short_name = dataset.split("/")[1]
    fstem = f"{timestamp}_{dataset_short_name}_{model}_{prompt_id}"
    fname = f"{fstem}_i2t.csv"
    save_path = settings.path_data_processed / fstem / "i2t" / fname
    df = pd.DataFrame({"corpus-id": ids, "text": results})
    if save_results:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info(f"Saved results to {save_path}")
        return df, save_path
    else:
        return df, None
