import logging
from functools import lru_cache, wraps

import openai
from loguru import logger
from openai.types.chat import ChatCompletion
from openai.types.create_embedding_response import CreateEmbeddingResponse
from PIL import Image
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_fixed

from ..settings import get_settings

settings = get_settings()


def format_message_content_for_log(content: str | list) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            parts.append(str(part))
            continue
        if part.get("type") == "image_url":
            image_url = part.get("image_url", {})
            url = (
                image_url.get("url", "")
                if isinstance(image_url, dict)
                else str(image_url)
            )
            b64 = url.split("base64,", 1)[-1] if "base64," in url else url
            parts.append(f"<-- IMAGE {b64[:5]} .. {b64[-5:]} -->")
        elif part.get("type") == "text":
            parts.append(part.get("text", ""))
        else:
            parts.append(str(part))
    return "\n\n".join(part for part in parts if part)


@lru_cache
def get_openai_client(api_key: str, api_base: str) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        api_key=api_key,
        base_url=api_base,
    )


@retry(
    stop=stop_after_attempt(settings.openai_max_retries),
    wait=wait_fixed(settings.openai_timeout),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def create_chat_completion(
    client: openai.AsyncOpenAI,
    parse: bool = False,
    *args,
    **kwargs,
) -> ChatCompletion:
    if messages := kwargs.get("messages", []):
        log_method = logger.info if settings.log_chat_completion_input else logger.debug
        log_method(
            f"Chat completion input with t={kwargs.get('temperature', None)}:\n"
            + "\n".join(
                [
                    f"{message['role']}: {format_message_content_for_log(message['content'])}"
                    for message in messages
                ]
            )
        )
    func = (
        client.chat.completions.create
        if not parse
        else client.beta.chat.completions.parse
    )
    out = await func(*args, **kwargs)
    if out is None:
        logger.error("Chat completion returned None")
        return None
    if parse:
        logger.debug(
            f"Chat completion output:\n{out.choices[0].message.parsed.model_dump_json(indent=2)}"
        )
    else:
        logger.debug(f"Chat completion output:\n{out.choices[0].message.content}")
    logger.debug(f"Chat completion usage:{out.usage.model_dump_json()}")
    return out


@retry(
    stop=stop_after_attempt(settings.openai_max_retries),
    wait=wait_fixed(settings.openai_timeout),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def create_embeddings(
    client: openai.AsyncOpenAI,
    *args,
    **kwargs,
) -> CreateEmbeddingResponse:
    return await client.embeddings.create(*args, **kwargs)
