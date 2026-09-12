# Agent guidelines

## Formatting

After changing Python code, always format with:

```bash
./scripts/format.sh
```

## Implementation style

Prefer a readable, inlined implementation over a thicket of helpers.

- Keep logic in the function that owns the task so a reader can follow it top to bottom.
- Extract a helper only when it is reused, hides a real boundary, or would otherwise make the caller harder to read.
- Do not split a one-off into tiny functions for "cleanliness." Granularity is for necessity, not default.
- Do not preserve backward compatibility unless it is explicitly requested. Prefer a clean break: delete unused APIs, update all call sites, and skip shims, aliases, deprecation wrappers, and dual-path logic.

```python
# Avoid: one-off helpers that force the reader to jump around
def _strip_chunk(chunk: str) -> str:
    return chunk.strip()

def _should_keep(chunk: str) -> bool:
    return bool(chunk)

for chunk in chunks:
    chunk = _strip_chunk(chunk)
    if _should_keep(chunk):
        nodes.append(TextNode(text=chunk, id_=f"{doc_id}:{i}"))

# Prefer: the same steps, visible in place
for i, chunk in enumerate(chunks):
    chunk = chunk.strip()
    if not chunk:
        continue
    nodes.append(TextNode(text=chunk, id_=f"{doc_id}:{i}"))
```

## Retries

Use [tenacity](https://github.com/jd/tenacity) for retries. Follow the existing pattern: `@retry` with `stop_after_attempt`, a wait strategy, `before_sleep_log`, and `reraise=True`.

```python
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_fixed

@retry(
    stop=stop_after_attempt(settings.openai_max_retries),
    wait=wait_fixed(settings.openai_timeout),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def create_chat_completion(...):
    ...
```

Do not invent ad-hoc retry loops, `time.sleep` backoff, or custom retry decorators.

## Vector operations

Use LlamaIndex for embeddings, nodes, indexes, and vector stores. Do not drop down to raw Qdrant or OpenAI embedding calls for indexing/retrieval unless LlamaIndex cannot express the operation.

Use `TextNode`, `VectorStoreIndex`, `StorageContext`, and `QdrantVectorStore` as in `src/etls/vectorize_docs.py`.
