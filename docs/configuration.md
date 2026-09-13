# Configuration Reference

Experiment configs are YAML files that validate against `ExperimentConfig` in `src/config.py`. Unknown keys are rejected, so misspellings usually fail early instead of being ignored.

Most commands also accept smaller stage configs. For example, `vdu vectorize queries --config ...` expects only an `EmbeddingConfig`, while `vdu experiment run --config ...` expects the complete experiment shape described here.

## Complete Experiment Shape

```yaml
name: ocr-cached-dense
dataset_id: 00000000-0000-0000-0000-000000000000
queries:
  rephrase_levels: [0]
preprocess:
  ocr_text_path: data/processed/.../DeepSeek-OCR-2_deepseek-ocr_i2t.csv
chunking:
  max_chars: 4000
  overlap_chars: 0
embeddings:
  dense:
    endpoint:
      model: BAAI/bge-m3
      base_url: http://localhost:8001/v1
      api_key_env: EMBEDDING_API_KEY
    space_id: bge-m3-dense-v1
    dimensions: 1024
corpus_unit: chunk
retrieval:
  mode: dense
  page_top_k: 20
ir:
  cutoffs: [1, 5, 10, 20]
ragas:
  metrics: []
```

The required top-level fields are `name`, `dataset_id`, and `embeddings`. Other sections have defaults or are needed only for specific workflows.

## Top-Level Fields

| Field | Meaning |
| --- | --- |
| `name` | Human-readable experiment name stored in PostgreSQL. It does not need to be globally unique, but unique names make comparisons easier to read. |
| `dataset_id` | UUID returned by `vdu dataset ingest`. This selects the relational dataset snapshot. |
| `queries` | Which normalized queries to include. Defaults to all rephrase levels. |
| `preprocess` | How to create OCR/text page representations. Needed for text retrieval unless you reuse an existing representation run. |
| `chunking` | How OCR text is split for `corpus_unit: chunk`. |
| `embeddings` | Dense, sparse, or hybrid embedding profile. |
| `corpus_unit` | `chunk` or `page`. Chunk retrieval embeds OCR chunks. Page retrieval embeds one point per page. |
| `retrieval` | Retrieval mode and depth. |
| `generation` | Optional answer generation configuration. Required if `ragas.metrics` is nonempty in a full experiment. |
| `ir` | IR metric configuration. Set to `null` to skip IR evaluation. |
| `ragas` | Optional Ragas metric configuration. Empty `metrics: []` means no Ragas calls. |
| `reuse` | Optional map of completed run IDs to pin instead of recomputing stages. |

## Endpoint Profiles

Endpoint profiles appear under OCR, embedding, generation, and Ragas judge settings:

```yaml
endpoint:
  model: BAAI/bge-m3
  base_url: http://localhost:8001/v1
  api_key_env: EMBEDDING_API_KEY
  timeout_seconds: 600
  max_retries: 3
  retry_wait_seconds: 2
  concurrency: 8
  extra_body: {}
```

| Field | Meaning |
| --- | --- |
| `model` | Model name sent to the OpenAI-compatible endpoint. |
| `base_url` | Endpoint base URL, usually ending in `/v1`. |
| `api_key_env` | Name of the environment variable containing the API key. The key itself is not stored in run configs. |
| `timeout_seconds` | Per-request timeout. |
| `max_retries` | Attempts for retried model calls. |
| `retry_wait_seconds` | Fixed wait between retry attempts. |
| `concurrency` | Maximum parallel requests for that endpoint profile. |
| `extra_body` | Extra JSON body sent to compatible model APIs. Useful for provider-specific options. |

## Query Selection

```yaml
queries:
  languages: [en]
  rephrase_levels: [0]
  limit: 100
```

| Field | Meaning |
| --- | --- |
| `languages` | Optional list of query languages to include. `null` includes all languages. |
| `rephrase_levels` | Which normalized query rows to include. `0` is the original query. `1`, `2`, and `3` are rephrases. |
| `limit` | Optional cap after filtering. Useful for smoke tests. |

For base-query-only retrieval comparisons, use:

```yaml
queries:
  rephrase_levels: [0]
```

## Preprocessing

Preprocessing creates page representations with `kind: ocr`. Use either cached OCR text or a live OCR endpoint.

Cached OCR:

```yaml
preprocess:
  ocr_text_path: data/processed/.../DeepSeek-OCR-2_deepseek-ocr_i2t.csv
```

The CSV must have `corpus-id` and `text` columns. `corpus-id` is matched to the dataset-local `Corpus.original_id`. This path skips image reads and does not call the OCR endpoint.

Live OCR:

```yaml
preprocess:
  endpoint:
    model: deepseek-ai/DeepSeek-OCR-2
    base_url: http://localhost:8000/v1
    api_key_env: OCR_API_KEY
  prompt: Convert the document to markdown.
  page_limit: 100
```

| Field | Meaning |
| --- | --- |
| `ocr_text_path` | CSV path for cached OCR text. |
| `endpoint` | OCR model endpoint. Required when `ocr_text_path` is absent. |
| `prompt` | Text prompt sent with each page image for live OCR. |
| `page_limit` | Optional limit for preprocessing only the first N pages. Mainly useful for debugging. |

## Chunking

```yaml
chunking:
  max_chars: 4000
  overlap_chars: 0
```

| Field | Meaning |
| --- | --- |
| `max_chars` | Maximum characters per text chunk. |
| `overlap_chars` | Character overlap between neighboring chunks. Must be smaller than `max_chars`. |

Chunking is used when `corpus_unit: chunk`. Page-level image retrieval can skip chunking, unless sparse OCR text is also part of the page-level setup.

## Embeddings

`embeddings` must include at least one of `dense` or `sparse`. If both are present, the run stores both vector types under the same point IDs and can be used for hybrid retrieval.

Dense-only example:

```yaml
embeddings:
  dense:
    endpoint:
      model: BAAI/bge-m3
      base_url: http://localhost:8001/v1
      api_key_env: EMBEDDING_API_KEY
    space_id: bge-m3-dense-v1
    dimensions: 1024
    modality: text
    adapter: openai
    distance: cosine
    query_instruction: ""
    document_instruction: ""
  batch_size: 64
```

Sparse-only example:

```yaml
embeddings:
  sparse:
    model: Qdrant/bm25
    language: english
    k: 1.2
    b: 0.75
    avg_len: 256
    token_max_length: 40
    disable_stemmer: false
```

Hybrid example:

```yaml
embeddings:
  dense:
    endpoint:
      model: BAAI/bge-m3
      base_url: http://localhost:8001/v1
      api_key_env: EMBEDDING_API_KEY
    space_id: bge-m3-dense-v1
    dimensions: 1024
  sparse:
    model: Qdrant/bm25
```

### Dense Fields

| Field | Meaning |
| --- | --- |
| `endpoint` | Embedding model endpoint. |
| `space_id` | Local compatibility label for the dense vector space. Query and corpus embedding runs must have the same `space_id` for dense or hybrid retrieval. |
| `dimensions` | Expected dense vector length. If omitted, it can be inferred from the first completed dense insertion, then stored. |
| `modality` | `text` or `image`. Chunk vectors require `text`; image vectors use page-level embedding. |
| `adapter` | `openai` for standard text embeddings, `vllm` for this repo's multimodal embedding request format. Image embeddings require `vllm`. |
| `query_instruction` | Prefix/instruction used when encoding queries. |
| `document_instruction` | Prefix/instruction used when encoding corpus text. |
| `distance` | Qdrant dense distance: `cosine`, `dot`, or `euclid`. |

### What `space_id` Does

`space_id` is a repo-side guardrail. It is stored in PostgreSQL with the embedding run profile. It is not sent to the embedding endpoint.

Dense retrieval loads already persisted query vectors and corpus vectors from Qdrant. Before searching, the code checks that query and corpus dense profiles match on:

- `space_id`
- `dimensions`
- `distance`

If any of those differ, retrieval fails instead of comparing incompatible vectors.

Use a stable value that names the vector space, not just the endpoint. Good examples:

```yaml
space_id: bge-m3-dense-v1
space_id: qwen3-embedding-8b-dense-1024-v1
space_id: deepseek-ocr-image-shared-v1
```

Change `space_id` when any change makes vectors incomparable:

- different embedding model
- different output dimensions
- different modality or adapter
- different pooling/projection setup on the server
- different instructions if those materially define the vector space for your comparison

Keep `space_id` the same when only operational details change, such as `base_url`, `api_key_env`, timeout, retry count, or batch size, and the returned vectors are semantically the same.

### Sparse Fields

| Field | Meaning |
| --- | --- |
| `model` | Currently `Qdrant/bm25`. |
| `language` | Language for tokenizer/stopword/stemming behavior. |
| `k` | BM25 term-frequency saturation parameter. |
| `b` | BM25 length-normalization parameter. |
| `avg_len` | Recorded average-length parameter for the local encoder. |
| `token_max_length` | Maximum token length used by the sparse encoder. |
| `disable_stemmer` | Disable stemming when `true`. |

Sparse query and corpus profiles must match exactly for sparse or hybrid retrieval.

## Corpus Unit

```yaml
corpus_unit: chunk
```

| Value | Meaning |
| --- | --- |
| `chunk` | Embed OCR chunks. Retrieval results are grouped back to corpus pages. Best for text OCR retrieval. |
| `page` | Embed one point per page. Required for image vectors and useful for full-page text/sparse setups. |

Retrieval metrics are always evaluated at corpus-page level, even when the underlying search points are chunks.

## Retrieval

```yaml
retrieval:
  mode: hybrid
  page_top_k: 20
  prefetch_limit: 200
```

| Field | Meaning |
| --- | --- |
| `mode` | `dense`, `sparse`, or `hybrid`. Must match the vectors stored in the query and corpus embedding runs. |
| `page_top_k` | Number of corpus pages to save per query. |
| `prefetch_limit` | Optional per-branch candidate budget before page grouping. Defaults to `min(point_count, max(100, 10 * page_top_k))`. |

Hybrid retrieval uses Qdrant native RRF over dense and sparse point rankings, then groups results by `corpus_id`.

## Generation

Generation is optional unless you select Ragas metrics in a full experiment.

```yaml
generation:
  endpoint:
    model: your-generation-model
    base_url: http://localhost:8002/v1
    api_key_env: GENERATION_API_KEY
  context:
    representations: [ocr]
    representation_run_id: 00000000-0000-0000-0000-000000000000
    page_top_k: 5
    max_text_chars: 100000
    max_image_bytes: 20000000
  prompt: Answer the question using only the supplied document pages.
  temperature: 0
  max_tokens: 2048
```

| Field | Meaning |
| --- | --- |
| `endpoint` | Generation model endpoint. |
| `context.representations` | Page representations to include: `ocr`, `image`, `original_image`, or registered custom kinds. |
| `context.representation_run_id` | OCR/preprocess run to use for non-image contexts. In full experiments this can be filled automatically from the preprocessing stage. |
| `context.page_top_k` | Number of retrieved pages to pass to generation. Cannot exceed `retrieval.page_top_k`. |
| `context.max_text_chars` | Hard budget for text context. Exceeding it fails the sample. |
| `context.max_image_bytes` | Hard budget for image bytes. Exceeding it fails the sample. |
| `prompt` | System/task prompt for generation. |
| `temperature` | Generation temperature. |
| `max_tokens` | Response token limit. |

Reference qrel answers are never inserted into generation prompts.

## IR Evaluation

```yaml
ir:
  cutoffs: [1, 5, 10, 20]
  relevance_threshold: 1
```

| Field | Meaning |
| --- | --- |
| `cutoffs` | Ranking cutoffs for `P`, `R`, `RR`, and `nDCG`. |
| `relevance_threshold` | Minimum qrel score counted as relevant. |

Set `ir: null` to skip IR evaluation in a full experiment.

## Ragas Evaluation

```yaml
ragas:
  ragas_max_concurrency: 4
  metrics:
    - id: exact_match
    - id: faithfulness
  judge:
    model: your-judge-model
    base_url: http://localhost:8002/v1
    api_key_env: JUDGE_API_KEY
  embeddings:
    endpoint:
      model: evaluator-embedding-model
      base_url: http://localhost:8003/v1
      api_key_env: EVAL_EMBEDDING_API_KEY
    space_id: evaluator-text-space-v1
    dimensions: 1024
  context:
    representations: [ocr]
    page_top_k: 5
```

| Field | Meaning |
| --- | --- |
| `metrics` | Selected Ragas metric configs. Empty list means no Ragas calls. |
| `ragas_max_concurrency` | Maximum simultaneous Ragas metric calls. |
| `judge` | Judge LLM endpoint for LLM-based Ragas metrics. Required only by metrics that need it. |
| `embeddings` | Evaluator embedding profile for semantic Ragas metrics. This is independent from retrieval embeddings. |
| `context` | Context configuration for retrieval-only Ragas CLI runs. Full experiment Ragas usually uses persisted generation context. |

Use `uv run vdu metrics list` to see supported metrics and prerequisites.

## Reuse

```yaml
reuse:
  representations: 00000000-0000-0000-0000-000000000000
  chunks: 00000000-0000-0000-0000-000000000000
  query_embeddings: 00000000-0000-0000-0000-000000000000
  corpus_embeddings: 00000000-0000-0000-0000-000000000000
  retrieval: 00000000-0000-0000-0000-000000000000
  generation: 00000000-0000-0000-0000-000000000000
```

`reuse` pins completed stage runs. The pipeline validates that reused runs belong to the same dataset and that their configuration, dependencies, and query cohort match what the experiment needs.

Completed stages are also reused automatically when their fingerprint matches. Explicit `reuse` is useful when you want to share a stage across several experiment configs.

## Common Config Patterns

Cached OCR + BM25:

```yaml
preprocess:
  ocr_text_path: data/processed/.../DeepSeek-OCR-2_deepseek-ocr_i2t.csv
embeddings:
  sparse:
    model: Qdrant/bm25
retrieval:
  mode: sparse
```

Cached OCR + dense text:

```yaml
preprocess:
  ocr_text_path: data/processed/.../DeepSeek-OCR-2_deepseek-ocr_i2t.csv
embeddings:
  dense:
    endpoint:
      model: BAAI/bge-m3
      base_url: http://localhost:8001/v1
      api_key_env: EMBEDDING_API_KEY
    space_id: bge-m3-dense-v1
    dimensions: 1024
retrieval:
  mode: dense
```

Cached OCR + hybrid:

```yaml
embeddings:
  dense:
    endpoint:
      model: BAAI/bge-m3
      base_url: http://localhost:8001/v1
      api_key_env: EMBEDDING_API_KEY
    space_id: bge-m3-dense-v1
    dimensions: 1024
  sparse:
    model: Qdrant/bm25
retrieval:
  mode: hybrid
```

Image dense + sparse OCR:

```yaml
embeddings:
  dense:
    endpoint:
      model: your-multimodal-embedding-model
      base_url: http://localhost:8001/v1
      api_key_env: EMBEDDING_API_KEY
    space_id: shared-text-image-space-v1
    modality: image
    adapter: vllm
  sparse:
    model: Qdrant/bm25
corpus_unit: page
retrieval:
  mode: hybrid
```

For image/text shared-space experiments, use the same `space_id` only when the query text vectors and page image vectors are designed to live in the same dense vector space.
