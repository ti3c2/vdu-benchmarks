# Visual document retrieval benchmarks

This project compares OCR/text, image, sparse, and hybrid retrieval pipelines. PostgreSQL owns dataset identities, run provenance, page representations, retrievals, answers, and metrics. MinIO stores page images. Qdrant stores reusable query and corpus vectors through LlamaIndex.

Retrieval is evaluated at **corpus-page level**. Chunks help find pages; generation receives complete page OCR, images, or configured combinations. Qrels and reference answers never enter retrieval or generation prompts.

## Setup

Requires Python 3.13+, `uv`, and Docker Compose. Model servers are external to this project.

```bash
uv sync --group dev
cp .env.example .env
docker compose up -d
uv run alembic upgrade head
uv run vdu --help
```

If you already have `.env`, merge the new settings instead of replacing it. Defaults use PostgreSQL on port 5432, Qdrant on 6333, and MinIO on 9000 (console: 9001). They are development credentials; configure environment values for your own deployment. MinIO uses the pinned official Quay image. The bucket is created idempotently on first ingestion.

Endpoint configuration contains `api_key_env`, the **name** of an environment variable. Credentials are resolved at runtime and excluded from stored run configurations.

## Ingest and run an experiment

```bash
uv run vdu dataset ingest --source ibm-research/REAL-MM-RAG_FinReport_BEIR --revision main
```

The command returns a dataset UUID. Put that UUID in `configs/ocr-cached-dense.yaml`, configure your embedding endpoint, and run:

```bash
uv run vdu experiment run --config configs/ocr-cached-dense.yaml
uv run vdu run show --run-id RUN_UUID
```

Configuration examples contain placeholder endpoint/model settings and the all-zero dataset UUID. Replace these before running. The `ocr-cached-*` examples load page text from `data/processed/20260907-212356_ibm-research-REAL-MM-RAG_FinReport_BEIR_deepseek-ai/i2t/DeepSeek-OCR-2_deepseek-ocr_i2t.csv`. The `ocr-*` examples keep the live OCR endpoint configuration for later model-backed OCR runs. See `docs/configuration.md` for a field-by-field config reference, including the meaning of `space_id`. `main` is resolved to a source revision during ingestion. Reingesting the same immutable source snapshot reuses its UUID.

FinReport has 19 documents, 2,687 pages, 853 original queries, and 853 qrels. Normalization produces 3,412 query rows when all three rephrases are present. Query variants inherit their base query's qrels and answer; source qrels are stored once.

Every document, page, query, and qrel has an internal UUID. Dataset-local identifiers are retained as `original_id`; they never serve as cross-dataset database keys. All non-original rephrases point directly to their base query. Corpus iteration order is not document page order.

## Independent stages

Each stage accepts input IDs and returns its run ID as JSON. Logs go to stderr, making IDs suitable for scripts. `--config` accepts a YAML file matching the stage's configuration object; see `src/config.py` and command help.

| Command | Inputs |
| --- | --- |
| `vdu preprocess run` | Dataset ID; OCR configuration |
| `vdu chunks build` | Representation-run ID; chunk configuration |
| `vdu vectorize queries` | Dataset ID; embedding configuration |
| `vdu vectorize chunks` | Chunk-run ID; embedding configuration |
| `vdu vectorize pages` | Dataset ID; embedding configuration; OCR representation-run ID when needed |
| `vdu retrieve run` | Query and corpus embedding-run IDs; retrieval configuration |
| `vdu generate run` | Retrieval-run ID; generation configuration |
| `vdu evaluate ir` | Experiment and retrieval-run IDs; IR configuration |
| `vdu evaluate ragas` | Experiment, retrieval, optional generation IDs; metric configuration |
| `vdu run resume` | Existing run ID |
| `vdu experiment compare` | Experiment IDs |
| `vdu suite run` | YAML list of complete experiment configurations |
| `vdu metrics list` | No model or database connection required |

An experiment pins its query cohort and all stage dependencies. `reuse` can explicitly select `representations`, `chunks`, `query_embeddings`, `corpus_embeddings`, `retrieval`, and `generation` run IDs. Completed stages are also reused automatically when their input IDs, query/page selection, configuration, and implementation fingerprint match. Changed configurations create new artifacts.

Incomplete stages return nonzero CLI status. Resume retries missing or failed items while retaining successful outputs. An interrupted external request whose result was never persisted can run again. PostgreSQL, MinIO, and Qdrant are reconciled through stable identities; they do not share a distributed transaction.

To compare OCR text retrieval on original queries only with the cached OCR file, use the step-by-step guide in `docs/ocr-text-retriever-comparison.md`.

Those configs use `queries.rephrase_levels: [0]`, so rephrased queries are excluded. BM25 needs no embedding model endpoint. Dense and hybrid need the configured text embedding endpoint. To run the same comparison with live OCR later, use `configs/ocr-sparse.yaml`, `configs/ocr-dense.yaml`, and `configs/ocr-hybrid.yaml`; those additionally require the OCR endpoint.

## Retrieval configurations

- **Dense OCR:** paragraph/table chunks embedded through an OpenAI-compatible endpoint.
- **Sparse OCR:** `Qdrant/bm25` encoding runs locally through LlamaIndex sparse callbacks. No sparse model endpoint is required.
- **Hybrid OCR:** dense and sparse named vectors share each chunk's UUID.
- **Dense images:** the `vllm` adapter sends images to the documented `/embeddings` messages extension. The configured endpoint must already implement that format and return one dense vector per page.
- **Image + sparse OCR:** dense image and full-page BM25 vectors share each page UUID. Select `corpus_unit: page`.

Query and corpus collections are separate. Retrieval loads the stored query vectors; it never embeds queries again. A declared `space_id` prevents mixing unrelated dense vector spaces. The query and document encoders can use different instructions but must produce compatible vectors.

BM25 uses separate document `.embed()` and query `.query_embed()` paths. Defaults are English, `k=1.2`, `b=0.75`, `avg_len=256`, token maximum length 40, with stemming. These are recorded settings, not a computed corpus average. Corpus collections apply Qdrant IDF and remain immutable after completion. Query collections retain unweighted sparse query vectors. The first BM25 use may download small tokenizer/stopword assets.

Hybrid retrieval uses **Qdrant's native RRF**, then groups by corpus UUID. Fusion operates on point IDs before grouping; different matching chunks of the same page do not reinforce each other during RRF. Arbitrary page/chunk or cross-collection fusion is rejected. Native default RRF parameters and Qdrant version are recorded.

Default retrieval depth is 20 pages. The per-branch candidate budget is `min(point_count, max(100, 10 * page_top_k))`, unless explicitly configured. A candidate budget dominated by one page can yield fewer unique pages; the shortfall is recorded and the budget stays fixed within the run.

## Generation and evaluation

Generation selects five complete pages by default. `context.representations` accepts `ocr`, `image`/`original_image`, or registered custom representations. Non-image contexts require a pinned `representation_run_id`. Missing representations and exceeded context budgets fail the sample; the pipeline never silently replaces full pages with matching chunks or truncates pages.

Register future whole-page rendering formats with `register_context_renderer`. Model/server changes, model training, and token/patch multivector retrieval are outside this repository.

IR defaults are `P`, `R`, `RR`, and `nDCG` at 1, 5, 10, and 20. The mean of `RR` is MRR. Precision divides by the requested cutoff. Relevance threshold is 1, unjudged results are nonrelevant, and nDCG uses the explicit `log2` convention with graded labels. Empty successful retrievals score zero; failed retrievals never become successful empty results. Queries with no usable relevance judgments are unscored.

```bash
uv run vdu metrics list
```

The versioned Ragas catalog lists classical comparison, semantic similarity, answer quality, context quality, grounding, multimodal, rubric, agent/tool, summarization, and SQL/tabular metric families. An empty `ragas.metrics` list makes no evaluator calls. Explicitly selected metrics validate their judge, embedding, field, and dependency requirements. Specialized metrics without matching dataset annotations remain listed with their prerequisite reason. Legacy-only supported metrics use isolated upstream adapters.

Examples:

```yaml
ragas:
  ragas_max_concurrency: 4
  metrics:
    - id: exact_match
    - id: rouge_score
      parameters:
        rouge_type: rougeL
        mode: fmeasure
    - id: faithfulness
  judge:
    model: your-judge-model
    base_url: http://localhost:8002/v1
    api_key_env: JUDGE_API_KEY
```

`answer_relevancy`, `semantic_similarity`, and semantically weighted `answer_correctness` additionally require `ragas.embeddings`. Keep evaluator profiles fixed while changing retrieval profiles. Vision metrics require a vision-capable judge; text-context metrics do not interpret image filenames or image data URLs as OCR text.

Generation-associated Ragas uses the exact persisted context membership/order. A sole distinct nonempty qrel answer becomes the reference. Multiple different answers are preserved and marked ambiguous rather than silently selecting one. Context-only Ragas can run without generation when an explicit context configuration is supplied.

Every evaluation stores per-query status/value/reason and dataset, language, and rephrase-level aggregates with expected, scored, skipped, and failed counts. Comparison includes standalone evaluations and reports cohort and evaluator compatibility alongside metrics. JSON and CSV identify each evaluation run and its configuration, preserving multiple metric configurations within an experiment. Categorical outputs are retained without inventing numeric averages.

## Development and tests

```bash
uv run pytest -q
VDU_INTEGRATION=1 uv run pytest -q
uv run alembic check
./scripts/format.sh
```

Integration tests require the Compose services and migrated database. They use isolated test schemas/buckets/collections, fake model endpoints, and saved OCR samples from:

`data/processed/20260907-212356_ibm-research-REAL-MM-RAG_FinReport_BEIR_deepseek-ai/i2t/DeepSeek-OCR-2_deepseek-ocr_i2t.csv`

No test makes a fresh OCR or paid model call. A small derived fixture is included for environments without the original local CSV. Native fusion/grouping tests run against actual Qdrant because its client's in-memory grouping behavior differs around candidate limits.

The PostgreSQL schema and CRUD operations live in `src/stor_rel`; object storage in `src/stor_obj`; ingestion/OCR/chunks in `src/etls`; vectorization/retrieval/generation/metrics in `src/evaluate`; and pipeline composition in `src/orchestrate`. The former CSV vectorization entry point has been removed; notebooks should call the ID-based stages.
