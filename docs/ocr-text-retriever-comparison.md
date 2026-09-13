# OCR Text Retriever Comparison

This guide compares retrieval quality on cached OCR text for original queries only. It runs three retrieval variants:

- BM25 only: `configs/ocr-cached-sparse.yaml`
- Dense only: `configs/ocr-cached-dense.yaml`
- Hybrid dense + BM25: `configs/ocr-cached-hybrid.yaml`

The cached configs load OCR text from:

```text
data/processed/20260907-212356_ibm-research-REAL-MM-RAG_FinReport_BEIR_deepseek-ai/i2t/DeepSeek-OCR-2_deepseek-ocr_i2t.csv
```

They all use:

```yaml
queries:
  rephrase_levels: [0]
```

That means the experiment uses only the original query text and excludes all rephrase levels.

## 1. Start Local Services

```bash
docker compose up -d
uv run alembic upgrade head
```

This starts PostgreSQL, Qdrant, and MinIO, then applies database migrations.

## 2. Ingest The Dataset

```bash
uv run vdu dataset ingest \
  --source ibm-research/REAL-MM-RAG_FinReport_BEIR \
  --revision main
```

The command prints a JSON object with a `dataset_id`. Copy that UUID into all three cached OCR configs:

```text
configs/ocr-cached-sparse.yaml
configs/ocr-cached-dense.yaml
configs/ocr-cached-hybrid.yaml
```

Replace:

```yaml
dataset_id: 00000000-0000-0000-0000-000000000000
```

with the returned dataset UUID.

## 3. Configure Embeddings

BM25 does not need an embedding endpoint.

For dense and hybrid retrieval, edit the dense endpoint in:

```text
configs/ocr-cached-dense.yaml
configs/ocr-cached-hybrid.yaml
```

Example:

```yaml
embeddings:
  dense:
    endpoint:
      model: BAAI/bge-m3
      base_url: http://localhost:8001/v1
      api_key_env: EMBEDDING_API_KEY
    space_id: bge-m3-dense-v1
    modality: text
    dimensions: 1024
```

Set `model`, `base_url`, `api_key_env`, `space_id`, and `dimensions` to match the embedding server you are running. `space_id` is a label that prevents accidentally mixing incompatible vector spaces across runs.

## 4. Run The Experiments

```bash
uv run vdu experiment run --config configs/ocr-cached-sparse.yaml
uv run vdu experiment run --config configs/ocr-cached-dense.yaml
uv run vdu experiment run --config configs/ocr-cached-hybrid.yaml
```

Each command returns an `experiment_id`. Keep all three IDs.

The default configs calculate IR metrics:

- `P@1`, `P@5`, `P@10`, `P@20`
- `R@1`, `R@5`, `R@10`, `R@20`
- `RR@1`, `RR@5`, `RR@10`, `RR@20`
- `nDCG@1`, `nDCG@5`, `nDCG@10`, `nDCG@20`

The mean of `RR` is MRR.

## 5. Compare The Results

```bash
uv run vdu experiment compare \
  --experiment-id BM25_EXPERIMENT_UUID \
  --experiment-id DENSE_EXPERIMENT_UUID \
  --experiment-id HYBRID_EXPERIMENT_UUID \
  --format csv
```

Replace the placeholders with the experiment IDs returned in the previous step.

Use `--format json` if you want structured output instead of CSV.

These configs evaluate retrieval only. To generate answers and evaluate the full RAG process from the same saved rankings, follow the [full RAG evaluation guide](rag-evaluation.md). It includes full experiment configs for all three retrievers and standalone generation/evaluation configs, with explicit retrieval reuse so OCR and embeddings do not need to run again.

## 6. Inspect A Run

If a command returns a run ID or an experiment is incomplete, inspect it with:

```bash
uv run vdu run show --run-id RUN_UUID
```

If a run is partial or failed and the cause is fixed, resume it with:

```bash
uv run vdu run resume --run-id RUN_UUID
```

## Running The Same Comparison With Live OCR

The cached configs skip OCR and read the existing CSV. When the OCR model is running, use the live OCR configs instead:

```text
configs/ocr-sparse.yaml
configs/ocr-dense.yaml
configs/ocr-hybrid.yaml
```

Those configs keep:

```yaml
preprocess:
  endpoint:
    model: deepseek-ai/DeepSeek-OCR-2
    base_url: http://localhost:8000/v1
    api_key_env: OCR_API_KEY
  prompt: Convert the document to markdown.
```

Set the same dataset UUID in the live configs, configure the OCR endpoint, and run the same `vdu experiment run` and `vdu experiment compare` commands with the live config filenames.
