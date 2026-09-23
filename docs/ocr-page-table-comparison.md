# Whole-page retrieval with and without tables

These four experiments index each OCR page as one document (`corpus_unit: page`),
without splitting it into chunks. They run dense or sparse retrieval separately,
followed by IR evaluation on the same 853 original queries.

| Config | Retriever | Tables in index |
| --- | --- | --- |
| `configs/ocr-cached-page-sparse-tables.yaml` | BM25 | Included |
| `configs/ocr-cached-page-sparse-no-tables.yaml` | BM25 | Removed |
| `configs/ocr-cached-page-dense-tables.yaml` | Qwen3 Embedding 8B | Included |
| `configs/ocr-cached-page-dense-no-tables.yaml` | Qwen3 Embedding 8B | Removed |

## Existing data and shared stages

Verified against the local stores on 2026-09-23:

- Dataset: `ada9f2df-81f9-466f-8fc5-b1912e6169f0` (2,687 pages).
- OCR run: `1161d65e-2832-46d8-b006-5316c4bf4bde` (all page texts match the CSV).
- Sparse query embeddings: `06df1d6e-b813-4879-a6da-6ae749a2ba5c` (853 points).
- Dense query embeddings: `e53c83b5-3562-40bb-9772-92972a9aae46` (853 points).

The CSV is:

```text
data/processed/20260907-212356_ibm-research-REAL-MM-RAG_FinReport_BEIR_deepseek-ai/i2t/DeepSeek-OCR-2_deepseek-ocr_i2t.csv
```

Each original-query config pins the OCR and matching query embedding run in `reuse`.
No ingestion, OCR requests, chunk building, or query embedding requests are needed.
The runs build new page vectors, retrieve pages, calculate metrics, and export
query-level results. PostgreSQL and Qdrant must be available.

The dense configs use the existing `qwen/qwen3-embedding-8b` profile on OpenRouter,
with 4,096 dimensions. Set `OPENAI_EMB_API_KEY` in `.env` or the environment.
The original-query configs only need corpus embedding calls. BM25 runs locally with the existing
English profile (`k=1.2`, `b=0.75`, `avg_len=256`); `avg_len` is deliberately held
fixed across variants and is not recomputed from the filtered corpus.

These run IDs belong to this database snapshot. On another database, update
`dataset_id` and remove or replace `reuse`; keeping `preprocess.ocr_text_path`
allows importing the same cached CSV without calling an OCR model.

## Run all four experiments

From the repository root:

```bash
uv run vdu experiment run --config configs/ocr-cached-page-sparse-tables.yaml
uv run vdu experiment run --config configs/ocr-cached-page-sparse-no-tables.yaml
uv run vdu experiment run --config configs/ocr-cached-page-dense-tables.yaml
uv run vdu experiment run --config configs/ocr-cached-page-dense-no-tables.yaml
```

Each command prints an `experiment_id` and saves a JSON results report under
`data/experiments`. The reported IR metrics are precision, recall, MRR (`RR`),
and nDCG at 1, 5, 10, and 20. These configs evaluate retrieval; they do not run
answer generation or Ragas.

Use the returned IDs to write a wide comparison CSV:

```bash
uv run vdu experiment compare \
  --experiment-id SPARSE_TABLES_UUID \
  --experiment-id SPARSE_NO_TABLES_UUID \
  --experiment-id DENSE_TABLES_UUID \
  --experiment-id DENSE_NO_TABLES_UUID \
  --format csv
```

The CSV is also saved under `data/experiments`. Find experiment IDs again with:

```bash
uv run vdu experiment list --dataset-id ada9f2df-81f9-466f-8fc5-b1912e6169f0
```

## Run the same experiments with all rephrase levels

The `-all-rephrases.yaml` variants select `queries.rephrase_levels: [0, 1, 2, 3]`:
the 853 original queries plus 853 queries at each of the three rephrase levels,
for 3,412 queries total. All other retrieval, table, encoder, and metric settings
match the corresponding original-query experiment.

```bash
uv run vdu experiment run --config configs/ocr-cached-page-sparse-tables-all-rephrases.yaml
uv run vdu experiment run --config configs/ocr-cached-page-sparse-no-tables-all-rephrases.yaml
uv run vdu experiment run --config configs/ocr-cached-page-dense-tables-all-rephrases.yaml
uv run vdu experiment run --config configs/ocr-cached-page-dense-no-tables-all-rephrases.yaml
```

To get new experiments ids:
```bash
uv run vdu experiment list \
  --dataset-id ada9f2df-81f9-466f-8fc5-b1912e6169f0 \
  | grep experiment_id \
  | awk -F '"' '{print $4}' \
  | tail -n4
```

These configs keep the pinned OCR run and remove `reuse.query_embeddings`:
the saved 853-query runs cannot serve the expanded cohort. No completed
3,412-query embedding run was present when these configs were added. The first
run for each encoder builds embeddings for all 3,412 queries; the second table
variant reuses that completed query run. Dense query embedding calls therefore
require `OPENAI_EMB_API_KEY`, in addition to any missing corpus embeddings.

Completed matching page indexes from the original-query experiments are reused
automatically when the implementation and settings match. Run the comparison
command above with the four new experiment IDs. Its default output aggregates
all selected query levels; add `--layout long` to see the stored aggregates by
rephrase level too.

## Meaning of table removal

`page_include_tables: true` encodes the original full-page OCR verbatim.
`page_include_tables: false` removes structured table content before both dense
and BM25 encoding. It covers Markdown pipe tables (including malformed OCR rows),
HTML tables, and LaTeX table/array environments. HTML tables used for page layout
are removed too. Captions and footnotes outside table blocks, surrounding prose,
and code blocks remain.
Unclosed HTML/LaTeX tables are removed through the end of the page. Text without
recognizable table structure is retained.

The filter changes 1,559 pages in this CSV. Four have no remaining text: original corpus IDs `719`, `1361`,
`1977`, and `2043`. They are omitted from the no-table index, leaving 2,683 points;
the table-inclusive index has 2,687 points. All queries and qrels remain in the
evaluation, including judgments for omitted pages.

The filter only changes indexed text. Stored OCR remains intact for inspection
or later generation. Table-inclusive and table-excluded corpus runs have distinct
identities, so they cannot silently reuse each other's vectors. Repeating an
experiment with the same implementation and settings reuses completed stages.

For the standalone `vdu vectorize pages` command, use an embedding-stage YAML
with `include_tables: false` alongside `dense` or `sparse`. Complete experiment
configs instead use the top-level `page_include_tables` field shown above.
