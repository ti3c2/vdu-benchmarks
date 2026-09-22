# Full RAG evaluation with OpenAI

Answer generation and Ragas evaluation are already implemented. The retrieval-only OCR configs omit `generation` and leave `ragas.metrics` empty. These full experiment examples enable both:

| Retrieval | Full RAG config | Retrieval-only counterpart |
| --- | --- | --- |
| BM25 | [ocr-cached-sparse-rag.yaml](../configs/ocr-cached-sparse-rag.yaml) | `configs/ocr-cached-sparse.yaml` |
| Dense | [ocr-cached-dense-rag.yaml](../configs/ocr-cached-dense-rag.yaml) | `configs/ocr-cached-dense.yaml` |
| Hybrid | [ocr-cached-hybrid-rag.yaml](../configs/ocr-cached-hybrid-rag.yaml) | `configs/ocr-cached-hybrid.yaml` |

Each selects original queries, retrieves 20 pages, generates an answer from the top five complete OCR pages, and scores retrieval and answers. Qrels and reference answers are used only for evaluation. Generation uses the default prompt to answer only from the supplied pages and acknowledge when they do not contain the answer.

Generation and judging use OpenAI's Chat Completions API with the pinned `gpt-4.1-mini-2025-04-14` snapshot. This model supports the existing client's request settings; see the [OpenAI model documentation](https://developers.openai.com/api/docs/models/gpt-4.1-mini). The model and endpoint can be configured independently for generation and judging. Keep both profiles and the generation context settings fixed across retrievers for a controlled comparison.

## Configure and run a full experiment

Follow the [retrieval setup guide](ocr-text-retriever-comparison.md) to start the stores, ingest the dataset, and configure retrieval. In the new RAG configs, replace the all-zero `dataset_id` with your dataset UUID and check the cached OCR path. Dense and hybrid configs use the current retrieval examples' Qwen embedding profile through OpenRouter; adjust it to match your retrieval setup.

Set `OPENAI_API_KEY` in the repository's `.env` or process environment. Generation and judging resolve that same environment variable; credentials are not stored in the YAML or run configuration. A fresh dense or hybrid run also needs `EMBEDDING_API_KEY` for the configured embedding endpoint. BM25 does not use an embedding endpoint.

```bash
uv run vdu experiment run --config configs/ocr-cached-sparse-rag.yaml
uv run vdu experiment run --config configs/ocr-cached-dense-rag.yaml
uv run vdu experiment run --config configs/ocr-cached-hybrid-rag.yaml
```

These commands make billed OpenAI calls for answer generation and judging. Each judge metric may make multiple requests per query. For a small first run, set `queries.limit: 10` consistently across all three configs before building retrieval. A limited query cohort cannot reuse a retrieval experiment's query embeddings from a different cohort.

The selected metrics cover:

| Metrics | What they measure |
| --- | --- |
| `P`, `R`, `RR`, `nDCG` at 1, 5, 10, 20 | Page retrieval quality against qrels |
| `exact_match` | Strict equality with the reference answer; wording differences can score zero |
| `factual_correctness` | Factual agreement between the answer and the reference |
| `faithfulness` | Whether the answer's claims are supported by the supplied OCR pages |
| `context_precision_with_reference` | Relevance and ordering of the five supplied pages for the reference answer |
| `context_recall` | Whether the supplied pages support the reference answer |

The selected Ragas metrics require a judge but no evaluator embedding model. Ragas uses the exact persisted generation contexts, so its context metrics evaluate the five pages supplied to generation. IR evaluates the saved retrieval ranking through depth 20. Missing or ambiguous reference answers produce skipped reference-dependent scores; inspect scored, skipped, and failed counts alongside averages.

## Reuse completed retrieval experiments

You can add generation and evaluation after retrieval has completed. An explicit retrieval run ID pins its query embeddings, corpus embeddings, chunks, and OCR dependencies. Those stages are attached to the new experiment without rerunning preprocessing, vectorization, or retrieval.

First, find the saved run IDs from the existing retrieval comparison:

```bash
uv run vdu experiment compare \
  --experiment-id BM25_EXPERIMENT_UUID \
  --experiment-id DENSE_EXPERIMENT_UUID \
  --experiment-id HYBRID_EXPERIMENT_UUID \
  --layout long \
  --format json
```

In each `experiments` entry, `runs.retrieval` is the retrieval run UUID and `runs.representations` is the OCR run UUID. `config` contains that experiment's saved settings. Experiment UUIDs and stage run UUIDs are different identifiers.

For each retriever, copy the matching full RAG config, set the same dataset UUID, and make its `queries`, `preprocess`, `chunking`, `embeddings`, `corpus_unit`, and `retrieval` settings match the saved experiment. Add:

```yaml
reuse:
  retrieval: REPLACE_WITH_COMPLETED_RETRIEVAL_RUN_UUID
```

The placeholder must be replaced with a real UUID before loading the config. Run the full experiment command as above. The pipeline infers `generation.context.representation_run_id` from the pinned OCR dependency. It generates answers and creates evaluations for the new experiment, while preserving the original retrieval experiment.

Reuse validates the dataset, completed status, dependencies, query cohort, and serialized upstream configuration. Even operational settings such as embedding batch size or endpoint URL must match. If the source YAML has changed since the retrieval run, use the saved experiment `config` as the source of truth. Copying the saved retrieval configuration and adding the examples' `generation`, `ragas`, and `reuse` blocks also works for the live OCR configs.

Automatic reuse also exists, but it requires a matching implementation fingerprint as well as matching inputs and settings. Explicit `reuse.retrieval` is the way to pin an existing retrieval result when adding generation after code changes. Reuse requires the persisted database records and page representations to remain available.

## Generate and evaluate as independent stages

To add answer metrics to the original experiment directly, use the stage configs instead of creating a full RAG experiment.

Edit [openai-generation.yaml](../configs/stages/openai-generation.yaml) and replace `context.representation_run_id` with the OCR run UUID (`runs.representations` above). Standalone generation requires this explicit ID for OCR context. It uses every query in the supplied retrieval run and does not rerun retrieval.

```bash
uv run vdu generate run \
  --retrieval-run-id RETRIEVAL_RUN_UUID \
  --config configs/stages/openai-generation.yaml
```

Use the returned `run_id` as `GENERATION_RUN_UUID`:

```bash
uv run vdu evaluate ragas \
  --experiment-id ORIGINAL_RETRIEVAL_EXPERIMENT_UUID \
  --retrieval-run-id RETRIEVAL_RUN_UUID \
  --generation-run-id GENERATION_RUN_UUID \
  --config configs/stages/openai-ragas.yaml
```

Repeat for each retrieval pipeline with its own experiment and retrieval IDs. The evaluator checks that generation belongs to the supplied retrieval run and uses its persisted context membership and order. Standalone evaluations are included by `experiment compare`; their generation IDs appear under `evaluations[].generation_run_id` even when they are not attached under `runs.generation`.

## Compare, inspect, and resume

Compare either the three new full RAG experiments or the three original experiments after adding standalone evaluations:

```bash
uv run vdu experiment compare \
  --experiment-id BM25_RAG_EXPERIMENT_UUID \
  --experiment-id DENSE_RAG_EXPERIMENT_UUID \
  --experiment-id HYBRID_RAG_EXPERIMENT_UUID \
  --format csv
```

The default table has one metric per row and one experiment per column. Use
`--layout long --format json` to inspect run IDs, evaluation configurations, and
compatibility reasons, or `--layout long --format csv` for the original detailed
table. Comparing an experiment that has Ragas scores with one that only has IR
scores reports different evaluator configurations.

```bash
uv run vdu run show --run-id GENERATION_OR_EVALUATION_RUN_UUID
uv run vdu run resume --run-id GENERATION_OR_EVALUATION_RUN_UUID
```

Resume retries missing or failed items and retains completed outputs when the original inputs, configuration, and implementation fingerprint still match. Missing whole-page OCR and exceeded context budgets fail samples rather than silently truncating pages. After resuming a standalone generation run, run its evaluation command. After a failed full experiment, rerun its config to create a new experiment that reuses the now-completed stages.

If an experiment is wedged or you want to abandon its partial progress, discard the experiment before rerunning the same config:

```bash
uv run vdu experiment discard --experiment-id FAILED_EXPERIMENT_UUID
uv run vdu experiment run --config configs/ocr-cached-dense-rag.yaml
```

Discard removes the experiment and incomplete runs that are exclusive to it. Completed shared stages are preserved by default; add `--include-completed` only when you also want to delete completed exclusive OCR, chunking, embedding, retrieval, generation, or evaluation runs.
