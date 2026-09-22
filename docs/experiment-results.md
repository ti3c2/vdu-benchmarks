# Inspect experiment results

Each successful `vdu experiment run` (including runs within `vdu suite run`)
automatically writes a query-level JSON report under `data/experiments`.
The filename is `<experiment-created-at>_<experiment-uuid>_results.json`.
The timestamp uses UTC and the format `YYYYMMDD-HHMMSS`. Files use UTF-8 and
two-space indentation. Exporting an experiment again atomically refreshes its
file with the latest saved results, including standalone evaluations.

Export existing experiments without running any models:

```bash
# One experiment. Repeat --experiment-id to select several.
uv run vdu experiment export \
  --experiment-id df76990d-0e9c-4169-a102-9aef5e62fbd2

# All experiments across all datasets, including partial or failed experiments.
uv run vdu experiment export

# Optional dataset filter and destination.
uv run vdu experiment export --dataset-id DATASET_UUID --output-dir data/results
```

Stdout contains a JSON list of export paths and query counts. The Python API is
`await export_experiments(...)` in `src.orchestrate.export`; use
`await build_experiment_report(experiment_id)` to obtain the report as a dict
without writing a file.

## Report fields

The top level includes experiment IDs, timestamps, status, config, pinned run
IDs, evaluation configurations/provenance, and a `queries` array. Each query has:

- `query_id`, `query_original_id`, `query_text`, language and rephrase metadata.
- `reference_doc_ids`, `reference_corpus_ids`, and `references` with original
  document/page IDs, relevance scores, and source answers. Only positive qrels
  are references, and rephrases inherit their base query's qrels.
- `reference_answer` when there is exactly one distinct nonempty reference.
  `reference_answers` preserves every distinct answer; ambiguity leaves the
  singular field `null`.
- `retrievals`, each with its run ID, status/error, latency, document/page ID
  lists, and rank-ordered `hits`. Each hit contains `rank`, `score`, `doc_id`,
  `doc_original_id`, `corpus_id`, `corpus_original_id`, `chunk_id`, and
  `chunk_text`. Page/image retrieval has `null` chunk fields. A chunk is the
  stored representative match for that page, not every candidate searched.
- `answer_text` and `answer_generation_run_id` for the pinned generation run,
  or the sole saved generation when there is no pinned run. Multiple unpinned
  generations leave these fields `null`.
- `generations` preserving all saved answers belonging to the pinned run or
  to this experiment's evaluations, with run IDs and status/error information.
- `metrics`, containing saved **per-query** values, metric IDs, framework,
  evaluation-run ID, status, reason, error, and raw/categorical values. Repeated
  metric IDs from different evaluation runs remain separate. The top-level
  `evaluations` list identifies their retrieval/generation inputs and configs.

Only the frozen experiment query cohort is exported, even when a reused run
contains additional queries. Retrieval runs referenced by standalone evaluations
are also included and kept separate. Missing retrievals have status `missing`;
successful empty retrievals have status `completed` and an empty `hits` list.
Absent answers are `null`, and unavailable metrics are never changed to zero.
The report exports saved data; it does not calculate missing metrics.

The benchmark evaluates **pages** (`corpus_id`), not whole documents (`doc_id`).
Matching a document ID alone does not establish that the relevant page was found.
Generation can use full-page context even when retrieval matched a short chunk.

## SQL inspection

Use [experiment_results.sql](sql/experiment_results.sql) with `:experiment_id`
bound to an experiment UUID. In editors without named parameters, replace
`CAST(:experiment_id AS uuid)` with a quoted UUID, such as
`'df76990d-0e9c-4169-a102-9aef5e62fbd2'::uuid`.

The query uses plain joins to show query text, retrieved and reference chunk
IDs/text, and generated and reference answers in scalar columns. Rows for each
query are consecutive and ordered by retrieval rank. No arrays or aggregation
are used.

References are page-level judgments, so reference chunks are all chunks from
the relevant pages in the experiment's pinned chunking run. Multiple reference
chunks produce multiple rows for each retrieved hit, ordered by reference page
and chunk ordinal within each rank. Queries without hits, chunks, or generation
remain visible with `NULL` fields. Metrics are omitted. Remove the final `WHERE`
filter to inspect every experiment.
