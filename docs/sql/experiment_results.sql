-- Bind :experiment_id to an experiment UUID. In a SQL editor without parameters,
-- replace CAST(:experiment_id AS uuid) with 'your-experiment-uuid'::uuid.
-- Flat rows ordered by query and retrieved rank. Multiple reference chunks
-- produce multiple rows per hit. All chunks come from this experiment's runs.
SELECT
    eq.experiment_id,
    q.id AS query_id,
    q.query AS query_text,
    h.rank,
    h.corpus_id AS retrieved_corpus_id,
    h.chunk_id AS retrieved_chunk_id,
    retrieved.text AS retrieved_chunk_text,
    qr.corpus_id AS reference_corpus_id,
    reference.id AS reference_chunk_id,
    reference.text AS reference_chunk_text,
    g.response AS answer_text,
    qr.answer AS reference_answer
FROM experiment_queries AS eq
JOIN queries AS q ON q.id = eq.query_id
LEFT JOIN experiment_runs AS rr
    ON rr.experiment_id = eq.experiment_id AND rr.role = 'retrieval'
LEFT JOIN retrievals AS r
    ON r.run_id = rr.run_id AND r.query_id = q.id
LEFT JOIN retrieval_hits AS h ON h.retrieval_id = r.id
LEFT JOIN chunks AS retrieved ON retrieved.id = h.chunk_id
LEFT JOIN qrels AS qr
    ON qr.query_id = COALESCE(q.rephrase_of_id, q.id)
    AND qr.dataset_id = q.dataset_id AND qr.score > 0
LEFT JOIN experiment_runs AS cr
    ON cr.experiment_id = eq.experiment_id AND cr.role = 'chunks'
LEFT JOIN chunks AS reference
    ON reference.corpus_id = qr.corpus_id AND reference.run_id = cr.run_id
LEFT JOIN experiment_runs AS gr
    ON gr.experiment_id = eq.experiment_id AND gr.role = 'generation'
LEFT JOIN generations AS g
    ON g.run_id = gr.run_id AND g.query_id = q.id AND g.retrieval_id = r.id
WHERE eq.experiment_id = CAST(:experiment_id AS uuid)
ORDER BY eq.experiment_id, q.id, h.rank, qr.corpus_id, reference.ordinal, reference.id;
