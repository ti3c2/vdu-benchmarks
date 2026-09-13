"""Explicit Ragas metric execution within PostgreSQL experiment runs."""

import asyncio
import hashlib
import importlib
import inspect
import json
import logging
import math
from time import perf_counter
from uuid import UUID

from openai import AsyncOpenAI
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_fixed

from src.config import RagasConfig
from src.evaluate.generation import CONTEXT_RENDERERS, render_page
from src.evaluate.ir import persist_aggregates
from src.evaluate.metrics import resolve_metric
from src.evaluate.model_endpoints import verify_model_endpoint
from src.stor_obj import ObjectStore
from src.stor_rel.crud import (
    find_records,
    finish_run,
    get_effective_qrels,
    get_record,
    start_run,
    upsert_record,
    validate_run,
)
from src.stor_rel.schema import (
    EvaluationRun,
    Experiment,
    ExperimentQuery,
    Generation,
    GenerationContext,
    MetricResult,
    PageRepresentation,
    Query,
    Retrieval,
)
from src.utils.progress import progress_bar

logger = logging.getLogger(__name__)


def metric_key(config):
    spec = resolve_metric(config.id)
    if not config.parameters:
        return spec.id
    digest = hashlib.sha256(
        json.dumps(config.parameters, sort_keys=True).encode()
    ).hexdigest()[:12]
    return f"{spec.id}:{digest}"


def validate_metrics(config: RagasConfig):
    keys = set()
    for selected in config.metrics:
        spec = resolve_metric(selected.id)
        if spec.unavailable_reason:
            raise ValueError(f"{spec.id}: {spec.unavailable_reason}")
        key = metric_key(selected)
        if key in keys:
            raise ValueError(f"Duplicate metric configuration: {key}")
        keys.add(key)
        if spec.judge and config.judge is None:
            raise ValueError(f"{spec.id} requires an explicit evaluator judge profile")
        needs_embeddings = spec.embeddings
        if spec.id == "answer_correctness":
            weights = selected.parameters.get("weights", [0.75, 0.25])
            if not isinstance(weights, list) or len(weights) != 2:
                raise ValueError(
                    "answer_correctness requires factual and semantic weights"
                )
            if weights[1] == 0:
                needs_embeddings = False
        if needs_embeddings and config.embeddings is None:
            raise ValueError(
                f"{spec.id} requires an independent evaluator embedding profile"
            )
        if spec.id in {
            "aspect_critic",
            "simple_criteria_score",
        } and not selected.parameters.get("definition"):
            raise ValueError(f"{spec.id} requires parameters.definition")
    if config.embeddings and config.embeddings.modality != "text":
        raise ValueError("Ragas semantic metrics require text evaluator embeddings")


def _metric_needs_embeddings(selected) -> bool:
    spec = resolve_metric(selected.id)
    if spec.id == "answer_correctness":
        weights = selected.parameters.get("weights", [0.75, 0.25])
        if isinstance(weights, list) and len(weights) == 2 and weights[1] == 0:
            return False
    return spec.embeddings


async def verify_ragas_endpoints(config: RagasConfig):
    if config.judge and any(
        resolve_metric(metric.id).judge for metric in config.metrics
    ):
        await verify_model_endpoint(config.judge, purpose="Ragas judge")
    if config.embeddings and any(
        _metric_needs_embeddings(metric) for metric in config.metrics
    ):
        await verify_model_endpoint(
            config.embeddings.endpoint, purpose="Ragas embedding"
        )


def create_metrics(config: RagasConfig):
    """Only selected metrics initialize their dependencies; clients are returned for cleanup."""
    validate_metrics(config)
    metrics = []
    client = None
    judge = None
    legacy_judge = None
    embedder = None
    if any(
        resolve_metric(m.id).judge and resolve_metric(m.id).api == "modern"
        for m in config.metrics
    ):
        from ragas.llms import llm_factory

        profile = config.judge
        client = AsyncOpenAI(
            base_url=profile.base_url,
            api_key=profile.resolve_api_key(),
            timeout=profile.timeout_seconds,
            max_retries=0,
        )
        judge = llm_factory(
            profile.model,
            provider="openai",
            client=client,
            adapter="instructor",
            temperature=0,
            extra_body=profile.extra_body or None,
        )
    if any(
        resolve_metric(m.id).judge and resolve_metric(m.id).api == "legacy"
        for m in config.metrics
    ):
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper

        profile = config.judge
        legacy_judge = LangchainLLMWrapper(
            ChatOpenAI(
                model=profile.model,
                base_url=profile.base_url,
                api_key=profile.resolve_api_key(),
                temperature=0,
                timeout=profile.timeout_seconds,
                max_retries=0,
                extra_body=profile.extra_body,
            )
        )
    if config.embeddings and any(
        resolve_metric(m.id).embeddings for m in config.metrics
    ):
        from ragas.embeddings.base import BaseRagasEmbedding

        from src.evaluate.embeddings import (
            VLLMEmbedding,
            create_dense_model,
            encode_dense,
            validate_dense,
        )

        model = create_dense_model(config.embeddings)
        profile = config.embeddings.endpoint

        class LlamaIndexEvaluatorEmbedding(BaseRagasEmbedding):
            def embed_text(self, text, **kwargs):
                return self.embed_texts([text], **kwargs)[0]

            async def aembed_text(self, text, **kwargs):
                return (await encode_dense(model, config.embeddings, [text], "corpus"))[
                    0
                ]

            @retry(
                stop=stop_after_attempt(
                    1 if isinstance(model, VLLMEmbedding) else profile.max_retries
                ),
                wait=wait_fixed(profile.retry_wait_seconds),
                before_sleep=before_sleep_log(logger, logging.WARNING),
                reraise=True,
            )
            def embed_texts(self, texts, **kwargs):
                # Some Ragas async metrics call the synchronous embedding interface.
                # vLLM owns its instructions and retry policy inside its adapter.
                values = (
                    texts
                    if isinstance(model, VLLMEmbedding)
                    else [
                        config.embeddings.document_instruction + text for text in texts
                    ]
                )
                vectors = model.get_text_embedding_batch(values)
                if len(vectors) != len(texts):
                    raise ValueError("Evaluator returned the wrong number of vectors")
                dimensions = config.embeddings.dimensions or (
                    len(vectors[0]) if vectors else None
                )
                return [validate_dense(vector, dimensions) for vector in vectors]

            async def aembed_texts(self, texts, **kwargs):
                return await encode_dense(model, config.embeddings, texts, "corpus")

        embedder = LlamaIndexEvaluatorEmbedding()
    for selected in config.metrics:
        spec = resolve_metric(selected.id)
        module = importlib.import_module(
            "ragas.metrics.collections" if spec.api == "modern" else "ragas.metrics"
        )
        cls = getattr(module, spec.implementation)
        kwargs = dict(selected.parameters)
        if spec.id == "non_llm_string_similarity" and "distance_measure" in kwargs:
            from ragas.metrics.collections._string import DistanceMeasure

            kwargs["distance_measure"] = DistanceMeasure(kwargs["distance_measure"])
        if spec.judge:
            kwargs["llm"] = judge if spec.api == "modern" else legacy_judge
        if spec.embeddings and embedder:
            kwargs["embeddings"] = embedder
        if spec.id in {"aspect_critic", "simple_criteria_score"}:
            kwargs.setdefault("name", spec.id)
        metrics.append((selected, spec, cls(**kwargs)))
    return metrics, client


async def evaluate_ragas(
    experiment_id: UUID,
    retrieval_run_id: UUID,
    generation_run_id: UUID | None,
    config: RagasConfig,
    resume_run_id: UUID | None = None,
    *,
    object_store=None,
    metric_factory=None,
) -> UUID:
    validate_metrics(config)
    if not config.metrics:
        raise ValueError("Select at least one Ragas metric")
    experiment = await get_record(Experiment, experiment_id)
    await validate_run(
        retrieval_run_id, kind="retrieval", dataset_id=experiment.dataset_id
    )
    context_config = config.context
    inputs = {"retrieval": retrieval_run_id}
    generation_map = {}
    if generation_run_id:
        from src.config import GenerationConfig
        from src.stor_rel.schema import GenerationRun

        generation_run = await validate_run(
            generation_run_id, kind="generation", dataset_id=experiment.dataset_id
        )
        generation_info = await get_record(GenerationRun, generation_run_id)
        if generation_info.retrieval_run_id != retrieval_run_id:
            raise ValueError("Generation belongs to a different retrieval run")
        inputs["generation"] = generation_run_id
        context_config = GenerationConfig.model_validate(generation_run.config).context
        generation_map = {
            g.query_id: g
            for g in await find_records(Generation, run_id=generation_run_id)
        }
    if not generation_run_id and context_config is None:
        raise ValueError(
            "Retrieval-only Ragas evaluation requires an explicit context configuration"
        )
    if context_config and context_config.representation_run_id:
        await validate_run(
            context_config.representation_run_id, dataset_id=experiment.dataset_id
        )
        inputs["representations"] = context_config.representation_run_id
    selection = await find_records(ExperimentQuery, experiment_id=experiment_id)
    queries = [await get_record(Query, row.query_id) for row in selection]
    if not queries:
        raise ValueError("Experiment has no selected queries")
    retrievals = {
        r.query_id: r for r in await find_records(Retrieval, run_id=retrieval_run_id)
    }
    if set(retrievals) != {q.id for q in queries}:
        raise ValueError("Retrieval query cohort does not match experiment")
    if metric_factory is None:
        await verify_ragas_endpoints(config)
    metrics, client = (metric_factory or create_metrics)(config)
    run = await start_run(
        experiment.dataset_id,
        "evaluate_ragas",
        {**config.model_dump(mode="json"), "experiment_id": str(experiment_id)},
        inputs=inputs,
        selection=sorted(str(q.id) for q in queries),
        resume_run_id=resume_run_id,
    )
    if run.status == "completed":
        if client:
            await client.close()
        return run.id
    await upsert_record(
        EvaluationRun,
        {"id": run.id},
        {
            "dataset_id": run.dataset_id,
            "experiment_id": experiment_id,
            "retrieval_run_id": retrieval_run_id,
            "generation_run_id": generation_run_id,
            "framework": "ragas",
        },
    )
    store = object_store or ObjectStore()
    qrels = await get_effective_qrels(experiment.dataset_id, [q.id for q in queries])
    prior = {
        (r.query_id, r.metric_id)
        for r in await find_records(MetricResult, evaluation_run_id=run.id)
        if r.status == "completed"
    }
    metric_ids = [metric_key(selected) for selected, _, _ in metrics]
    prior_count = sum(
        (query.id, metric_id) in prior for query in queries for metric_id in metric_ids
    )
    metric_semaphore = asyncio.Semaphore(config.ragas_max_concurrency)

    async def score_query(query):
        retrieval = retrievals[query.id]
        generation = generation_map.get(query.id)
        pending = [
            (selected, spec, metric)
            for selected, spec, metric in metrics
            if (query.id, metric_key(selected)) not in prior
        ]
        if not pending:
            return
        required = set().union(*(set(spec.fields) for _, spec, _ in pending))
        if any(
            spec.id == "domain_specific_rubrics"
            and selected.parameters.get("with_reference", False)
            for selected, spec, _ in pending
        ):
            required.add("reference")
        needs_text = any(
            "retrieved_contexts" in spec.fields and spec.modality != "multimodal"
            for _, spec, _ in pending
        )
        needs_multimodal = any(
            "retrieved_contexts" in spec.fields and spec.modality == "multimodal"
            for _, spec, _ in pending
        )
        errors = {}
        text_contexts, multimodal_contexts, context_ids = [], [], []
        image_context_present = False
        context_ids_loaded = False
        reference_contexts = None
        selected_qrels = [
            r for r in qrels if UUID(str(r["query_id"])) == query.id and r["score"] > 0
        ]
        answers = list(
            dict.fromkeys(
                r["answer"]
                for r in selected_qrels
                if r.get("answer") and r["answer"].strip()
            )
        )
        if generation_run_id and (
            generation is None or generation.status != "completed"
        ):
            errors["response"] = "Generation failed or is missing"
        if required & {"retrieved_contexts", "retrieved_context_ids"}:
            try:
                if retrieval.status != "completed":
                    raise ValueError("Retrieval failed")
                from src.stor_rel.schema import RetrievalHit

                representations = []
                if generation:
                    contexts = sorted(
                        await find_records(
                            GenerationContext, generation_id=generation.id
                        ),
                        key=lambda c: c.position,
                    )
                    if "retrieved_context_ids" in required:
                        for context in contexts:
                            hit = await get_record(RetrievalHit, context.hit_id)
                            context_ids.append(str(hit.corpus_id))
                        context_ids_loaded = True
                    if "retrieved_contexts" in required:
                        for context in contexts:
                            representations.append(
                                await get_record(
                                    PageRepresentation, context.representation_id
                                )
                            )
                else:
                    hits = sorted(
                        await find_records(RetrievalHit, retrieval_id=retrieval.id),
                        key=lambda h: h.rank,
                    )[: context_config.page_top_k]
                    context_ids = [str(hit.corpus_id) for hit in hits]
                    context_ids_loaded = True
                    if "retrieved_contexts" in required:
                        for hit in hits:
                            for kind in context_config.representations:
                                kind = "original_image" if kind == "image" else kind
                                filters = {
                                    "dataset_id": experiment.dataset_id,
                                    "corpus_id": hit.corpus_id,
                                    "kind": kind,
                                }
                                if kind != "original_image":
                                    filters["run_id"] = (
                                        context_config.representation_run_id
                                    )
                                found = await find_records(
                                    PageRepresentation, **filters
                                )
                                if len(found) != 1:
                                    raise ValueError(
                                        "Requested complete-page context is missing or ambiguous"
                                    )
                                representations.extend(found)
                for representation in representations:
                    image_only = (
                        representation.text is None
                        and representation.kind not in CONTEXT_RENDERERS
                    )
                    if image_only:
                        image_context_present = True
                        if not needs_multimodal:
                            continue
                    try:
                        parts = await render_page(representation, store)
                        for part in parts:
                            if part["type"] == "text":
                                text_contexts.append(part["text"])
                                multimodal_contexts.append(part["text"])
                            elif part["type"] == "image_url":
                                image_context_present = True
                                multimodal_contexts.append(part["image_url"]["url"])
                    except Exception as exc:
                        if needs_multimodal:
                            errors["multimodal_contexts"] = str(exc)
                        if needs_text and not image_only:
                            errors["text_contexts"] = str(exc)
            except Exception as exc:
                errors["retrieved_contexts"] = str(exc)
                if "retrieved_context_ids" in required and not context_ids_loaded:
                    errors["retrieved_context_ids"] = str(exc)
        if "reference_contexts" in required and context_config.representation_run_id:
            try:
                gold_texts = []
                for page_id in dict.fromkeys(
                    UUID(str(r["corpus_id"])) for r in selected_qrels
                ):
                    reps = await find_records(
                        PageRepresentation,
                        run_id=context_config.representation_run_id,
                        corpus_id=page_id,
                        kind="ocr",
                    )
                    if len(reps) != 1 or reps[0].text is None:
                        break
                    gold_texts.append(reps[0].text)
                else:
                    reference_contexts = gold_texts
            except Exception as exc:
                errors["reference_contexts"] = str(exc)
        sample = {
            "user_input": query.query,
            "response": generation.response if generation else None,
            "reference": answers[0] if len(answers) == 1 else None,
            "reference_contexts": reference_contexts,
            "retrieved_context_ids": list(dict.fromkeys(context_ids)),
            "reference_context_ids": list(
                dict.fromkeys(str(r["corpus_id"]) for r in selected_qrels)
            ),
        }

        async def score_metric(selected, spec, metric):
            key = metric_key(selected)
            accepted_fields = set(spec.fields)
            if spec.id == "domain_specific_rubrics" and selected.parameters.get(
                "with_reference", False
            ):
                accepted_fields.add("reference")
            fields = {
                key: value for key, value in sample.items() if key in accepted_fields
            }
            if "retrieved_contexts" in accepted_fields:
                fields["retrieved_contexts"] = (
                    multimodal_contexts
                    if spec.modality == "multimodal"
                    else text_contexts
                    if text_contexts or not image_context_present
                    else None
                )

            started = perf_counter()
            values = {
                "dataset_id": run.dataset_id,
                "status": "completed",
                "value": None,
                "raw_value": None,
                "reason": None,
                "error": None,
                "traces": None,
            }
            try:
                relevant_errors = [
                    errors[field] for field in accepted_fields if field in errors
                ]
                if "retrieved_contexts" in accepted_fields:
                    context_error = errors.get(
                        "multimodal_contexts"
                        if spec.modality == "multimodal"
                        else "text_contexts"
                    )
                    if context_error:
                        relevant_errors.append(context_error)
                if relevant_errors:
                    raise RuntimeError("; ".join(dict.fromkeys(relevant_errors)))
                missing = [
                    field for field in accepted_fields if fields.get(field) is None
                ]
                if missing:
                    values.update(
                        status="skipped",
                        reason=f"Missing or ambiguous inputs: {', '.join(missing)}",
                    )
                else:
                    profile = config.judge

                    @retry(
                        stop=stop_after_attempt(
                            profile.max_retries if profile and spec.judge else 1
                        ),
                        wait=wait_fixed(profile.retry_wait_seconds if profile else 0),
                        before_sleep=before_sleep_log(logger, logging.WARNING),
                        reraise=True,
                    )
                    async def call_metric():
                        async with metric_semaphore:
                            if spec.api == "legacy":
                                from ragas import SingleTurnSample

                                return await metric.single_turn_ascore(
                                    SingleTurnSample(
                                        **{
                                            k: v
                                            for k, v in fields.items()
                                            if v is not None
                                        }
                                    )
                                )
                            signature = inspect.signature(metric.ascore)
                            return await metric.ascore(
                                **{
                                    k: v
                                    for k, v in fields.items()
                                    if k in signature.parameters and v is not None
                                }
                            )

                    result = await call_metric()
                    raw_value = getattr(result, "value", result)
                    numeric = (
                        float(raw_value)
                        if isinstance(raw_value, (float, int))
                        else None
                    )
                    if numeric is not None and not math.isfinite(numeric):
                        raise ValueError("Metric returned a nonfinite value")
                    values.update(
                        value=numeric,
                        raw_value=json.loads(json.dumps(raw_value, default=str)),
                        reason=getattr(result, "reason", None),
                        traces=json.loads(
                            json.dumps(getattr(result, "traces", None), default=str)
                        ),
                    )
            except Exception as exc:
                values.update(status="failed", error=str(exc))
            values["latency_ms"] = (perf_counter() - started) * 1000
            await upsert_record(
                MetricResult,
                {
                    "evaluation_run_id": run.id,
                    "query_id": query.id,
                    "metric_id": key,
                },
                values,
            )
            progress.update()

        await asyncio.gather(
            *(
                score_metric(selected, spec, metric)
                for selected, spec, metric in pending
            )
        )

    try:
        with progress_bar(
            total=len(queries) * len(metrics), desc="Evaluating Ragas", unit="metric"
        ) as progress:
            progress.update(prior_count)
            await asyncio.gather(*(score_query(query) for query in queries))
        await persist_aggregates(run.id, queries)
        results = await find_records(MetricResult, evaluation_run_id=run.id)
        failed = sum(r.status == "failed" for r in results)
        await finish_run(
            run.id,
            status="partial" if failed else "completed",
            expected_count=len(queries) * len(metrics),
            completed_count=sum(r.status == "completed" for r in results),
            failed_count=failed,
        )
    except Exception as exc:
        await finish_run(run.id, status="failed", error=str(exc))
        raise
    finally:
        if client:
            await client.close()
    return run.id
