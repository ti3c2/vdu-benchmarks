"""Versioned metric catalog; listing it never initializes models or downloads assets."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MetricSpec:
    id: str
    implementation: str
    fields: tuple[str, ...]
    judge: bool = False
    embeddings: bool = False
    modality: str = "text"
    api: str = "modern"
    unavailable_reason: str | None = None
    direction: str = "higher"


METRICS: dict[str, MetricSpec] = {}


def _register(id, implementation, fields, **kwargs):
    METRICS[id] = MetricSpec(id, implementation, tuple(fields.split()), **kwargs)


for metric_id, cls in [
    ("exact_match", "ExactMatch"),
    ("string_presence", "StringPresence"),
    ("non_llm_string_similarity", "NonLLMStringSimilarity"),
    ("bleu_score", "BleuScore"),
    ("chrf_score", "CHRFScore"),
    ("rouge_score", "RougeScore"),
]:
    _register(metric_id, cls, "response reference", modality="any")
_register(
    "quoted_spans_alignment", "QuotedSpansAlignment", "response retrieved_contexts"
)
_register(
    "semantic_similarity",
    "SemanticSimilarity",
    "response reference",
    embeddings=True,
    modality="any",
)
_register(
    "answer_relevancy",
    "AnswerRelevancy",
    "user_input response",
    judge=True,
    embeddings=True,
    modality="any",
)
_register(
    "answer_correctness",
    "AnswerCorrectness",
    "user_input response reference",
    judge=True,
    embeddings=True,
    modality="any",
)
_register(
    "answer_accuracy",
    "AnswerAccuracy",
    "user_input response reference",
    judge=True,
    modality="any",
)
_register(
    "factual_correctness",
    "FactualCorrectness",
    "response reference",
    judge=True,
    modality="any",
)
_register(
    "faithfulness", "Faithfulness", "user_input response retrieved_contexts", judge=True
)
_register(
    "response_groundedness",
    "ResponseGroundedness",
    "response retrieved_contexts",
    judge=True,
)
_register(
    "context_precision_with_reference",
    "ContextPrecisionWithReference",
    "user_input reference retrieved_contexts",
    judge=True,
)
_register(
    "context_precision_without_reference",
    "ContextPrecisionWithoutReference",
    "user_input response retrieved_contexts",
    judge=True,
)
_register(
    "context_recall",
    "ContextRecall",
    "user_input reference retrieved_contexts",
    judge=True,
)
_register(
    "context_entity_recall",
    "ContextEntityRecall",
    "reference retrieved_contexts",
    judge=True,
)
_register(
    "context_relevance", "ContextRelevance", "user_input retrieved_contexts", judge=True
)
_register(
    "noise_sensitivity",
    "NoiseSensitivity",
    "user_input response reference retrieved_contexts",
    judge=True,
    direction="lower",
)
_register(
    "multi_modal_faithfulness",
    "MultiModalFaithfulness",
    "response retrieved_contexts",
    judge=True,
    modality="multimodal",
)
_register(
    "multi_modal_relevance",
    "MultiModalRelevance",
    "user_input response retrieved_contexts",
    judge=True,
    modality="multimodal",
)
for metric_id, cls, fields in [
    ("domain_specific_rubrics", "DomainSpecificRubrics", "user_input response"),
    (
        "rubrics_score_with_reference",
        "RubricsScoreWithReference",
        "user_input response reference",
    ),
    (
        "rubrics_score_without_reference",
        "RubricsScoreWithoutReference",
        "user_input response",
    ),
]:
    _register(metric_id, cls, fields, judge=True)

for metric_id, cls, fields, judge, reason in [
    (
        "summary_score",
        "SummaryScore",
        "response reference_contexts",
        True,
        "Requires a summarization dataset with reference contexts",
    ),
    (
        "instance_specific_rubrics",
        "InstanceSpecificRubrics",
        "response rubrics",
        True,
        "Requires per-instance rubrics",
    ),
    (
        "tool_call_accuracy",
        "ToolCallAccuracy",
        "user_input reference_tool_calls",
        False,
        "Requires conversation and reference tool-call traces",
    ),
    (
        "tool_call_f1",
        "ToolCallF1",
        "user_input reference_tool_calls",
        False,
        "Requires conversation and reference tool-call traces",
    ),
    (
        "agent_goal_accuracy_with_reference",
        "AgentGoalAccuracyWithReference",
        "user_input reference",
        True,
        "Requires a multi-turn agent dataset",
    ),
    (
        "agent_goal_accuracy_without_reference",
        "AgentGoalAccuracyWithoutReference",
        "user_input",
        True,
        "Requires a multi-turn agent dataset",
    ),
    (
        "topic_adherence",
        "TopicAdherence",
        "user_input reference_topics",
        True,
        "Requires conversation and topic annotations",
    ),
    (
        "data_compy_score",
        "DataCompyScore",
        "response reference",
        False,
        "Requires tabular answer annotations and optional datacompy dependency",
    ),
    (
        "sql_semantic_equivalence",
        "SQLSemanticEquivalence",
        "response reference",
        True,
        "Requires SQL answer annotations",
    ),
]:
    _register(metric_id, cls, fields, judge=judge, unavailable_reason=reason)

for metric_id, cls, fields in [
    (
        "id_based_context_precision",
        "IDBasedContextPrecision",
        "retrieved_context_ids reference_context_ids",
    ),
    (
        "id_based_context_recall",
        "IDBasedContextRecall",
        "retrieved_context_ids reference_context_ids",
    ),
    (
        "non_llm_context_precision_with_reference",
        "NonLLMContextPrecisionWithReference",
        "retrieved_contexts reference_contexts",
    ),
    (
        "non_llm_context_recall",
        "NonLLMContextRecall",
        "retrieved_contexts reference_contexts",
    ),
]:
    _register(metric_id, cls, fields, api="legacy")
_register(
    "faithfulness_with_hhem",
    "FaithfulnesswithHHEM",
    "user_input response retrieved_contexts",
    api="legacy",
    unavailable_reason="Requires optional transformers/torch and a separately provisioned HHEM evaluator",
)
_register(
    "aspect_critic", "AspectCritic", "user_input response", judge=True, api="legacy"
)
_register(
    "simple_criteria_score",
    "SimpleCriteriaScore",
    "user_input response",
    judge=True,
    api="legacy",
)

ALIASES = {
    "context_precision": "context_precision_with_reference",
    "context_utilization": "context_precision_without_reference",
    "agent_goal_accuracy": "agent_goal_accuracy_with_reference",
}


def metric_catalog():
    return {
        "ragas_version": "0.4.3",
        "metrics": [asdict(spec) for spec in METRICS.values()],
        "aliases": ALIASES,
    }


def resolve_metric(metric_id: str) -> MetricSpec:
    canonical = ALIASES.get(metric_id, metric_id)
    if canonical not in METRICS:
        raise ValueError(f"Unknown Ragas metric {metric_id!r}; use 'vdu metrics list'")
    return METRICS[canonical]
