from functools import partial
from typing import Any

import datasets
from loguru import logger

from ..settings import get_settings

settings = get_settings()


def load_real_mm_rag(dataset: str) -> dict[str, Any]:
    """data looks like this:
    {'ibm-research/REAL-MM-RAG_FinReport_BEIR': {
        'corpus': Dataset({
            features: ['corpus-id', 'image', 'image_filename', 'doc-id'],
            num_rows: 2687
        }),
        'queries': Dataset({
            features: ['query-id', 'query', 'rephrase_level_1', 'rephrase_level_2', 'rephrase_level_3', 'language'],
            num_rows: 853
        }),
        'qrels': Dataset({
            features: ['query-id', 'corpus-id', 'answer', 'score'],
            num_rows: 853
        }),
        'docs': Dataset({
            features: ['doc-id'],
            num_rows: 19
        })
    }}
    """
    subsets = ["corpus", "queries", "qrels", "docs"]
    data = dict()
    dataset_dir = settings.path_data_raw / dataset.split("/")[1]
    data[dataset] = dict()
    for subset in subsets:
        data[dataset][subset] = datasets.load_dataset(
            dataset,
            split="test",
            name=subset,
            cache_dir=dataset_dir,
        )
    logger.info(f"Loaded {dataset} dataset")
    return data


dataset_loaders = {
    "ibm-research/REAL-MM-RAG_FinReport_BEIR": partial(
        load_real_mm_rag, dataset="ibm-research/REAL-MM-RAG_FinReport_BEIR"
    ),
    "ibm-research/REAL-MM-RAG_FinSlides_BEIR": partial(
        load_real_mm_rag, dataset="ibm-research/REAL-MM-RAG_FinSlides_BEIR"
    ),
    "ibm-research/REAL-MM-RAG_TechReport_BEIR": partial(
        load_real_mm_rag, dataset="ibm-research/REAL-MM-RAG_TechReport_BEIR"
    ),
    "ibm-research/REAL-MM-RAG_TechSlides_BEIR": partial(
        load_real_mm_rag, dataset="ibm-research/REAL-MM-RAG_TechSlides_BEIR"
    ),
}
