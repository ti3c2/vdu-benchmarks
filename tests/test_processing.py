"""Use already generated OCR; no unit test makes an OCR request."""

import csv
import json
from pathlib import Path

import pytest

from src.etls.text_transforms import split_markdown

OCR_FIXTURE = (
    Path(__file__).parents[1]
    / "data/processed/20260907-212356_ibm-research-REAL-MM-RAG_FinReport_BEIR_deepseek-ai"
    / "i2t/DeepSeek-OCR-2_deepseek-ocr_i2t.csv"
)


@pytest.fixture(scope="module")
def existing_ocr():
    if not OCR_FIXTURE.exists():
        return json.loads(
            (Path(__file__).parent / "fixtures/ocr_samples.json").read_text()
        )["rows"]
    with OCR_FIXTURE.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2687
    assert len({row["corpus-id"] for row in rows}) == 2687
    return rows[:4]


def test_existing_ocr_preserves_content_and_tables(existing_ocr):
    for row in existing_ocr:
        chunks = list(split_markdown(row["text"], max_chars=600))
        assert chunks
        assert all(0 < len(text) <= 600 for _, text in chunks)
        # Ignore only whitespace introduced/removed at block boundaries.
        assert "".join("".join(text.split()) for _, text in chunks) == "".join(
            row["text"].split()
        )
    assert any(kind == "table" for kind, _ in chunks)


def test_table_order_and_markdown_are_retained():
    text = "**Before**\n\n| Name | Value |\n| --- | --- |\n| A | 10 |\n\n_After_"
    chunks = list(split_markdown(text))
    assert [kind for kind, _ in chunks] == ["prose", "table", "prose"]
    assert chunks[0][1] == "**Before**"
    assert chunks[-1][1] == "_After_"
    assert "| A | 10 |" in chunks[1][1]


def test_html_table_and_oversized_tokens():
    text = "Intro\n<table>\n<tr><td>A</td></tr>\n</table>\nOutro"
    assert [kind for kind, _ in split_markdown(text)] == ["prose", "table", "prose"]
    chunks = list(split_markdown("a" * 31, max_chars=10, overlap_chars=2))
    assert [len(text) for _, text in chunks] == [10, 10, 10, 7]


def test_invalid_chunk_bounds():
    with pytest.raises(ValueError):
        list(split_markdown("text", max_chars=0))
    with pytest.raises(ValueError):
        list(split_markdown("text", max_chars=4, overlap_chars=4))
