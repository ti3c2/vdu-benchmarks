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


@pytest.mark.parametrize(
    "markers", [("-", "-"), ("*", "*"), ("+", "+"), ("1.", "2."), ("1)", "2)")]
)
def test_loose_lists_keep_nested_items_and_continuation_paragraphs(markers):
    first, second = markers
    indent = " " * (len(first) + 1)
    listing = (
        f"{first} First item\n"
        f"{indent}wrapped continuation\n\n"
        f"{indent}Another paragraph in the same item.\n\n"
        f"{indent}- Nested item\n\n"
        f"{indent}- Another nested item\n\n"
        f"{second} Second item"
    )
    text = f"Introduction.\n\n{listing}\n\nConclusion."
    assert list(split_markdown(text)) == [
        ("prose", "Introduction."),
        ("prose", listing),
        ("prose", "Conclusion."),
    ]


def test_list_can_interrupt_a_paragraph_without_a_blank_line():
    text = "Benefits:\n- First benefit\n- Second benefit\n\nConclusion."
    assert list(split_markdown(text)) == [
        ("prose", "Benefits:"),
        ("prose", "- First benefit\n- Second benefit"),
        ("prose", "Conclusion."),
    ]


def test_oversized_list_splits_between_whole_items():
    items = [
        "- Short item",
        "- A longer item\n  with a continuation paragraph.",
        "- Last item",
    ]
    text = "\n\n".join(items)
    assert list(split_markdown(text, max_chars=len(items[1]))) == [
        ("prose", item) for item in items
    ]


@pytest.mark.parametrize(
    "text",
    [
        "- " + "long item " * 40 + "\n\n- Next item",
        "| Name | Value |\n| --- | --- |\n| " + "long cell " * 40 + "| 10 |",
    ],
)
def test_oversized_items_and_table_rows_remain_bounded_and_lossless(text):
    chunks = list(split_markdown(text, max_chars=60))
    assert all(0 < len(chunk) <= 60 for _, chunk in chunks)
    assert "".join("".join(chunk.split()) for _, chunk in chunks) == "".join(
        text.split()
    )


@pytest.mark.parametrize(
    "block",
    [
        "A paragraph wrapped\nonto another line.",
        "> A quoted paragraph.\n>\n> Another quoted paragraph.",
        "```markdown\n- Not an actual list\n\n| A | B |\n| --- | --- |\n```",
    ],
)
def test_markdown_blocks_preserve_their_original_text(block):
    text = f"# Heading\n\n{block}\n\nAfter."
    assert list(split_markdown(text)) == [
        ("prose", "# Heading"),
        ("prose", block),
        ("prose", "After."),
    ]


@pytest.mark.parametrize(
    "table",
    [
        "Name | Value\n--- | ---\nA | 10",
        "| Name | Value |\n| --- | --- |\n| A | 10 | extra OCR cell |",
        "| Name | Value |\n| --- | --- | --- |\n| A | 10 |",
        "| Name | Value |\n| A | 10 |",
    ],
)
def test_pipe_tables_preserve_cells_even_with_imperfect_ocr(table):
    text = f"Before.\n\n{table}\n\nAfter."
    assert list(split_markdown(text)) == [
        ("prose", "Before."),
        ("table", table),
        ("prose", "After."),
    ]


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_source_maps_preserve_reference_definitions_and_line_endings(newline):
    text = newline.join(
        [
            "[first]: /first",
            "",
            "Wrapped\u2028text [first]",
            "",
            "- One",
            "",
            "- Two",
            "",
            "[last]: /last",
        ]
    )
    chunks = list(split_markdown(text))
    assert ("prose", newline.join(["- One", "", "- Two"])) in chunks
    assert "".join("".join(chunk.split()) for _, chunk in chunks) == "".join(
        text.split()
    )
    assert any("[first]: /first" in chunk for _, chunk in chunks)
    assert any("[last]: /last" in chunk for _, chunk in chunks)


def test_overlap_stays_inside_each_markdown_block():
    text = "- " + "x" * 31 + "\n\n| A | B |\n| --- | --- |\n\nAfter."
    chunks = list(split_markdown(text, max_chars=10, overlap_chars=2))
    assert chunks[-1] == ("prose", "After.")
    assert all("x" not in chunk for kind, chunk in chunks if kind == "table")
    assert all("|" not in chunk for kind, chunk in chunks if kind == "prose")


@pytest.mark.parametrize("text", ["", " \n\t\r\n "])
def test_empty_markdown_has_no_chunks(text):
    assert list(split_markdown(text)) == []


def test_oversized_tokens():
    chunks = list(split_markdown("a" * 31, max_chars=10, overlap_chars=2))
    assert [len(text) for _, text in chunks] == [10, 10, 10, 7]


def test_invalid_chunk_bounds():
    with pytest.raises(ValueError):
        list(split_markdown("text", max_chars=0))
    with pytest.raises(ValueError):
        list(split_markdown("text", max_chars=4, overlap_chars=4))
