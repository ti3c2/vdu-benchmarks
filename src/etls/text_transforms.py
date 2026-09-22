"""Segment OCR Markdown by source blocks, preserving lists and pipe tables."""

import re

from markdown_it import MarkdownIt


def remove_tables(text: str) -> str:
    """Remove Markdown, HTML, and LaTeX tables, retaining surrounding source text.

    As in chunking, pipe rows without a valid separator are treated as OCR
    tables, including rows missing an outer pipe. HTML layout tables and LaTeX
    array/tabular environments are removed too. Unclosed HTML/LaTeX tables
    extend to the end of the page. Code blocks are kept.
    """
    line_offsets = [0, *(match.end() for match in re.finditer(r"\r\n?|\n", text))]
    line_offsets.append(len(text))
    tokens = MarkdownIt("commonmark", {"html": False}).enable("table").parse(text)
    removed = []
    code_blocks = []
    for token in tokens:
        if token.map is None:
            continue
        start, end = (line_offsets[line] for line in token.map)
        if token.type in {"fence", "code_block"}:
            code_blocks.append((start, end))
        elif token.type == "table_open":
            removed.append((start, end))
        elif token.type == "paragraph_open":
            pipe_rows = []
            for line in range(*token.map):
                line_start, line_end = line_offsets[line : line + 2]
                stripped = text[line_start:line_end].strip()
                if len(re.findall(r"(?<!\\)\|", stripped)) >= 2:
                    pipe_rows.append((line, line_start, line_end, stripped))
            for index, (line, line_start, line_end, stripped) in enumerate(pipe_rows):
                if (
                    stripped.startswith("|")
                    or stripped.endswith("|")
                    or (index > 0 and pipe_rows[index - 1][0] == line - 1)
                    or (
                        index + 1 < len(pipe_rows)
                        and pipe_rows[index + 1][0] == line + 1
                    )
                ):
                    removed.append((line_start, line_end))

    depth = 0
    table_start = None
    for tag in re.finditer(r"</?table\b[^>]*>", text, re.IGNORECASE):
        if any(start <= tag.start() < end for start, end in code_blocks):
            continue
        if tag.group().startswith("</"):
            if depth:
                depth -= 1
                if depth == 0:
                    removed.append((table_start, tag.end()))
        else:
            if depth == 0:
                table_start = tag.start()
            depth += 1
    if depth:
        removed.append((table_start, len(text)))

    environments = []
    for tag in re.finditer(
        r"\\(begin|end)\{(array|tabular\*?|table\*?|smalltable)\}", text
    ):
        if any(start <= tag.start() < end for start, end in code_blocks):
            continue
        if tag[1] == "begin":
            if not environments:
                table_start = tag.start()
                # Remove an adjacent display-math wrapper with its table.
                wrapper = re.search(r"(?:\\\[|\$\$)\s*$", text[:table_start])
                if wrapper:
                    table_start = wrapper.start()
            environments.append(tag[2])
        elif environments and environments[-1] == tag[2]:
            environments.pop()
            if not environments:
                wrapper = re.match(r"\s*(?:\\\]|\$\$)", text[tag.end() :])
                removed.append(
                    (table_start, tag.end() + (wrapper.end() if wrapper else 0))
                )
    if environments:
        removed.append((table_start, len(text)))

    parts = []
    cursor = 0
    for start, end in sorted(removed):
        if start > cursor:
            parts.append(text[cursor:start])
        cursor = max(cursor, end)
    parts.append(text[cursor:])
    return "".join(parts)


def split_markdown(text: str, max_chars: int = 4000, overlap_chars: int = 0):
    """Yield ``(kind, text)`` chunks without deleting OCR content.

    Markdown containers (lists, blockquotes, and code blocks) stay together when
    they fit. Oversized lists prefer item boundaries; other oversized blocks
    prefer line/word boundaries before falling back to character boundaries.
    Overlap applies only within an oversized block. Only pipe tables get the
    ``table`` kind; other Markdown blocks use ``prose``.
    """
    if max_chars <= 0 or not 0 <= overlap_chars < max_chars:
        raise ValueError("Require max_chars > 0 and 0 <= overlap_chars < max_chars")

    # Use source maps, not rendered tokens: OCR tables can contain extra cells
    # that a Markdown renderer would discard. Match the parser's line endings
    # without treating Unicode line separators inside a paragraph as new lines.
    line_offsets = [0, *(match.end() for match in re.finditer(r"\r\n?|\n", text))]
    line_offsets.append(len(text))
    tokens = MarkdownIt("commonmark", {"html": False}).enable("table").parse(text)
    item_ends = [
        line_offsets[token.map[1]]
        for token in tokens
        if token.type == "list_item_open" and token.level == 1 and token.map
    ]
    blocks = []
    cursor = 0
    for token in tokens:
        if token.level != 0 or token.map is None:
            continue
        start, end = (line_offsets[line] for line in token.map)
        # Reference definitions have no visible tokens, but remain OCR content.
        if start > cursor:
            blocks.append(("prose", cursor, start))
        if token.type == "paragraph_open":
            # OCR can omit table separators or give them the wrong cell count.
            # Retain such pipe runs as tables, but never scan inside containers
            # such as lists or fenced code for apparent table rows.
            kind = None
            block_start = start
            for line in range(*token.map):
                line_start, line_end = line_offsets[line : line + 2]
                stripped = text[line_start:line_end].strip()
                next_kind = (
                    "table"
                    if stripped.startswith("|") and stripped.count("|") >= 2
                    else "prose"
                )
                if kind is not None and next_kind != kind:
                    blocks.append((kind, block_start, line_start))
                    block_start = line_start
                kind = next_kind
            blocks.append((kind, block_start, end))
        else:
            kind = (
                "table"
                if token.type == "table_open"
                else "list"
                if token.type in {"bullet_list_open", "ordered_list_open"}
                else "prose"
            )
            blocks.append((kind, start, end))
        cursor = end
    if cursor < len(text):
        blocks.append(("prose", cursor, len(text)))

    for kind, start, end in blocks:
        source = text[start:end]
        start += len(source) - len(source.lstrip())
        block = source.strip()
        boundaries = (
            [
                len(text[start:item_end].rstrip())
                for item_end in item_ends
                if start < item_end <= end
            ]
            if kind == "list"
            else []
        )
        offset = 0
        while offset < len(block):
            # Separating blank lines must not force an otherwise fitting item
            # over the limit after a split at the preceding item's end.
            while offset < len(block) and block[offset].isspace():
                offset += 1
            end = min(offset + max_chars, len(block))
            if end < len(block):
                item_boundary = max(
                    (b for b in boundaries if offset + overlap_chars < b <= end),
                    default=None,
                )
                if item_boundary is not None:
                    end = item_boundary
                else:
                    # A single item/row may exceed the limit. Avoid tiny pieces
                    # and ensure forward progress even with very large overlap.
                    lower_bound = offset + max(max_chars // 2, overlap_chars + 1)
                    boundary = block.rfind("\n", lower_bound, end)
                    if boundary == -1:
                        boundary = block.rfind(" ", lower_bound, end)
                    if boundary != -1:
                        end = boundary + 1
            chunk = block[offset:end].strip()
            if chunk:
                yield "table" if kind == "table" else "prose", chunk
            if end == len(block):
                break
            offset = end - overlap_chars
