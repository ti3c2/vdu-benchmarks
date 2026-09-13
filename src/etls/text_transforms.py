"""Conservative OCR segmentation: preserve prose and tables in source order."""


def split_markdown(text: str, max_chars: int = 4000, overlap_chars: int = 0):
    """Yield ``(kind, text)`` chunks without deleting OCR content.

    Pipe tables and HTML tables retain their position amongst prose. Large blocks
    prefer line/word boundaries, and fall back to character boundaries for long
    unbroken tokens. Overlap applies only within a block, never between tables
    and prose. The unmodified source remains in the page representation.
    """
    if max_chars <= 0 or not 0 <= overlap_chars < max_chars:
        raise ValueError("Require max_chars > 0 and 0 <= overlap_chars < max_chars")

    blocks = []
    lines = []
    kind = "prose"
    in_html_table = False
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        starts_html = "<table" in stripped.lower()
        is_pipe = stripped.startswith("|") and stripped.count("|") >= 2
        next_kind = "table" if in_html_table or starts_html or is_pipe else "prose"
        if lines and (next_kind != kind or (not stripped and kind == "prose")):
            block = "".join(lines).strip()
            if block:
                blocks.append((kind, block))
            lines = []
        if stripped or lines:
            lines.append(line)
        kind = next_kind
        if starts_html:
            in_html_table = True
        if "</table>" in stripped.lower():
            in_html_table = False
    if lines:
        block = "".join(lines).strip()
        if block:
            blocks.append((kind, block))

    for kind, block in blocks:
        offset = 0
        while offset < len(block):
            end = min(offset + max_chars, len(block))
            if end < len(block):
                # Avoid tiny pieces when the only available boundary is near
                # the start, including in a single oversized table row.
                lower_bound = offset + max(max_chars // 2, overlap_chars + 1)
                boundary = block.rfind("\n", lower_bound, end)
                if boundary == -1:
                    boundary = block.rfind(" ", lower_bound, end)
                if boundary != -1:
                    end = boundary + 1
            chunk = block[offset:end].strip()
            if chunk:
                yield kind, chunk
            if end == len(block):
                break
            offset = end - overlap_chars
