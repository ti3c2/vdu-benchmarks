import typing as tp

import regex as re

TABLE_RE = re.compile(
    r"(?P<title_row>(?:\|\s*[^|\r\n]+\s*)+\|)"
    r"(?:\r?\n)"
    r"((?:\|[\s:]?-+[\s:]?)+\|)"
    r"(?P<rows>(?:(?:\r?\n)(?:\|\s*[^|\r\n]+\s*)+\|)+)"
)


def split_tables(text: str) -> tuple[str, list[str]]:
    """Return prose with tables removed, plus the extracted table strings."""
    tables = [m.group(0) for m in TABLE_RE.finditer(text)]
    text_without_tables = TABLE_RE.sub("", text)
    text_without_tables = re.sub(r"\n{3,}", "\n\n", text_without_tables)
    return text_without_tables, tables


def remove_hrule(text: str) -> str:
    return re.sub(r"^\n[-]{2,}$\n", "", text, flags=re.MULTILINE)


def remove_emphasis(text: str) -> str:
    return re.sub(r"[_\*\~]", r"", text, flags=re.MULTILINE)


def remove_first_char_on_line_space(text: str) -> str:
    return re.sub(r"^ ", "", text, flags=re.MULTILINE)


def remove_nonstring_starting_lines(text: str) -> str:
    return re.sub(r"^[^\w\d#\n].+$", "", text, flags=re.MULTILINE)


def remove_nl_sequences(text: str) -> str:
    return re.sub(r"\n{2,}", "\n\n", text, flags=re.MULTILINE)


def sequential_transforms(*funcs: tp.Callable[[str], str]) -> str:
    def wrapper(text: str) -> str:
        for func in funcs:
            text = func(text)
        return text

    return wrapper


text_cleanup_transforms = sequential_transforms(
    remove_hrule,
    remove_emphasis,
    remove_first_char_on_line_space,
    remove_nonstring_starting_lines,
    remove_nl_sequences,
)
