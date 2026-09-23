"""Shared tqdm progress bars for CLI-facing benchmark stages."""

import sys

from tqdm import tqdm


def progress_bar(*, total: int, desc: str, unit: str = "item"):
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        leave=False,
        disable=None,
        file=sys.stderr,
    )
