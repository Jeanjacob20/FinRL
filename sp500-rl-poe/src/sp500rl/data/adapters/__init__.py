"""Source adapters: each exposes ``to_canonical(df) -> df``.

Adding a source
---------------
Create ``src/sp500rl/data/adapters/<name>.py`` with::

    def to_canonical(df: pd.DataFrame) -> pd.DataFrame:
        ...

and register it in ``ADAPTERS`` below. The function must return a frame that
passes :func:`sp500rl.data.schema.validate`.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from sp500rl.data.adapters import generic as generic_mod
from sp500rl.data.adapters import wrds_crsp as wrds_crsp_mod
from sp500rl.data.adapters import yahoo as yahoo_mod

AdapterFn = Callable[[pd.DataFrame], pd.DataFrame]

ADAPTERS: dict[str, AdapterFn] = {
    "generic": generic_mod.to_canonical,
    "wrds_crsp": wrds_crsp_mod.to_canonical,
    "yahoo": yahoo_mod.to_canonical,
}


def get_adapter(name: str) -> AdapterFn:
    """Look up an adapter by config name (``generic``, ``wrds_crsp``, ``yahoo``)."""

    key = name.lower()
    if key not in ADAPTERS:
        known = ", ".join(sorted(ADAPTERS))
        raise KeyError(f"Unknown adapter {name!r}. Known: {known}")
    return ADAPTERS[key]
