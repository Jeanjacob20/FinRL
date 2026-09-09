"""No-op adapter for files that are already canonical.

Input / output columns
----------------------
``date, tic, open, high, low, close, volume`` (+ optional extras).
Shape ``(N, >=7)``.
"""

from __future__ import annotations

import pandas as pd


def to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` unchanged (copy)."""

    return df.copy()
