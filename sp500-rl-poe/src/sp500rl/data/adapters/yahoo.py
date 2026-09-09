"""Yahoo Finance / FinRL ``YahooDownloader`` → canonical schema.

Handles
-------
* ``Date`` / ``Adj Close`` / ``Open`` / ``High`` / ``Low`` / ``Close`` / ``Volume``
* FinRL ``YahooDownloader`` output: ``date, open, high, low, close, volume, tic``
  and optional ``adjcp``
* ``ticker`` / ``symbol`` as the ticker column

Input
-----
Long-format Yahoo CSV, any of the spellings above. Shape ``(N, >=6)``.

Output
------
Canonical ``date, tic, open, high, low, close, volume`` plus ``adj_close``
when present. Shape ``(N, >=7)``.
"""

from __future__ import annotations

import pandas as pd


def to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Map Yahoo-style columns onto the canonical schema.

    Parameters
    ----------
    df
        Long-format frame. Loaders may already have renamed ``Date`` → ``date``.

    Returns
    -------
    pd.DataFrame
        Canonical columns. Prices are left as-is (Yahoo already uses positive
        last-trade prices).
    """

    out = df.copy()
    rename: dict[str, str] = {}
    cols = {c.lower(): c for c in out.columns}

    def take(canonical: str, *aliases: str) -> None:
        if canonical in out.columns:
            return
        for alias in aliases:
            if alias in out.columns:
                rename[alias] = canonical
                return
            if alias.lower() in cols:
                rename[cols[alias.lower()]] = canonical
                return

    take("date", "Date", "Datetime")
    take("tic", "ticker", "symbol", "Ticker", "Symbol")
    take("open", "Open")
    take("high", "High")
    take("low", "Low")
    take("close", "Close")
    take("volume", "Volume")
    take("adj_close", "Adj Close", "adjcp", "AdjClose")
    if rename:
        out = out.rename(columns=rename)

    if "volume" in out.columns:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0).clip(lower=0)
    return out
