"""CRSP daily-file conventions → canonical schema.

CRSP conventions handled here (applied only when the columns exist):

* ``prc`` is the close; a **negative** value is a bid/ask midpoint → ``abs``
* ``openprc`` / ``askhi`` / ``bidlo`` / ``vol`` map to OHLCV
* ``cfacpr`` / ``cfacshr`` split-adjust prices and volume when present:
  ``price_adj = abs(price) / cfacpr``, ``volume_adj = volume * cfacshr``
* ``TICKER`` / ``ticker`` → ``tic``; else ``permno`` as string
* If several name records exist for a ``permno``, the ticker on that row is
  used; missing tickers are forward-filled then back-filled within ``permno``

Input columns (any subset)
--------------------------
``date``/``caldt``, ``permno``, ``ticker``/``TICKER``, ``prc``, ``openprc``,
``askhi``, ``bidlo``, ``vol``, ``cfacpr``, ``cfacshr``, ``ret``

Output
------
Canonical ``date, tic, open, high, low, close, volume`` plus ``permno`` when
present. Shape ``(N, >=7)``.
"""

from __future__ import annotations

import pandas as pd


def _col(df: pd.DataFrame, *names: str) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for name in names:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a CRSP-like daily file to the canonical schema.

    Parameters
    ----------
    df
        Long-format CRSP extract. Shape ``(N, >=3)``.

    Returns
    -------
    pd.DataFrame
        Canonical OHLCV with strictly positive prices (absolute value +
        optional split adjustment).
    """

    out = df.copy()
    rename: dict[str, str] = {}

    date_c = _col(out, "date", "caldt", "datadate")
    if date_c and date_c != "date":
        rename[date_c] = "date"

    close_c = _col(out, "close", "prc")
    if close_c and close_c != "close":
        rename[close_c] = "close"

    open_c = _col(out, "open", "openprc")
    if open_c and open_c != "open":
        rename[open_c] = "open"

    high_c = _col(out, "high", "askhi")
    if high_c and high_c != "high":
        rename[high_c] = "high"

    low_c = _col(out, "low", "bidlo")
    if low_c and low_c != "low":
        rename[low_c] = "low"

    vol_c = _col(out, "volume", "vol")
    if vol_c and vol_c != "volume":
        rename[vol_c] = "volume"

    tic_c = _col(out, "tic", "ticker", "TICKER")
    if tic_c and tic_c != "tic":
        rename[tic_c] = "tic"

    permno_c = _col(out, "permno")
    if permno_c and permno_c != "permno":
        rename[permno_c] = "permno"

    if rename:
        out = out.rename(columns=rename)

    if "tic" not in out.columns and "permno" in out.columns:
        out["tic"] = out["permno"].astype(str)
    elif "permno" in out.columns and "tic" in out.columns:
        out["tic"] = out["tic"].astype("string")
        out["tic"] = out.groupby("permno")["tic"].transform(lambda s: s.ffill().bfill())
        missing = out["tic"].isna() | (out["tic"].astype(str).str.lower().isin(["nan", "<na>", "none"]))
        out.loc[missing, "tic"] = out.loc[missing, "permno"].astype(str)

    price_cols = [c for c in ("open", "high", "low", "close") if c in out.columns]
    for col in price_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").abs()

    if "volume" not in out.columns:
        out["volume"] = 0.0
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0).abs()

    cfacpr_c = _col(out, "cfacpr")
    cfacshr_c = _col(out, "cfacshr")
    if cfacpr_c is not None:
        cfacpr = pd.to_numeric(out[cfacpr_c], errors="coerce").replace(0, pd.NA)
        cfacpr = cfacpr.fillna(1.0)
        for col in price_cols:
            out[col] = out[col] / cfacpr
    if cfacshr_c is not None:
        cfacshr = pd.to_numeric(out[cfacshr_c], errors="coerce").fillna(1.0)
        out["volume"] = out["volume"] * cfacshr

    # If OHLC are incomplete, fill from close.
    if "close" in out.columns:
        for col in ("open", "high", "low"):
            if col not in out.columns:
                out[col] = out["close"]
            else:
                out[col] = out[col].fillna(out["close"])

    keep_extra = [c for c in ("permno", "adj_close", "market_cap", "sector", "ret") if c in out.columns]
    cols = ["date", "tic", "open", "high", "low", "close", "volume"] + keep_extra
    cols = list(dict.fromkeys(c for c in cols if c in out.columns))
    return out[cols]
