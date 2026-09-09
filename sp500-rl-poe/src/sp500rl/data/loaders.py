"""Tolerant CSV/parquet loaders that emit the canonical long-format schema.

Input
-----
A CSV or parquet file that is either:

* long format with some spelling of ``date``, ``tic``, OHLCV
* wide format: a date column plus one column per ticker (values treated as close)

Output
------
``pd.DataFrame`` with columns ``date, tic, open, high, low, close, volume``
plus any extra columns that survived mapping. Shape ``(N, >=7)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from sp500rl.data.schema import REQUIRED_COLUMNS, validate

DATE_ALIASES: tuple[str, ...] = (
    "date",
    "Date",
    "DATE",
    "datadate",
    "caldt",
    "time",
    "Time",
    "datetime",
    "Datetime",
)
TIC_ALIASES: tuple[str, ...] = (
    "tic",
    "ticker",
    "symbol",
    "TICKER",
    "Tic",
    "Ticker",
    "Symbol",
    "permno",
)
OHLCV_ALIASES: dict[str, tuple[str, ...]] = {
    "open": ("open", "Open", "OPEN", "openprc", "Openprc"),
    "high": ("high", "High", "HIGH", "askhi", "Askhi"),
    "low": ("low", "Low", "LOW", "bidlo", "Bidlo"),
    "close": ("close", "Close", "CLOSE", "prc", "PRC", "adjcp", "Adj Close", "adj_close"),
    "volume": ("volume", "Volume", "VOLUME", "vol", "VOL", "Vol"),
}
OPTIONAL_ALIASES: dict[str, tuple[str, ...]] = {
    "adj_close": ("adj_close", "Adj Close", "adjcp", "ADJ_CLOSE"),
    "market_cap": ("market_cap", "mktcap", "MKTCAP", "marketcap"),
    "sector": ("sector", "Sector", "gsector"),
    "permno": ("permno", "PERMNO"),
}

AdapterFn = Callable[[pd.DataFrame], pd.DataFrame]


def _flatten_columns(columns) -> list[str]:
    if isinstance(columns, pd.MultiIndex):
        return ["_".join(str(x) for x in tup if str(x) != "").strip() for tup in columns]
    return [str(c) for c in columns]


def _rename_with_aliases(
    df: pd.DataFrame, column_map: Mapping[str, str] | None
) -> pd.DataFrame:
    """Rename using an explicit map first, then built-in aliases."""

    out = df.copy()
    out.columns = _flatten_columns(out.columns)
    if column_map:
        out = out.rename(columns=dict(column_map))

    rename: dict[str, str] = {}
    cols = list(out.columns)

    if "date" not in out.columns:
        for alias in DATE_ALIASES:
            if alias in cols:
                rename[alias] = "date"
                break

    if "tic" not in out.columns:
        for alias in TIC_ALIASES:
            if alias in cols and alias != "permno":
                rename[alias] = "tic"
                break

    for canonical, aliases in {**OHLCV_ALIASES, **OPTIONAL_ALIASES}.items():
        if canonical in out.columns or canonical in rename.values():
            continue
        for alias in aliases:
            if alias in cols and alias not in rename:
                # Do not steal Adj Close as close when a Close column exists.
                if canonical == "close" and alias in {"adjcp", "Adj Close", "adj_close"}:
                    if any(c in cols for c in ("close", "Close", "CLOSE", "prc", "PRC")):
                        continue
                rename[alias] = canonical
                break

    if rename:
        out = out.rename(columns=rename)
    return out


def _maybe_reset_date_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex) or df.index.name in DATE_ALIASES:
        out = df.reset_index()
        return out
    return df


def _is_wide(df: pd.DataFrame) -> bool:
    """True when there is a date column and no close/tic long-format fields."""

    if "date" not in df.columns:
        return False
    if "tic" in df.columns and "close" in df.columns:
        return False
    if "close" in df.columns and "open" in df.columns:
        return False
    value_cols = [c for c in df.columns if c != "date"]
    return len(value_cols) >= 2


def _wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Melt date + ticker columns into canonical OHLCV (OHL = close, volume=0).

    Input columns
    -------------
    date, then one column per ticker.

    Output columns
    --------------
    date, tic, open, high, low, close, volume
    """

    value_cols = [c for c in df.columns if c != "date"]
    long = df.melt(id_vars=["date"], value_vars=value_cols, var_name="tic", value_name="close")
    long["open"] = long["close"]
    long["high"] = long["close"]
    long["low"] = long["close"]
    long["volume"] = 0.0
    return long


def _coerce_canonical(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["tic"] = out["tic"].astype(str)
    for col in ("open", "high", "low", "close", "volume"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "permno" in out.columns:
        out["permno"] = pd.to_numeric(out["permno"], errors="coerce")
    if "market_cap" in out.columns:
        out["market_cap"] = pd.to_numeric(out["market_cap"], errors="coerce")
    out = out.sort_values(["tic", "date"]).reset_index(drop=True)
    return out


def _resolve_adapter(adapter: str | AdapterFn | None) -> AdapterFn | None:
    if adapter is None:
        return None
    if callable(adapter):
        return adapter
    name = str(adapter).lower()
    from sp500rl.data.adapters import get_adapter

    return get_adapter(name)


def _load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_table(
    path: str | Path,
    adapter: str | AdapterFn | None = None,
    column_map: Mapping[str, str] | None = None,
    do_validate: bool = True,
) -> pd.DataFrame:
    """Load CSV or parquet, map columns, optionally run an adapter, validate.

    Parameters
    ----------
    path
        File path.
    adapter
        ``None``/``"generic"``, ``"wrds_crsp"``, ``"yahoo"``, or a
        ``to_canonical(df) -> df`` callable.
    column_map
        Source column name → canonical name. Applied before aliases.
    do_validate
        If True (default), run :func:`validate` and raise on failure.

    Returns
    -------
    pd.DataFrame
        Canonical long format, shape ``(N, >=7)``.
    """

    raw = _load_table(path)
    raw = _maybe_reset_date_index(raw)
    mapped = _rename_with_aliases(raw, column_map)
    if _is_wide(mapped):
        mapped = _wide_to_long(mapped)

    adapter_fn = _resolve_adapter(adapter)
    if adapter_fn is not None:
        mapped = adapter_fn(mapped)

    if "tic" not in mapped.columns and "permno" in mapped.columns:
        mapped = mapped.copy()
        mapped["tic"] = mapped["permno"].astype(str)

    missing = [c for c in REQUIRED_COLUMNS if c not in mapped.columns]
    if missing:
        raise ValueError(
            f"{path} is missing canonical columns {missing} after mapping. "
            "Pass column_map or a source adapter."
        )

    canonical = _coerce_canonical(mapped)
    if do_validate:
        validate(canonical).raise_if_invalid()
    return canonical


def load_csv(
    path: str | Path,
    adapter: str | AdapterFn | None = None,
    column_map: Mapping[str, str] | None = None,
    do_validate: bool = True,
) -> pd.DataFrame:
    """See :func:`load_table`. CSV only."""

    return load_table(path, adapter=adapter, column_map=column_map, do_validate=do_validate)


def load_parquet(
    path: str | Path,
    adapter: str | AdapterFn | None = None,
    column_map: Mapping[str, str] | None = None,
    do_validate: bool = True,
) -> pd.DataFrame:
    """See :func:`load_table`. Parquet only."""

    return load_table(path, adapter=adapter, column_map=column_map, do_validate=do_validate)
