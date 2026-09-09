"""Ingest the AB_finRL WRDS extract: ``wrds_tickers`` + ``wrds_processed``.

See ``docs/ab_finrl_contract.md`` for column aliases. This module does not
call WRDS; it only joins the two CSVs onto the canonical schema.

Input
-----
wrds_processed : daily CRSP-like rows, shape ``(N, >=3)``
wrds_tickers   : PERMNO ↔ ticker (+ optional membership dates), shape ``(M, >=2)``

Output
------
Canonical OHLCV (``date, tic, open, high, low, close, volume`` + ``permno``),
shape ``(N', >=8)``, after the CRSP adapter.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sp500rl.data.adapters.wrds_crsp import to_canonical as crsp_to_canonical
from sp500rl.data.schema import validate

TICKER_START_ALIASES = (
    "start",
    "start_date",
    "namedt",
    "mbrstartdt",
    "startdt",
    "begdt",
)
TICKER_END_ALIASES = (
    "ending",
    "end",
    "end_date",
    "nameenddt",
    "mbrenddt",
    "enddt",
    "finish",
)


def _col(df: pd.DataFrame, *names: str) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for name in names:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def load_wrds_tickers(path: str | Path) -> pd.DataFrame:
    """Load the universe / name file.

    Output columns
    --------------
    permno : float/int
    ticker : str
    start, ending : datetime64 or NaT
    Shape ``(M, 4)``. Extra source columns are preserved as ``_extra_*`` only
    when already named; callers typically only need the four fields.
    """

    path = Path(path)
    raw = pd.read_csv(path)
    permno_c = _col(raw, "permno", "PERMNO")
    tic_c = _col(raw, "ticker", "TICKER", "tic", "symbol")
    if permno_c is None:
        raise ValueError(f"{path} has no permno column")
    out = pd.DataFrame(
        {
            "permno": pd.to_numeric(raw[permno_c], errors="coerce"),
            "ticker": raw[tic_c].astype(str) if tic_c else raw[permno_c].astype(str),
        }
    )
    start_c = _col(raw, *TICKER_START_ALIASES)
    end_c = _col(raw, *TICKER_END_ALIASES)
    out["start"] = pd.to_datetime(raw[start_c], errors="coerce") if start_c else pd.NaT
    out["ending"] = pd.to_datetime(raw[end_c], errors="coerce") if end_c else pd.NaT
    out = out.dropna(subset=["permno"])
    return out


def attach_tickers(prices: pd.DataFrame, tickers: pd.DataFrame) -> pd.DataFrame:
    """Left-join ticker (and membership window) onto a price extract.

    Parameters
    ----------
    prices
        Must contain ``permno`` (any case). Shape ``(N, k)``.
    tickers
        Output of :func:`load_wrds_tickers`. Shape ``(M, 4)``.

    Returns
    -------
    pd.DataFrame
        ``prices`` plus ``ticker`` from the matching membership row. Rows
        whose date falls outside every membership window for that PERMNO are
        dropped **only when** ``start`` is populated; otherwise all price
        rows are kept.
    """

    permno_c = _col(prices, "permno", "PERMNO")
    if permno_c is None:
        return prices
    p = prices.copy()
    p["_permno_join"] = pd.to_numeric(p[permno_c], errors="coerce")
    t = tickers.copy()
    t["_permno_join"] = pd.to_numeric(t["permno"], errors="coerce")

    date_c = _col(p, "date", "caldt", "dlycaldt", "datadate")
    if date_c:
        p["_date_join"] = pd.to_datetime(p[date_c], errors="coerce")
    else:
        p["_date_join"] = pd.NaT

    has_window = t["start"].notna().any()
    if not has_window:
        t = t.drop_duplicates("_permno_join", keep="last")
    merged = p.merge(t, on="_permno_join", how="left", suffixes=("", "_mbr"))
    if has_window:
        start = pd.to_datetime(merged["start"], errors="coerce")
        ending = pd.to_datetime(merged["ending"], errors="coerce").fillna(pd.Timestamp("2262-04-11"))
        in_window = merged["_date_join"].isna() | (
            (merged["_date_join"] >= start.fillna(pd.Timestamp("1678-01-01")))
            & (merged["_date_join"] <= ending)
        )
        # Keep rows with no ticker match (start NaT) so unlisted names are not silently dropped.
        no_match = merged["ticker"].isna() & merged["start"].isna()
        merged = merged.loc[in_window | no_match]

    if "ticker" in merged.columns:
        existing = _col(merged, "TICKER", "tic")
        if existing and existing != "ticker":
            merged["ticker"] = merged["ticker"].fillna(merged[existing].astype(str))
        merged["TICKER"] = merged["ticker"]

    return merged.drop(columns=["_permno_join", "_date_join"], errors="ignore")


def load_ab_extract(
    processed_path: str | Path,
    tickers_path: str | Path | None = None,
    *,
    do_validate: bool = True,
) -> pd.DataFrame:
    """Load AB_finRL ``wrds_processed`` (+ optional ``wrds_tickers``) → canonical.

    Parameters
    ----------
    processed_path
        Daily price CSV/parquet.
    tickers_path
        Universe file. If omitted, tickers come from the price file itself.
    do_validate
        Run :func:`validate` after the CRSP adapter.

    Returns
    -------
    pd.DataFrame
        Canonical long format. Shape ``(N, >=7)``.
    """

    processed_path = Path(processed_path)
    suffix = processed_path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        prices = pd.read_parquet(processed_path)
    else:
        prices = pd.read_csv(processed_path)

    if tickers_path is not None and Path(tickers_path).exists():
        tickers = load_wrds_tickers(tickers_path)
        prices = attach_tickers(prices, tickers)

    canonical = crsp_to_canonical(prices)
    # load_table would also coerce date/tic — adapter already maps them.
    canonical["date"] = pd.to_datetime(canonical["date"], errors="coerce")
    canonical["tic"] = canonical["tic"].astype(str)
    canonical = canonical.sort_values(["tic", "date"]).reset_index(drop=True)
    if do_validate:
        validate(canonical).raise_if_invalid()
    return canonical


def listed_tickers(tickers_path: str | Path) -> list[str]:
    """Unique ticker symbols from ``wrds_tickers``, sorted."""

    df = load_wrds_tickers(tickers_path)
    return sorted(df["ticker"].astype(str).str.upper().unique().tolist())
