"""Generate fake price files (canonical, Yahoo, wide, or CRSP-shaped).

The synthetic panel uses the AB_finRL 10-name list, a business-day calendar,
and GBM-like prices so tests and notebooks run without a real extract.
:func:`write_fake` writes a dataset's files in that dataset's ``format``.

Output columns (canonical)
--------------------------
date, tic, open, high, low, close, volume, sector
Shape ``(n_dates * n_tickers, 8)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from sp500rl.data.universe import DEFAULT_SANDBOX_TICKERS

SECTORS = {
    "AAPL": "Information Technology",
    "MSFT": "Information Technology",
    "JNJ": "Health Care",
    "JPM": "Financials",
    "XOM": "Energy",
    "PG": "Consumer Staples",
    "HD": "Consumer Discretionary",
    "UNH": "Health Care",
    "CAT": "Industrials",
    "DIS": "Communication Services",
}


def make_synthetic_canonical(
    start: str = "2019-01-02",
    end: str = "2023-12-29",
    tickers: Sequence[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate a balanced OHLCV panel.

    Parameters
    ----------
    start, end
        Inclusive calendar bounds (business days).
    tickers
        Defaults to the 10-name sandbox.
    seed
        RNG seed.

    Returns
    -------
    pd.DataFrame
        Canonical long format, shape ``(D * n, 8)``.
    """

    tickers = list(tickers or DEFAULT_SANDBOX_TICKERS)
    dates = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for i, tic in enumerate(tickers):
        n = len(dates)
        mu = 0.00025 + 0.00005 * (i - 4)
        sigma = 0.012 + 0.002 * (i % 3)
        shocks = rng.normal(mu, sigma, size=n)
        close = 50.0 + 10.0 * i * np.exp(np.cumsum(shocks))
        close = np.maximum(close, 1.0)
        high = close * (1.0 + rng.uniform(0.001, 0.015, size=n))
        low = close * (1.0 - rng.uniform(0.001, 0.015, size=n))
        open_ = np.r_[close[0], close[:-1]] * (1.0 + rng.normal(0, 0.003, size=n))
        open_ = np.maximum(open_, 0.5)
        volume = rng.integers(500_000, 5_000_000, size=n).astype(float)
        mktcap = close * rng.uniform(1e8, 5e8)
        for j, dt in enumerate(dates):
            rows.append(
                {
                    "date": dt,
                    "tic": tic,
                    "open": float(open_[j]),
                    "high": float(max(high[j], open_[j], close[j])),
                    "low": float(min(low[j], open_[j], close[j])),
                    "close": float(close[j]),
                    "volume": float(volume[j]),
                    "sector": SECTORS.get(tic, "Other"),
                    "market_cap": float(mktcap[j]),
                }
            )
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["tic", "date"]).reset_index(drop=True)


def to_yahoo_format(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical → YahooDownloader-like columns.

    Output columns: ``Date, Open, High, Low, Close, Adj Close, Volume, tic``.
    """

    out = pd.DataFrame(
        {
            "Date": df["date"].dt.strftime("%Y-%m-%d"),
            "Open": df["open"],
            "High": df["high"],
            "Low": df["low"],
            "Close": df["close"],
            "Adj Close": df["close"],
            "Volume": df["volume"].astype(int),
            "tic": df["tic"],
        }
    )
    return out


def to_wide_close(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical → wide close matrix (``date`` + one column per ticker)."""

    wide = df.pivot(index="date", columns="tic", values="close").reset_index()
    wide.columns.name = None
    return wide


def to_wrds_processed(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical → CRSP daily-file shaped ``wrds_processed``.

    Output columns: ``date, permno, TICKER, prc, openprc, askhi, bidlo, vol,
    cfacpr, cfacshr``. One synthetic PERMNO per ticker (10000+i). Close is
    written as ``prc`` (positive).
    """

    tics = sorted(df["tic"].astype(str).unique())
    permno = {tic: 10000 + i for i, tic in enumerate(tics)}
    out = pd.DataFrame(
        {
            "date": df["date"].dt.strftime("%Y-%m-%d"),
            "permno": df["tic"].astype(str).map(permno),
            "TICKER": df["tic"],
            "prc": df["close"],
            "openprc": df["open"],
            "askhi": df["high"],
            "bidlo": df["low"],
            "vol": df["volume"].astype(int),
            "cfacpr": 1.0,
            "cfacshr": 1.0,
        }
    )
    return out


def to_wrds_tickers(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical → ``wrds_tickers`` universe (one row per ticker).

    Output columns: ``permno, ticker, start, ending``.
    """

    tics = sorted(df["tic"].astype(str).unique())
    start = df["date"].min()
    end = df["date"].max()
    rows = [
        {
            "permno": 10000 + i,
            "ticker": tic,
            "start": pd.Timestamp(start).strftime("%Y-%m-%d"),
            "ending": pd.Timestamp(end).strftime("%Y-%m-%d"),
        }
        for i, tic in enumerate(tics)
    ]
    return pd.DataFrame(rows)




def write_fake(
    files: Mapping[str, str | Path],
    fmt: str = "canonical",
    start: str = "2019-01-02",
    end: str = "2023-12-29",
    tickers: Sequence[str] | None = None,
    seed: int = 42,
) -> dict[str, Path]:
    """Write fake files for one dataset entry, in its own column format.

    Parameters
    ----------
    files
        ``{"prices": path, "tickers": path}`` from ``datasets.yaml`` (``tickers``
        optional). Parent directories are created.
    fmt
        ``canonical`` | ``yahoo`` | ``wrds_crsp`` | ``wide``.
    start, end, tickers, seed
        Passed to :func:`make_synthetic_canonical`.

    Returns
    -------
    dict[str, Path]
        Role → written path.
    """

    canonical = make_synthetic_canonical(start=start, end=end, tickers=tickers, seed=seed)
    fmt = fmt.lower()
    if fmt in {"canonical", "generic"}:
        prices = canonical
    elif fmt == "yahoo":
        prices = to_yahoo_format(canonical)
    elif fmt in {"wrds_crsp", "ab_finrl"}:
        prices = to_wrds_processed(canonical)
    elif fmt == "wide":
        prices = to_wide_close(canonical)
    else:
        raise ValueError(f"Unknown format {fmt!r} for fake data")

    written: dict[str, Path] = {}
    prices_path = Path(files["prices"])
    prices_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(prices_path, index=False)
    written["prices"] = prices_path
    if files.get("tickers"):
        tickers_path = Path(files["tickers"])
        tickers_path.parent.mkdir(parents=True, exist_ok=True)
        to_wrds_tickers(canonical).to_csv(tickers_path, index=False)
        written["tickers"] = tickers_path
    return written
