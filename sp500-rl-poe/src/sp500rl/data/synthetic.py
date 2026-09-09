"""Generate a fake canonical (and Yahoo / wide) price file for offline runs.

The synthetic panel has the sandbox ticker list, a business-day calendar, and
GBM-like prices so notebooks run without a WRDS extract.

Output columns (canonical)
--------------------------
date, tic, open, high, low, close, volume, sector
Shape ``(n_dates * n_tickers, 8)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

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


def write_synthetic(
    raw_dir: str | Path,
    start: str = "2019-01-02",
    end: str = "2023-12-29",
    seed: int = 42,
) -> dict[str, Path]:
    """Write canonical, Yahoo, and wide CSVs under ``raw_dir``.

    Returns a dict of kind → path.
    """

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    canonical = make_synthetic_canonical(start=start, end=end, seed=seed)
    paths = {
        "canonical": raw_dir / "synthetic_canonical.csv",
        "yahoo": raw_dir / "synthetic_yahoo.csv",
        "wide": raw_dir / "synthetic_wide.csv",
    }
    canonical.to_csv(paths["canonical"], index=False)
    to_yahoo_format(canonical).to_csv(paths["yahoo"], index=False)
    to_wide_close(canonical).to_csv(paths["wide"], index=False)
    return paths
