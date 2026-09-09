"""Per-ticker technical indicators with no look-ahead (past + current bar only).

Indicators match the FinRL Portfolio Allocation tutorial:

* ``macd``
* ``rsi_30``
* ``cci_30``
* ``dx_30``

plus ``log_return`` and ``rolling_vol``. The 252-day rolling covariance
state from that notebook is **not** built; POE uses a raw ``time_window``.

Input
-----
Canonical OHLCV: ``date, tic, open, high, low, close, volume``. Shape ``(N, >=7)``.

Output
------
The input columns plus indicator columns. Shape ``(N, >=7 + k)``. Leading
indicator NaNs are left in place for :mod:`sp500rl.data.panel` to drop after
warm-up.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from stockstats import StockDataFrame


def _indicators_one_ticker(
    g: pd.DataFrame,
    indicators: Sequence[str],
    rolling_vol_window: int,
    log_return: bool,
) -> pd.DataFrame:
    """Compute features for a single ticker.

    Input columns: ``open, high, low, close, volume`` (and ``date``, ``tic``).
    Output: same rows with extra feature columns. No future rows are used.
    """

    out = g.sort_values("date").copy()
    ohlcv = out[["open", "high", "low", "close", "volume"]].astype(float)
    sdf = StockDataFrame.retype(ohlcv.copy())
    for name in indicators:
        out[name] = pd.to_numeric(sdf[name], errors="coerce").to_numpy()
    close = out["close"].astype(float)
    if log_return:
        out["log_return"] = np.log(close / close.shift(1))
    if rolling_vol_window:
        lr = out["log_return"] if "log_return" in out.columns else np.log(close / close.shift(1))
        out["rolling_vol"] = lr.rolling(window=int(rolling_vol_window), min_periods=int(rolling_vol_window)).std()
    return out


def add_features(
    df: pd.DataFrame,
    indicators: Iterable[str] = ("macd", "rsi_30", "cci_30", "dx_30"),
    log_return: bool = True,
    rolling_vol_window: int = 20,
) -> pd.DataFrame:
    """Add indicators, grouping by ``tic``.

    Parameters
    ----------
    df
        Canonical prices, shape ``(N, >=7)``.
    indicators
        stockstats column names.
    log_return
        If True, add ``log_return = ln(close_t / close_{t-1})``.
    rolling_vol_window
        Window for ``rolling_vol`` (std of log returns). 0 disables it.

    Returns
    -------
    pd.DataFrame
        Shape ``(N, >=7 + k)``, sorted by ``tic``, ``date``.
    """

    indicators = list(indicators)
    frames: list[pd.DataFrame] = []
    for _, g in df.groupby("tic", sort=False):
        frames.append(
            _indicators_one_ticker(
                g,
                indicators=indicators,
                rolling_vol_window=rolling_vol_window,
                log_return=log_return,
            )
        )
    if not frames:
        return df.copy()
    return pd.concat(frames, ignore_index=True).sort_values(["tic", "date"]).reset_index(drop=True)
