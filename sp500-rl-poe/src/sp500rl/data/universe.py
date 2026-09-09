"""Fixed-universe selection on a canonical price frame.

POE requires the *same ticker set on every date* (balanced panel). That
conflicts with a point-in-time S&P 500. Every rule below is therefore
survivorship-biased except the explicit sandbox list. Report the bias.

Rules (config ``universe.rule``)
--------------------------------
sandbox
    Hand-picked liquid names across sectors.
full_window
    Tickers present on **every** date in ``[start, end]``.
top_n
    Top-N by ``market_cap`` on the first training date if that column exists,
    else top-N by average dollar volume over the window. Held fixed after that.

Input
-----
Canonical long-format frame: ``date, tic, open, high, low, close, volume``
(+ optional ``market_cap``). Shape ``(N, >=7)``.

Output
------
The same columns, filtered to the chosen tickers. Shape ``(M, >=7)``, ``M <= N``.
"""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

DEFAULT_SANDBOX_TICKERS: tuple[str, ...] = (
    "AAPL",
    "MSFT",
    "JNJ",
    "JPM",
    "XOM",
    "PG",
    "HD",
    "UNH",
    "CAT",
    "DIS",
)


def _window(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    out = df
    if start is not None:
        out = out.loc[out["date"] >= pd.Timestamp(start)]
    if end is not None:
        out = out.loc[out["date"] <= pd.Timestamp(end)]
    return out


def sandbox_tickers(cfg: dict[str, Any] | None = None) -> list[str]:
    """Return the configured sandbox list (10 names by default)."""

    if cfg is None:
        return list(DEFAULT_SANDBOX_TICKERS)
    uni = cfg.get("universe", cfg)
    tickers = uni.get("sandbox_tickers", DEFAULT_SANDBOX_TICKERS)
    return [str(t) for t in tickers]


def select_full_window(df: pd.DataFrame) -> list[str]:
    """Tickers observed on every distinct date in ``df``.

    Input columns: ``date``, ``tic``. Output: list of ticker strings.
    """

    n_dates = df["date"].nunique()
    counts = df.groupby("tic")["date"].nunique()
    return sorted(counts[counts == n_dates].index.astype(str).tolist())


def select_top_n(
    df: pd.DataFrame,
    n: int,
    asof: str | pd.Timestamp | None = None,
    metric: str = "market_cap",
) -> list[str]:
    """Top-N tickers by market cap on ``asof``, else average dollar volume.

    Parameters
    ----------
    df
        Canonical frame. Uses ``market_cap`` if present and ``metric`` is
        ``market_cap``; otherwise ``close * volume``.
    n
        Number of names to keep.
    asof
        Ranking date (first training day). Defaults to the first date in ``df``.
    metric
        ``market_cap`` or ``dollar_volume``.

    Returns
    -------
    list[str]
        Tickers, highest rank first.
    """

    if asof is None:
        asof = df["date"].min()
    asof_ts = pd.Timestamp(asof)
    day = df.loc[df["date"] == asof_ts]
    if day.empty:
        day = df.loc[df["date"] == df["date"].min()]

    if metric == "market_cap" and "market_cap" in df.columns and day["market_cap"].notna().any():
        ranked = day.dropna(subset=["market_cap"]).sort_values("market_cap", ascending=False)
        return ranked["tic"].astype(str).head(n).tolist()

    dollar = df.assign(dollar_volume=df["close"] * df["volume"])
    avg = dollar.groupby("tic")["dollar_volume"].mean().sort_values(ascending=False)
    return avg.head(n).index.astype(str).tolist()


def select_universe(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    rule: str | None = None,
) -> pd.DataFrame:
    """Filter a canonical frame to a fixed ticker set.

    Parameters
    ----------
    df
        Canonical OHLCV, shape ``(N, >=7)``.
    cfg
        Full YAML config (uses ``universe`` and ``dates``).
    rule
        Override ``cfg['universe']['rule']``.

    Returns
    -------
    pd.DataFrame
        Rows whose ``tic`` is in the selected set. Same columns as ``df``.
    """

    uni = cfg.get("universe", {})
    dates = cfg.get("dates", {})
    rule = (rule or uni.get("rule") or "sandbox").lower()
    start = dates.get("start")
    end = dates.get("end")
    scoped = _window(df, start, end)

    if rule == "sandbox":
        tickers = sandbox_tickers(cfg)
    elif rule in {"full_window", "full-window", "survivors"}:
        tickers = select_full_window(scoped)
    elif rule in {"top_n", "topn", "top-n"}:
        n = int(uni.get("top_n", 50))
        metric = str(uni.get("top_n_metric", "market_cap"))
        asof = dates.get("train_start", scoped["date"].min())
        tickers = select_top_n(scoped, n=n, asof=asof, metric=metric)
    else:
        raise ValueError(f"Unknown universe rule {rule!r}")

    tickers_u = {t.upper() for t in tickers}
    out = df.loc[df["tic"].astype(str).str.upper().isin(tickers_u)].copy()
    missing = tickers_u - set(out["tic"].astype(str).str.upper().unique())
    if missing and rule == "sandbox":
        raise ValueError(
            f"Sandbox tickers missing from the input file: {sorted(missing)}"
        )
    if out.empty:
        raise ValueError(f"Universe rule {rule!r} selected no tickers.")
    return out.sort_values(["tic", "date"]).reset_index(drop=True)


def list_rules() -> tuple[str, ...]:
    """Config names accepted by :func:`select_universe`."""

    return ("sandbox", "full_window", "top_n")
