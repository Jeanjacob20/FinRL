"""Fixed-universe selection on a canonical price frame.

POE requires the *same ticker set on every date* (balanced panel). That
conflicts with a point-in-time S&P 500. Every rule below except a named list
is therefore survivorship-biased. Report the bias.

Rules (config ``universe.rule``)
--------------------------------
<name of a list in ``universe.lists``>
    Fixed tickers, e.g. ``ab_finrl`` (the 10 names selected in AB_finRL) or
    ``sandbox``.
full_window
    Tickers present on **every** date in ``[start, end]``.
top_n
    Top-N by ``market_cap`` on the first training date if that column exists,
    else top-N by average dollar volume over the window. Held fixed after that.
listed
    Names in the dataset's ``tickers`` file that also appear in the price
    frame, then the full-window survivors.

Input
-----
Canonical long-format frame: ``date, tic, open, high, low, close, volume``
(+ optional ``market_cap``). Shape ``(N, >=7)``.

Output
------
The same columns, filtered to the chosen tickers. Shape ``(M, >=7)``, ``M <= N``.
"""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

# The 10 names selected in AB_finRL. Also the fallback when YAML has no lists.
AB_FINRL_TICKERS: tuple[str, ...] = (
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
DEFAULT_SANDBOX_TICKERS: tuple[str, ...] = AB_FINRL_TICKERS

BUILTIN_RULES: tuple[str, ...] = ("full_window", "top_n", "listed")


def _window(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    out = df
    if start is not None:
        out = out.loc[out["date"] >= pd.Timestamp(start)]
    if end is not None:
        out = out.loc[out["date"] <= pd.Timestamp(end)]
    return out


def ticker_lists(cfg: dict[str, Any] | None) -> dict[str, list[str]]:
    """Named ticker lists from ``universe.lists`` (upper-cased)."""

    lists: dict[str, list[str]] = {
        "ab_finrl": list(AB_FINRL_TICKERS),
        "sandbox": list(DEFAULT_SANDBOX_TICKERS),
    }
    if cfg:
        raw = cfg.get("universe", {}).get("lists") or {}
        for name, tickers in raw.items():
            lists[str(name).lower()] = [str(t).upper() for t in tickers]
    return lists


def list_tickers(name: str, cfg: dict[str, Any] | None = None) -> list[str]:
    """Tickers of one named list (``ab_finrl``, ``sandbox``, ...)."""

    lists = ticker_lists(cfg)
    key = name.lower()
    if key not in lists:
        raise KeyError(f"No ticker list {name!r}. Known: {sorted(lists)}")
    return list(lists[key])


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
    listed: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Filter a canonical frame to a fixed ticker set.

    Parameters
    ----------
    df
        Canonical OHLCV, shape ``(N, >=7)``.
    cfg
        Full YAML config (uses ``universe`` and ``dates``).
    rule
        Override ``cfg['universe']['rule']``: a list name, ``full_window``,
        ``top_n`` or ``listed``.
    listed
        Tickers from the dataset's ``tickers`` file; required for ``listed``.

    Returns
    -------
    pd.DataFrame
        Rows whose ``tic`` is in the selected set. Same columns as ``df``.
    """

    uni = cfg.get("universe", {})
    dates = cfg.get("dates", {})
    rule = (rule or uni.get("rule") or "ab_finrl").lower()
    scoped = _window(df, dates.get("start"), dates.get("end"))
    lists = ticker_lists(cfg)

    if rule in lists:
        tickers = lists[rule]
    elif rule in {"full_window", "full-window", "survivors"}:
        tickers = select_full_window(scoped)
    elif rule in {"top_n", "topn", "top-n"}:
        n = int(uni.get("top_n", 50))
        metric = str(uni.get("top_n_metric", "market_cap"))
        asof = dates.get("train_start", scoped["date"].min())
        tickers = select_top_n(scoped, n=n, asof=asof, metric=metric)
    elif rule in {"listed", "wrds_tickers"}:
        if listed is None:
            raise ValueError(
                "Universe rule 'listed' needs the dataset's tickers file "
                "(datasets.yaml files.tickers)."
            )
        wanted = {str(t).upper() for t in listed}
        survivors = scoped.loc[scoped["tic"].astype(str).str.upper().isin(wanted)]
        tickers = select_full_window(survivors)
    else:
        raise ValueError(
            f"Unknown universe rule {rule!r}. Lists: {sorted(lists)}; rules: {BUILTIN_RULES}"
        )

    tickers_u = {t.upper() for t in tickers}
    out = df.loc[df["tic"].astype(str).str.upper().isin(tickers_u)].copy()
    missing = tickers_u - set(out["tic"].astype(str).str.upper().unique())
    if missing and rule in lists:
        raise ValueError(f"Universe list {rule!r}: tickers missing from the dataset: {sorted(missing)}")
    if out.empty:
        raise ValueError(f"Universe rule {rule!r} selected no tickers.")
    return out.sort_values(["tic", "date"]).reset_index(drop=True)


def list_rules(cfg: dict[str, Any] | None = None) -> tuple[str, ...]:
    """Names accepted by :func:`select_universe`: the lists plus built-in rules."""

    return tuple(sorted(ticker_lists(cfg))) + BUILTIN_RULES
