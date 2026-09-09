"""Build the balanced long-format panel POE consumes.

POE contract (see ``docs/poe_contract.md``): columns ``date``, ``tic``, then
feature columns; every date has every ticker; no NaNs after warm-up; ``date``
is ``datetime64``; rows sorted.

Missing days are handled by an explicit policy (never silent):

* ``drop_ticker`` — drop any ticker that does not appear on every date
* ``ffill`` — forward-fill within each ticker up to ``ffill_max_gap``
  consecutive missing sessions; tickers that still have holes are dropped

Input
-----
Canonical (or featured) long frame: ``date, tic, ...``. Shape ``(N, k)``.

Output
------
Balanced panel, columns ``date, tic, <features...>``. Shape ``(D * n, 2 + f)``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


class PanelError(ValueError):
    """The frame cannot be made into a balanced POE panel under the policy."""


def assert_balanced_panel(
    df: pd.DataFrame,
    feature_cols: Sequence[str] | None = None,
) -> None:
    """Raise ``PanelError`` unless the POE panel invariants hold.

    Parameters
    ----------
    df
        Candidate panel, shape ``(D * n, >=2)``.
    feature_cols
        Columns that must be NaN-free. Defaults to all columns except
        ``date`` and ``tic``.
    """

    if df.empty:
        raise PanelError("panel is empty")
    if "date" not in df.columns or "tic" not in df.columns:
        raise PanelError("panel must have 'date' and 'tic' columns")
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise PanelError(f"'date' must be datetime64, got {df['date'].dtype}")

    n_tic = df["tic"].nunique()
    per_date = df.groupby("date")["tic"].nunique()
    if (per_date != n_tic).any():
        bad = per_date[per_date != n_tic]
        raise PanelError(
            f"unbalanced panel: {len(bad)} dates do not have all {n_tic} tickers"
        )
    if df.duplicated(["date", "tic"]).any():
        raise PanelError("duplicate (date, tic) in panel")

    ordered = df.sort_values(["date", "tic"])
    if not ordered["date"].is_monotonic_increasing:
        raise PanelError("dates are not sorted")

    # Contiguous in the balanced sense: every ticker shares the same date index.
    date_sets = df.groupby("tic")["date"].apply(lambda s: tuple(s.sort_values()))
    if date_sets.nunique() != 1:
        raise PanelError("tickers do not share a common contiguous date index")

    cols = list(feature_cols) if feature_cols is not None else [
        c for c in df.columns if c not in {"date", "tic"}
    ]
    if cols and df[cols].isna().any().any():
        n = int(df[cols].isna().any(axis=1).sum())
        raise PanelError(f"NaNs remain in feature columns after warm-up ({n} rows)")


def _calendar(df: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(df["date"].unique()))


def _apply_missing_policy(
    df: pd.DataFrame,
    policy: str,
    ffill_max_gap: int,
) -> pd.DataFrame:
    """Return a subset/filled frame that can be balanced.

    Input/output columns are whatever ``df`` has. Shape may shrink.
    """

    policy = policy.lower()
    dates = _calendar(df)
    tickers = sorted(df["tic"].astype(str).unique())
    if policy == "drop_ticker":
        keep: list[str] = []
        for tic in tickers:
            have = set(pd.to_datetime(df.loc[df["tic"] == tic, "date"]))
            if have == set(dates):
                keep.append(tic)
        dropped = sorted(set(tickers) - set(keep))
        if dropped:
            print(f"[panel] drop_ticker removed {len(dropped)} names: {dropped[:12]}...")
        if not keep:
            raise PanelError("drop_ticker removed every ticker")
        return df.loc[df["tic"].astype(str).isin(keep)].copy()

    if policy == "ffill":
        frames: list[pd.DataFrame] = []
        dropped: list[str] = []
        for tic in tickers:
            g = (
                df.loc[df["tic"] == tic]
                .set_index("date")
                .sort_index()
                .reindex(dates)
            )
            missing = g.drop(columns=["tic"], errors="ignore").isna().all(axis=1)
            # gap lengths: consecutive True runs
            too_long = False
            run = 0
            for flag in missing.tolist():
                run = run + 1 if flag else 0
                if run > int(ffill_max_gap):
                    too_long = True
                    break
            if too_long:
                dropped.append(tic)
                continue
            g = g.ffill(limit=int(ffill_max_gap))
            g["tic"] = tic
            g = g.dropna(how="any")
            have = set(g.index)
            if have != set(dates):
                dropped.append(tic)
                continue
            frames.append(g.reset_index())
        if dropped:
            print(f"[panel] ffill dropped {len(dropped)} names over max_gap={ffill_max_gap}")
        if not frames:
            raise PanelError("ffill policy produced an empty panel")
        return pd.concat(frames, ignore_index=True)

    raise PanelError(f"unknown missing-day policy {policy!r} (use drop_ticker|ffill)")


def build_panel(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    missing_policy: str = "drop_ticker",
    ffill_max_gap: int = 5,
    dropna_warmup: bool = True,
) -> pd.DataFrame:
    """Assemble a POE-ready balanced panel.

    Parameters
    ----------
    df
        Featured long frame. Must contain ``date``, ``tic``, and
        ``feature_cols``. Shape ``(N, k)``.
    feature_cols
        Columns passed to POE as ``features`` (e.g. close/high/low/macd/...).
    missing_policy
        ``drop_ticker`` or ``ffill``.
    ffill_max_gap
        Max consecutive sessions to fill when ``missing_policy='ffill'``.
    dropna_warmup
        Drop rows with NaN in ``feature_cols`` (indicator warm-up), then
        re-intersect dates across tickers.

    Returns
    -------
    pd.DataFrame
        Columns ``['date', 'tic', *feature_cols]`` (plus any extra columns
        that survived), sorted by ``date``, ``tic``. ``date`` is datetime64.
    """

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["tic"] = out["tic"].astype(str)
    missing = [c for c in feature_cols if c not in out.columns]
    if missing:
        raise PanelError(f"feature columns not in frame: {missing}")

    out = _apply_missing_policy(out, missing_policy, ffill_max_gap)

    if dropna_warmup:
        out = out.dropna(subset=list(feature_cols))
        # Re-balance: keep dates present for every remaining ticker.
        n_tic = out["tic"].nunique()
        counts = out.groupby("date")["tic"].nunique()
        good_dates = counts[counts == n_tic].index
        out = out.loc[out["date"].isin(good_dates)]

    cols = ["date", "tic", *list(feature_cols)]
    extra = [c for c in out.columns if c not in cols]
    out = out[cols + extra].sort_values(["date", "tic"]).reset_index(drop=True)
    assert_balanced_panel(out, feature_cols=feature_cols)
    return out


def panel_from_config(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """``build_panel`` using ``cfg['poe']['features']`` and ``cfg['missing_days']``."""

    features = list(cfg.get("poe", {}).get("features", ["close", "high", "low"]))
    missing = cfg.get("missing_days", {})
    return build_panel(
        df,
        feature_cols=features,
        missing_policy=str(missing.get("policy", "drop_ticker")),
        ffill_max_gap=int(missing.get("ffill_max_gap", 5)),
    )
