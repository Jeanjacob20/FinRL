"""Portfolio metrics implemented here (no pyfolio).

Inputs are the arrays returned by a POE rollout: portfolio values, simple
returns, submitted weights, and TRF factors.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


def sharpe(returns: np.ndarray, periods: int = 252, rf: float = 0.0) -> float:
    """Annualised Sharpe of simple returns. Shape ``(T,)``."""

    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    excess = r - rf / periods
    std = excess.std(ddof=1)
    if std == 0:
        return float("nan")
    return float(np.sqrt(periods) * excess.mean() / std)


def max_drawdown(values: np.ndarray) -> float:
    """Maximum drawdown of a value path, in ``(-1, 0]``. Shape ``(T,)``."""

    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    peak = np.maximum.accumulate(v)
    dd = v / np.maximum(peak, 1e-12) - 1.0
    return float(dd.min())


def turnover(actions: np.ndarray) -> float:
    """Mean one-way turnover ``0.5 * |w_t - w_{t-1}|_1``.

    Parameters
    ----------
    actions
        Submitted weights, shape ``(T, n+1)``.
    """

    w = np.asarray(actions, dtype=np.float64)
    if w.ndim != 2 or len(w) < 2:
        return 0.0
    delta = np.abs(np.diff(w, axis=0)).sum(axis=1) * 0.5
    return float(delta.mean())


def total_cost_paid(values: np.ndarray, trf_mu: np.ndarray) -> float:
    """Dollar drag implied by TRF: ``sum(V_before * (1 - mu))``.

    ``values`` is POE ``_asset_memory['final']`` (includes t0). ``trf_mu`` is
    one factor per env step (length ``T-1`` or ``T``). We align to the last
    ``len(trf_mu)`` post-trade values.
    """

    v = np.asarray(values, dtype=np.float64)
    mu = np.asarray(trf_mu, dtype=np.float64)
    if mu.size == 0 or v.size == 0:
        return 0.0
    # After reset, first value is initial cash. Each step multiplies by mu
    # *before* the price move. Approximate cost at step t as
    # previous_final * (1-mu).
    prev = v[: mu.size]
    if prev.size != mu.size:
        prev = v[-(mu.size + 1) : -1] if v.size > mu.size else v[: mu.size]
        if prev.size != mu.size:
            prev = np.resize(prev, mu.size)
    return float(np.sum(prev * (1.0 - mu)))


def cost_drag(trf_mu: np.ndarray) -> float:
    """Cumulative remaining-value factor ``prod(mu) - 1`` (negative is drag)."""

    mu = np.asarray(trf_mu, dtype=np.float64)
    mu = mu[np.isfinite(mu)]
    if mu.size == 0:
        return 0.0
    return float(np.prod(mu) - 1.0)


def summarise_rollout(result: Mapping[str, Any], name: str = "strategy") -> pd.Series:
    """One-row summary: Sharpe, max DD, turnover, cost drag, terminal value."""

    values = np.asarray(result["values"], dtype=np.float64)
    returns = np.asarray(result["returns"], dtype=np.float64)
    actions = np.asarray(result["actions"], dtype=np.float64)
    trf = np.asarray(result.get("trf_mu", []), dtype=np.float64)
    return pd.Series(
        {
            "name": name,
            "terminal_value": float(values[-1]) if values.size else float("nan"),
            "cumulative_return": float(values[-1] / values[0] - 1.0) if values.size > 1 else float("nan"),
            "sharpe": sharpe(returns),
            "max_drawdown": max_drawdown(values),
            "turnover": turnover(actions),
            "cost_drag": cost_drag(trf),
            "total_cost_paid": total_cost_paid(values, trf),
        }
    )


def compare_rollouts(rollouts: dict[str, Mapping[str, Any]]) -> pd.DataFrame:
    """Stack :func:`summarise_rollout` for several named strategies."""

    rows = [summarise_rollout(v, name=k) for k, v in rollouts.items()]
    return pd.DataFrame(rows).set_index("name")
