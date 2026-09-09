"""Simple baselines that emit POE action vectors (cash + n assets, sum to 1).

Each policy is rolled through a **fresh** ``PortfolioOptimizationEnv`` built
with the same kwargs as the RL agent so TRF costs are applied identically.

Actions
-------
Shape ``(n + 1,)``: index 0 is cash, then one weight per ticker in POE's
``_tic_list`` order.

Buy-and-hold submits equal-weight once, then resubmits the drifted holdings
(``weights * price_variation``, renormalised) so TRF cost is ~0 after t0.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd


def equal_weight_action(n_assets: int, cash: float = 0.0) -> np.ndarray:
    """Equal weight on the assets, optional cash sleeve.

    Returns
    -------
    np.ndarray
        Shape ``(n_assets + 1,)``, sums to 1.
    """

    if not 0.0 <= cash < 1.0:
        raise ValueError("cash must be in [0, 1)")
    w = np.zeros(n_assets + 1, dtype=np.float64)
    w[0] = cash
    w[1:] = (1.0 - cash) / n_assets
    return w


def inverse_vol_weights(returns: pd.DataFrame, min_periods: int = 20) -> pd.DataFrame:
    """Expanding-window inverse-volatility (risk-parity) weights.

    Parameters
    ----------
    returns
        Wide log-return frame, index ``date``, columns tickers. Shape ``(D, n)``.
        Only past data at each date is used (``shift(1)`` then expanding std).

    Returns
    -------
    pd.DataFrame
        Same shape as ``returns``, rows sum to 1. Early rows may be equal-weight.
    """

    past = returns.shift(1)
    vol = past.expanding(min_periods=min_periods).std().replace(0.0, np.nan)
    inv = 1.0 / vol
    inv = inv.div(inv.sum(axis=1), axis=0)
    eq = pd.DataFrame(1.0 / returns.shape[1], index=returns.index, columns=returns.columns)
    return inv.fillna(eq)


def _unpack_reset(env) -> Any:
    out = env.reset()
    if isinstance(out, tuple):
        return out[0]
    return out


def _unpack_step(env, action) -> tuple[Any, float, bool, dict]:
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, float(reward), bool(terminated or truncated), info
    obs, reward, done, info = out
    return obs, float(reward), bool(done), info


def rollout_pg_policy(env, policy) -> dict[str, Any]:
    """Roll a FinRL EIIE-style ``policy(obs_batch, last_action_batch)``.

    ``last_action`` tracks submitted weights (cash-first), matching PolicyGradient's PVM.
    """

    last = {"w": np.array([1.0] + [0.0] * env.portfolio_size, dtype=np.float64)}

    def fn(obs, info, t):
        obs_b = np.expand_dims(np.asarray(obs), axis=0)
        last_b = np.expand_dims(last["w"], axis=0)
        action = np.asarray(policy(obs_b, last_b), dtype=np.float64).reshape(-1)
        last["w"] = action
        return action

    return rollout(env, fn)


def rollout(env, action_fn: Callable[[Any, dict, int], np.ndarray]) -> dict[str, Any]:
    """Step ``env`` until done.

    ``action_fn(obs, info, t) -> weights`` must return a simplex vector
    of shape ``(n+1,)``.

    Returns
    -------
    dict
        ``dates``, ``values``, ``returns``, ``actions``, ``trf_mu``,
        ``rewards``. Values include the initial cash level.
    """

    obs = _unpack_reset(env)
    info: dict = {}
    done = False
    t = 0
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    trf: list[float] = []
    dates = [getattr(env, "_date_memory", [None])[0]]
    while not done:
        action = np.asarray(action_fn(obs, info, t), dtype=np.float64)
        obs, reward, done, info = _unpack_step(env, action)
        actions.append(action)
        rewards.append(reward)
        trf.append(float(info.get("trf_mu", 1.0)) if info else 1.0)
        if info and "end_time" in info:
            dates.append(info["end_time"])
        t += 1

    values = np.asarray(env._asset_memory["final"], dtype=np.float64)
    rets = np.asarray(env._portfolio_return_memory, dtype=np.float64)
    return {
        "dates": dates,
        "values": values,
        "returns": rets,
        "actions": np.vstack(actions) if actions else np.zeros((0, env.portfolio_size + 1)),
        "trf_mu": np.asarray(trf, dtype=np.float64),
        "rewards": np.asarray(rewards, dtype=np.float64),
    }


def rollout_equal_weight(env) -> dict[str, Any]:
    """Always rebalance to equal asset weights (zero cash)."""

    action = equal_weight_action(env.portfolio_size)
    return rollout(env, lambda obs, info, t: action)


def rollout_buy_and_hold(env) -> dict[str, Any]:
    """Equal-weight at t0, then hold (resubmit drifted weights)."""

    n = env.portfolio_size
    weights = {"w": equal_weight_action(n)}

    def fn(obs, info, t):
        if t == 0 or not info:
            return weights["w"]
        pv = np.asarray(info["price_variation"], dtype=np.float64)
        drifted = weights["w"] * pv
        drifted = drifted / drifted.sum()
        weights["w"] = drifted
        return drifted

    return rollout(env, fn)


def rollout_risk_parity(env, panel: pd.DataFrame, min_periods: int = 20) -> dict[str, Any]:
    """Inverse-vol weights looked up by POE ``end_time``, cash = 0.

    Parameters
    ----------
    env
        A POE instance (same costs as the agent).
    panel
        Featured panel with ``date``, ``tic``, ``close`` (or ``log_return``).
    """

    tic_order = list(env._tic_list)
    wide = panel.pivot(index="date", columns="tic", values="close")[tic_order]
    rets = np.log(wide / wide.shift(1))
    w_assets = inverse_vol_weights(rets, min_periods=min_periods)

    def fn(obs, info, t):
        ts = info.get("end_time") if info else None
        if ts is None or ts not in w_assets.index:
            return equal_weight_action(env.portfolio_size)
        row = w_assets.loc[ts].to_numpy(dtype=np.float64)
        action = np.concatenate([[0.0], row])
        action = np.nan_to_num(action, nan=0.0)
        s = action.sum()
        if s <= 0:
            return equal_weight_action(env.portfolio_size)
        return action / s

    return rollout(env, fn)
