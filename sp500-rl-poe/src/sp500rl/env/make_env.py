"""Factory for a configured FinRL ``PortfolioOptimizationEnv``.

``make_poe(df, cfg, mode="train"|"test")`` slices dates from the YAML config
(never from hard-coded notebook cells) and passes the POE kwargs listed in
``docs/poe_contract.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pandas as pd

from sp500rl.config import project_root
from sp500rl.env.gymnasium_wrapper import GymnasiumPOE
from sp500rl.finrl_bootstrap import import_poe


def _slice_mode(df: pd.DataFrame, cfg: dict[str, Any], mode: str) -> pd.DataFrame:
    dates = cfg.get("dates", {})
    mode = mode.lower()
    if mode == "train":
        start, end = dates["train_start"], dates["train_end"]
    elif mode == "test":
        start, end = dates["test_start"], dates["test_end"]
    elif mode == "all":
        start = dates.get("start", df["date"].min())
        end = dates.get("end", df["date"].max())
    else:
        raise ValueError(f"mode must be train|test|all, got {mode!r}")
    out = df.loc[
        (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
    ].copy()
    if out.empty:
        raise ValueError(f"No rows for mode={mode} in [{start}, {end}]")
    return out


def make_poe(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    mode: Literal["train", "test", "all"] = "train",
    *,
    wrap_gymnasium: bool = False,
    return_last_action: bool | None = None,
    new_gym_api: bool | None = None,
    cwd: str | Path | None = None,
):
    """Build a ``PortfolioOptimizationEnv``.

    Parameters
    ----------
    df
        POE panel: ``date, tic, <features>``. Shape ``(D * n, 2 + f)``.
    cfg
        YAML config; reads ``poe.*`` and ``dates.*``.
    mode
        ``train`` / ``test`` / ``all`` date slice.
    wrap_gymnasium
        If True, wrap with :class:`GymnasiumPOE` for SB3 2.x.
    return_last_action
        Override config. **False** for FinRL ``PolicyGradient`` (Box obs);
        **True** for SB3 ``MultiInputPolicy`` (Dict obs).
    new_gym_api
        Override config. False for PolicyGradient; True when wrapping SB3.
    cwd
        Directory where POE writes ``results/rl/*.png``.

    Returns
    -------
    PortfolioOptimizationEnv or GymnasiumPOE
        Observation is ``(f, n, t)`` or
        ``{"state": (f, n, t), "last_action": (n+1,)}``.
        Action is ``(n+1,)`` cash + assets.
    """

    poe_cfg = dict(cfg.get("poe", {}))
    sliced = _slice_mode(df, cfg, mode)

    if return_last_action is None:
        return_last_action = bool(poe_cfg.get("return_last_action", True))
    if new_gym_api is None:
        new_gym_api = bool(poe_cfg.get("new_gym_api", False)) or wrap_gymnasium

    results_cwd = Path(cwd) if cwd is not None else project_root() / poe_cfg.get("cwd", "results")
    results_cwd.mkdir(parents=True, exist_ok=True)

    normalize = poe_cfg.get("normalize_df", "by_previous_time")
    if normalize in {"null", "None", None}:
        normalize = None

    PortfolioOptimizationEnv = import_poe()
    env = PortfolioOptimizationEnv(
        df=sliced,
        initial_amount=float(poe_cfg.get("initial_amount", 1_000_000)),
        order_df=True,
        return_last_action=return_last_action,
        normalize_df=normalize,
        reward_scaling=float(poe_cfg.get("reward_scaling", 1)),
        comission_fee_model=poe_cfg.get("comission_fee_model", "trf"),
        comission_fee_pct=float(poe_cfg.get("comission_fee_pct", 0.0)),
        features=list(poe_cfg.get("features", ["close", "high", "low"])),
        valuation_feature=poe_cfg.get("valuation_feature", "close"),
        time_column="date",
        time_format="%Y-%m-%d",
        tic_column="tic",
        tics_in_portfolio="all",
        time_window=int(poe_cfg.get("time_window", 50)),
        cwd=str(results_cwd),
        new_gym_api=new_gym_api,
    )
    if wrap_gymnasium:
        return GymnasiumPOE(env)
    return env
