"""Sandbox run of on-policy PPO on the 10-ticker POE demo portfolio.

Mirrors ``examples/FinRL_PortfolioOptimizationEnv_Demo.ipynb`` (TOP_BRL,
train 2011–2019, test 2020/2021/2022) but trains the PPO agent that keeps
the original seven PPO steps instead of Jiang policy gradient.

This is a sandbox: the original notebook trains EIIE for 40 episodes.
Default here is 3 episodes so the run finishes in this environment.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MaxAbsScaler

logging.getLogger("matplotlib.font_manager").disabled = True

TOP_BRL = [
    "VALE3.SA",
    "PETR4.SA",
    "ITUB4.SA",
    "BBDC4.SA",
    "BBAS3.SA",
    "RENT3.SA",
    "LREN3.SA",
    "PRIO3.SA",
    "WEGE3.SA",
    "ABEV3.SA",
]


def bootstrap_finrl_packages():
    """Load FinRL submodules without executing ``finrl/__init__.py``.

    That init file imports the train/trade stack (Alpaca, etc.), which is
    unrelated to this PPO-POE sandbox.
    """

    import sys
    import types

    root = Path(__file__).resolve().parents[1]
    packages = {
        "finrl": root / "finrl",
        "finrl.meta": root / "finrl/meta",
        "finrl.meta.preprocessor": root / "finrl/meta/preprocessor",
        "finrl.meta.env_portfolio_optimization": root
        / "finrl/meta/env_portfolio_optimization",
        "finrl.agents": root / "finrl/agents",
        "finrl.agents.portfolio_optimization": root
        / "finrl/agents/portfolio_optimization",
    }
    for name, path in packages.items():
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        module.__package__ = name
        sys.modules[name] = module


def quiet_quantstats_plots():
    """POE calls quantstats at every episode end; skip those plots in the sandbox."""

    import quantstats as qs

    qs.plots.snapshot = lambda *args, **kwargs: None


def make_env(df, cwd):
    from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import (
        PortfolioOptimizationEnv,
    )

    Path(cwd).mkdir(parents=True, exist_ok=True)
    return PortfolioOptimizationEnv(
        df,
        initial_amount=100000,
        comission_fee_pct=0.0025,
        time_window=50,
        features=["close", "high", "low"],
        normalize_df=None,
        cwd=str(cwd),
    )


def uniform_action(n_assets):
    return [0.0] + [1.0 / n_assets] * n_assets


def run_uniform_buy_and_hold(env, n_assets):
    terminated = False
    env.reset()
    while not terminated:
        _, _, terminated, _ = env.step(uniform_action(n_assets))
    return list(env._asset_memory["final"])


def plot_period(out_path, title, ubah, ppo_values):
    plt.figure(figsize=(10, 5))
    plt.plot(ubah, label="Buy and Hold")
    plt.plot(ppo_values, label="PPO")
    plt.title(title)
    plt.ylabel("Portfolio value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def fetch_top_brl(start_date="2011-01-01", end_date="2022-12-31") -> pd.DataFrame:
    """Yahoo download without the ``proxy`` kwarg FinRL's downloader still passes."""

    import yfinance as yf

    frames = []
    for tic in TOP_BRL:
        temp = yf.download(
            tic,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=True,
        )
        if temp.empty:
            raise RuntimeError(f"No Yahoo data for {tic}")
        if getattr(temp.columns, "nlevels", 1) != 1:
            temp.columns = temp.columns.droplevel(1)
        temp = temp.reset_index()
        temp["tic"] = tic
        frames.append(temp)
    data = pd.concat(frames, ignore_index=True)
    data = data.rename(
        columns={
            "Date": "date",
            "Close": "close",
            "High": "high",
            "Low": "low",
            "Open": "open",
            "Volume": "volume",
        }
    )
    data["date"] = pd.to_datetime(data["date"]).dt.strftime("%Y-%m-%d")
    keep = [c for c in ["date", "open", "high", "low", "close", "volume", "tic"] if c in data.columns]
    data = data[keep].dropna().sort_values(["date", "tic"]).reset_index(drop=True)
    print("Shape of DataFrame:", data.shape)
    return data


def load_or_download(cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        print(f"Loading cached prices from {cache_path}")
        return pd.read_csv(cache_path)

    print("Downloading TOP_BRL from Yahoo Finance (2011-01-01 to 2022-12-31)")
    df = fetch_top_brl()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def scale_by_ticker(df: pd.DataFrame) -> pd.DataFrame:
    try:
        from finrl.meta.preprocessor.preprocessors import GroupByScaler

        return GroupByScaler(by="tic", scaler=MaxAbsScaler).fit_transform(df)
    except Exception as exc:  # stockstats / finrl extras are optional here
        print(f"GroupByScaler unavailable ({exc}); scaling close/high/low per ticker")
        scaled = df.copy()
        for tic, group in df.groupby("tic"):
            scaler = MaxAbsScaler()
            cols = ["close", "high", "low"]
            scaled.loc[group.index, cols] = scaler.fit_transform(group[cols])
        return scaled


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/ppo_poe_sandbox",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    bootstrap_finrl_packages()
    quiet_quantstats_plots()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"device={device} episodes={args.episodes} tickers={len(TOP_BRL)}")

    raw = load_or_download(out_dir / "top_brl_2011_2022.csv")
    print(raw.head())
    print("tickers:", sorted(raw["tic"].unique().tolist()))
    print("rows:", len(raw), "dates:", raw["date"].nunique())
    if raw["tic"].nunique() != len(TOP_BRL):
        raise RuntimeError(
            f"Expected {len(TOP_BRL)} tickers, got {raw['tic'].nunique()}"
        )

    norm = scale_by_ticker(raw)
    df_portfolio = norm[["date", "tic", "close", "high", "low"]].copy()
    df_portfolio["date"] = pd.to_datetime(df_portfolio["date"]).dt.strftime("%Y-%m-%d")

    splits = {
        "train": df_portfolio[
            (df_portfolio["date"] >= "2011-01-01")
            & (df_portfolio["date"] < "2019-12-31")
        ],
        "2020": df_portfolio[
            (df_portfolio["date"] >= "2020-01-01")
            & (df_portfolio["date"] < "2020-12-31")
        ],
        "2021": df_portfolio[
            (df_portfolio["date"] >= "2021-01-01")
            & (df_portfolio["date"] < "2021-12-31")
        ],
        "2022": df_portfolio[
            (df_portfolio["date"] >= "2022-01-01")
            & (df_portfolio["date"] < "2022-12-31")
        ],
    }
    for name, frame in splits.items():
        print(f"{name}: {frame['date'].nunique()} dates, {len(frame)} rows")

    train_env = make_env(splits["train"], out_dir / "train")
    print(
        "POE train env: "
        f"action_dim={train_env.action_space.shape} "
        f"obs={train_env.observation_space.shape} "
        f"episode_length={train_env.episode_length}"
    )

    from finrl.agents.portfolio_optimization.models import DRLAgent

    model_kwargs = {
        "lr": 3e-4,
        "n_steps": args.n_steps,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "n_epochs": 4,
        "minibatch_size": 64,
    }
    policy_kwargs = {"hidden_sizes": (64, 64)}
    agent = DRLAgent(train_env)
    model = agent.get_model("ppo", device, model_kwargs, policy_kwargs)
    print("Training on-policy PPO (kept seven steps) on the 10-ticker POE sandbox")
    DRLAgent.train_model(model, episodes=args.episodes)
    torch.save(model.actor_critic.state_dict(), out_dir / "policy_ppo.pt")

    ppo_results = {
        "train_last_episode": list(train_env._asset_memory["final"]),
    }
    summary_rows = [
        {
            "period": "train_last_episode",
            "strategy": "PPO (exploring)",
            "final_value": ppo_results["train_last_episode"][-1],
        }
    ]

    # Greedy evaluation on train + held-out years (on-policy test, no extra learning)
    eval_splits = {"train": splits["train"], "2020": splits["2020"], "2021": splits["2021"], "2022": splits["2022"]}
    test_envs = {}
    for period, frame in eval_splits.items():
        env = make_env(frame, out_dir / f"eval_{period}")
        test_envs[period] = env
        print(f"Validating greedy PPO on {period}")
        DRLAgent.DRL_validation(model, env)
        ppo_results[period] = list(env._asset_memory["final"])
        summary_rows.append(
            {
                "period": period,
                "strategy": "PPO",
                "final_value": ppo_results[period][-1],
            }
        )

    ubah = {}
    ubah_train_env = make_env(splits["train"], out_dir / "ubah_train")
    ubah["train"] = run_uniform_buy_and_hold(ubah_train_env, len(TOP_BRL))
    summary_rows.append(
        {
            "period": "train",
            "strategy": "UBAH",
            "final_value": ubah["train"][-1],
        }
    )
    for period in ("2020", "2021", "2022"):
        env = make_env(splits[period], out_dir / f"ubah_{period}")
        ubah[period] = run_uniform_buy_and_hold(env, len(TOP_BRL))
        summary_rows.append(
            {
                "period": period,
                "strategy": "UBAH",
                "final_value": ubah[period][-1],
            }
        )

    plot_period(
        out_dir / "ppo_vs_ubah_train.png",
        "PPO vs Buy and Hold — training (2011-2019)",
        ubah["train"],
        ppo_results["train"],
    )
    for period in ("2020", "2021", "2022"):
        plot_period(
            out_dir / f"ppo_vs_ubah_{period}.png",
            f"PPO vs Buy and Hold — {period}",
            ubah[period],
            ppo_results[period],
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote plots and summary under {out_dir.resolve()}")


if __name__ == "__main__":
    main()
