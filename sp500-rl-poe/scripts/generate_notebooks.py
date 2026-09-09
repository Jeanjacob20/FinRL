#!/usr/bin/env python3
"""Generate the five experiment notebooks (valid nbformat v4)."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"

PREAMBLE = """\
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("..").resolve()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sp500rl.config import load_config
from sp500rl.seed import set_seed

CFG = load_config(ROOT / "configs" / "default.yaml")
SEED = set_seed(int(CFG["seed"]))
DATASET = CFG["dataset"]            # configs/datasets.yaml entry (synthetic | ab_wrds | yahoo | ...)
UNIVERSE = CFG["universe"]["rule"]  # ab_finrl | sandbox | full_window | top_n | listed
print(f"seed={SEED} dataset={DATASET} universe={UNIVERSE}")
print("train", CFG["dates"]["train_start"], "→", CFG["dates"]["train_end"])
print("test ", CFG["dates"]["test_start"], "→", CFG["dates"]["test_end"])
"""


def md(source: str):
    return nbf.v4.new_markdown_cell(source)


def code(source: str):
    return nbf.v4.new_code_cell(source)


def write(name: str, cells: list) -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nb["cells"] = cells
    path = NB / name
    nbf.write(nb, path)
    print("wrote", path)


def nb00() -> None:
    cells = [
        md("# 00 — Data: what / how / where / use\n\nEvery dataset is one entry in `configs/datasets.yaml`. This notebook walks the four questions and ends with a POE-ready panel."),
        code(PREAMBLE),
        md("## 1. What datasets exist?"),
        code("""\
from sp500rl.data import list_datasets, status, fetch, load_prices, get_panel, DatasetNotReady

for ds in list_datasets():
    print(f"{ds.name:<10} how={ds.how:<9} format={ds.format:<9} {ds.what}")
"""),
        md("## 2. Where do the files go, and are they there?"),
        code("""\
for name, info in status().items():
    print(name, "ready" if info["ready"] else "MISSING")
    for role, f in info["files"].items():
        print("   ", "ok " if f["exists"] else "-- ", role, f["path"])
"""),
        md("## 3. How do I retrieve one?\n\n`fetch` generates / downloads when it can. A manual dataset (the AB_finRL extract) raises with the instruction and the exact paths."),
        code("""\
written = fetch(DATASET, CFG)
print("files:", {k: str(v) for k, v in written.items()})

try:
    fetch("ab_wrds", CFG)
except DatasetNotReady as exc:
    print(exc)
"""),
        md("## 4. Where do I use it?\n\n`load_prices` → canonical `date, tic, open, high, low, close, volume`. `get_panel` → universe → features → balanced POE panel, cached in `data/processed/<dataset>__<universe>.parquet`."),
        code("""\
import matplotlib.pyplot as plt
from sp500rl.data.schema import validate
from sp500rl.data.panel import assert_balanced_panel

prices = load_prices(DATASET, CFG)
print(validate(prices).summary())
print("tickers", sorted(prices["tic"].unique()))

panel = get_panel(DATASET, UNIVERSE, CFG, refresh=True)
assert_balanced_panel(panel, feature_cols=CFG["poe"]["features"])
print("panel", panel.shape, panel["date"].min().date(), "→", panel["date"].max().date())

pivot = panel.pivot(index="date", columns="tic", values="close")
ax = pivot.iloc[:, :4].plot(figsize=(10, 4), title=f"{DATASET} close (first 4 tickers)")
ax.set_ylabel("close")
plt.tight_layout()
plt.show()
"""),
        md("## Same code, other datasets\n\nSwitch `dataset:` in `configs/default.yaml` (or pass a name). Below: the AB_finRL-shaped path with fake files so it runs before the real CSVs exist."),
        code("""\
fetch("ab_wrds", CFG, fake=True)   # writes CRSP-shaped wrds_processed / wrds_tickers
ab_panel = get_panel("ab_wrds", "ab_finrl", CFG, refresh=True)
print("ab_wrds panel", ab_panel.shape, sorted(ab_panel["tic"].unique()))
assert set(ab_panel["tic"]) == set(panel["tic"])
"""),
    ]
    write("00_data_pipeline_check.ipynb", cells)


TRAIN_EVAL = """\
from sp500rl.baselines.simple import rollout, rollout_buy_and_hold, rollout_equal_weight, rollout_risk_parity
from sp500rl.eval.metrics import compare_rollouts
from sp500rl.env.make_env import make_poe
import matplotlib.pyplot as plt
import pandas as pd
"""


def nb01() -> None:
    cells = [
        md("# 01 — 10 stocks, FinRL EIIE / PolicyGradient\n\nTrain on the configured dataset + universe (AB_finRL 10 names by default), evaluate on the test window, plot cumulative value vs equal-weight and risk parity."),
        code(PREAMBLE),
        code("""\
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sp500rl.data import get_panel
from sp500rl.env.make_env import make_poe
from sp500rl.finrl_bootstrap import import_pg_stack
from sp500rl.baselines.simple import (
    rollout_equal_weight,
    rollout_pg_policy,
    rollout_risk_parity,
)
from sp500rl.eval.metrics import compare_rollouts

panel = get_panel(DATASET, UNIVERSE, CFG)
print("panel", panel.shape, panel["date"].min().date(), "→", panel["date"].max().date())
print("tickers", sorted(panel["tic"].unique()))
"""),
        md("PolicyGradient treats `obs` as a tensor and uses the old gym 4-tuple, so `return_last_action=False` and `new_gym_api=False`. Test-window evaluation uses the frozen `train_policy` (no online learning) rolled through a fresh POE copy so TRF costs match the baselines."),
        code("""\
DRLAgent, EIIE, GPM, PolicyGradient = import_pg_stack()
print("imported", DRLAgent, EIIE, GPM, PolicyGradient)

features = list(CFG["poe"]["features"])
time_window = int(CFG["poe"]["time_window"])
train_env = make_poe(
    panel, CFG, mode="train",
    return_last_action=False, new_gym_api=False,
    cwd=ROOT / "results" / "eiie_train",
)
print("train obs shape", train_env.reset().shape, "action", train_env.action_space.shape)

agent = DRLAgent(train_env)
model = agent.get_model(
    "pg",
    device="cpu",
    model_kwargs={
        "policy": EIIE,
        "lr": 1e-3,
        "batch_size": 64,
    },
    policy_kwargs={
        "initial_features": len(features),
        "k_size": 3,
        "time_window": time_window,
        "device": "cpu",
    },
)
EPISODES = 2  # bump for research runs
print(f"training EIIE for {EPISODES} episodes, seed={SEED}")
DRLAgent.train_model(model, episodes=EPISODES)
"""),
        code("""\
test_kw = dict(mode="test", return_last_action=False, new_gym_api=False)
agent_env = make_poe(panel, CFG, cwd=ROOT / "results" / "eiie_agent", **test_kw)
eq_env = make_poe(panel, CFG, cwd=ROOT / "results" / "eiie_eq", **test_kw)
rp_env = make_poe(panel, CFG, cwd=ROOT / "results" / "eiie_rp", **test_kw)

agent_roll = rollout_pg_policy(agent_env, model.train_policy)
eq_roll = rollout_equal_weight(eq_env)
rp_roll = rollout_risk_parity(rp_env, panel)

summary = compare_rollouts(
    {"EIIE": agent_roll, "equal_weight": eq_roll, "risk_parity": rp_roll}
)
print(summary.to_string())

def _series(roll, name):
    n = min(len(roll["dates"]), len(roll["values"]))
    return pd.Series(roll["values"][:n], index=pd.to_datetime(roll["dates"][:n]), name=name)

fig, ax = plt.subplots(figsize=(10, 4))
for roll, name in ((agent_roll, "EIIE"), (eq_roll, "equal-weight"), (rp_roll, "risk-parity")):
    _series(roll, name).plot(ax=ax)
ax.set_title("Test-window portfolio value")
ax.set_ylabel("value")
plt.tight_layout()
plt.show()
"""),
    ]
    write("01_sandbox_10_stocks_EIIE.ipynb", cells)


def nb_sb3(filename: str, algo: str, title: str) -> None:
    cells = [
        md(f"# {title}\n\nSame protocol as notebook 01: configured dataset + universe, YAML dates, custom `(f,n,t)` extractor. Default `MlpPolicy`/`CnnPolicy` are not used."),
        code(PREAMBLE),
        code("""\
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC

from sp500rl.data import get_panel
from sp500rl.env.extractors import sb3_policy_kwargs
from sp500rl.env.make_env import make_poe
from sp500rl.baselines.simple import rollout, rollout_equal_weight, rollout_risk_parity
from sp500rl.eval.metrics import compare_rollouts

panel = get_panel(DATASET, UNIVERSE, CFG)
print("panel", panel.shape, "tickers", sorted(panel["tic"].unique()))
"""),
        code(f"""\
ALGO = "{algo}"
print(f"SB3 {{ALGO}} seed={{SEED}}")

train_env = make_poe(
    panel, CFG, mode="train",
    wrap_gymnasium=True, return_last_action=True, new_gym_api=True,
    cwd=ROOT / "results" / f"sb3_{{ALGO.lower()}}_train",
)
obs, info = train_env.reset(seed=SEED)
print("obs keys", obs.keys(), "state", obs["state"].shape, "last_action", obs["last_action"].shape)

Algo = PPO if ALGO == "PPO" else SAC
kwargs = dict(
    policy="MultiInputPolicy",
    env=train_env,
    seed=SEED,
    verbose=0,
    policy_kwargs=sb3_policy_kwargs(64),
)
if ALGO == "PPO":
    kwargs.update(n_steps=256, batch_size=64, n_epochs=4)
else:
    kwargs.update(buffer_size=5_000, batch_size=64, learning_starts=256, train_freq=64)

model = Algo(**kwargs)
TIMESTEPS = 2048  # bump for research runs
print("learning", TIMESTEPS, "timesteps")
model.learn(total_timesteps=TIMESTEPS)
"""),
        code("""\
def rollout_sb3(model, env):
    def fn(obs, info, t):
        action, _ = model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=float)
    return rollout(env, fn)

test_kw = dict(mode="test", wrap_gymnasium=True, return_last_action=True, new_gym_api=True)
agent_env = make_poe(panel, CFG, cwd=ROOT / "results" / f"sb3_{ALGO.lower()}_agent", **test_kw)
eq_env = make_poe(panel, CFG, cwd=ROOT / "results" / f"sb3_{ALGO.lower()}_eq", **test_kw)
rp_env = make_poe(panel, CFG, cwd=ROOT / "results" / f"sb3_{ALGO.lower()}_rp", **test_kw)

agent_roll = rollout_sb3(model, agent_env)
eq_roll = rollout_equal_weight(eq_env)
rp_roll = rollout_risk_parity(rp_env, panel)
summary = compare_rollouts({ALGO: agent_roll, "equal_weight": eq_roll, "risk_parity": rp_roll})
print(summary.to_string())

def _series(roll, name):
    n = min(len(roll["dates"]), len(roll["values"]))
    return pd.Series(roll["values"][:n], index=pd.to_datetime(roll["dates"][:n]), name=name)

fig, ax = plt.subplots(figsize=(10, 4))
for roll, name in ((agent_roll, ALGO), (eq_roll, "equal-weight"), (rp_roll, "risk-parity")):
    _series(roll, name).plot(ax=ax)
ax.set_title("Test-window portfolio value")
ax.set_ylabel("value")
plt.tight_layout()
plt.show()
"""),
    ]
    write(filename, cells)


def nb04() -> None:
    cells = [
        md("# 04 — Full-universe template\n\nParameterised scaffold. No results. Pick a dataset and a universe rule; dates and seed still come from YAML."),
        code(PREAMBLE),
        code("""\
# Parameters (not dates — dates stay in configs/default.yaml)
DATASET = "ab_wrds"          # any name in configs/datasets.yaml
UNIVERSE_RULE = "full_window"  # ab_finrl | sandbox | full_window | top_n | listed
print("dataset", DATASET, "universe", UNIVERSE_RULE, "seed", SEED)
"""),
        code("""\
from sp500rl.data import get_panel, DatasetNotReady
from sp500rl.env.make_env import make_poe

try:
    panel = get_panel(DATASET, UNIVERSE_RULE, CFG)
except DatasetNotReady as exc:
    print(exc)
    raise
print("tickers", sorted(panel["tic"].unique()))
print("shape", panel.shape)

# Train / test env handles — plug in any agent from notebooks 01–03.
train_env = make_poe(panel, CFG, mode="train", return_last_action=False)
test_env = make_poe(panel, CFG, mode="test", return_last_action=False)
print("train episode_length", train_env.episode_length, "n", train_env.portfolio_size)
print("test  episode_length", test_env.episode_length)
print("TODO: attach DRLAgent or SB3 here; do not commit results in this template.")
"""),
    ]
    write("04_full_universe_template.ipynb", cells)


if __name__ == "__main__":
    NB.mkdir(parents=True, exist_ok=True)
    nb00()
    nb01()
    nb_sb3("02_sandbox_10_stocks_SB3_PPO.ipynb", "PPO", "02 — 10 stocks, SB3 PPO")
    nb_sb3("03_sandbox_10_stocks_SB3_SAC.ipynb", "SAC", "03 — 10 stocks, SB3 SAC")
    nb04()
