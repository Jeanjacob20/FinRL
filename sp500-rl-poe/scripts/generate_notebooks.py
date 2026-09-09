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
print(f"seed={SEED}")
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
        md("# 00 — Data pipeline check\n\nLoad a raw CSV through the loader + adapter, show the validation report, build the POE panel, and prove the same interface works on a Yahoo-format file."),
        code(PREAMBLE),
        md("## Canonical CSV (synthetic if needed)"),
        code("""\
from sp500rl.data.loaders import load_csv
from sp500rl.data.pipeline import build_panel_from_file
from sp500rl.data.schema import validate
from sp500rl.data.synthetic import write_synthetic
from sp500rl.data.panel import assert_balanced_panel

raw_dir = ROOT / CFG["paths"]["raw_dir"]
raw_dir.mkdir(parents=True, exist_ok=True)
canonical_path = raw_dir / "synthetic_canonical.csv"
if not canonical_path.exists():
    write_synthetic(raw_dir, start=CFG["dates"]["start"], end=CFG["dates"]["end"], seed=SEED)
    print("wrote synthetic CSVs under", raw_dir)

prices = load_csv(canonical_path, adapter="generic")
report = validate(prices)
print(report.summary())
print("tickers", sorted(prices["tic"].unique()))
print("date range", prices["date"].min().date(), "→", prices["date"].max().date())
print("shape", prices.shape)
"""),
        md("## Panel + price plot"),
        code("""\
import matplotlib.pyplot as plt

panel = build_panel_from_file(canonical_path, cfg=CFG, adapter="generic", universe="sandbox")
assert_balanced_panel(panel, feature_cols=CFG["poe"]["features"])
print("panel shape", panel.shape)
print("panel dates", panel["date"].min().date(), "→", panel["date"].max().date())
print("tickers", sorted(panel["tic"].unique()))

pivot = panel.pivot(index="date", columns="tic", values="close")
ax = pivot.iloc[:, :4].plot(figsize=(10, 4), title="Synthetic close (first 4 tickers)")
ax.set_ylabel("close")
plt.tight_layout()
plt.show()

processed = ROOT / CFG["paths"]["processed_dir"]
processed.mkdir(parents=True, exist_ok=True)
out = processed / "panel_sandbox.parquet"
panel.to_parquet(out, index=False)
print("wrote", out)
"""),
        md("## Same interface, Yahoo-format CSV"),
        code("""\
yahoo_path = raw_dir / "synthetic_yahoo.csv"
yahoo_prices = load_csv(yahoo_path, adapter="yahoo")
print(validate(yahoo_prices).summary())
yahoo_panel = build_panel_from_file(yahoo_path, cfg=CFG, adapter="yahoo", universe="sandbox")
assert_balanced_panel(yahoo_panel, feature_cols=CFG["poe"]["features"])
print("yahoo panel shape", yahoo_panel.shape, "tickers", sorted(yahoo_panel["tic"].unique()))
assert set(yahoo_panel["tic"].unique()) == set(panel["tic"].unique())
print("Yahoo-format path matches canonical ticker set.")
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
        md("# 01 — Sandbox 10 stocks, FinRL EIIE / PolicyGradient\n\nTrain on the sandbox universe, evaluate on the test window, plot cumulative value vs equal-weight and risk parity."),
        code(PREAMBLE),
        code("""\
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sp500rl.data.pipeline import build_panel_from_file
from sp500rl.env.make_env import make_poe
from sp500rl.finrl_bootstrap import import_pg_stack
from sp500rl.baselines.simple import (
    rollout_equal_weight,
    rollout_pg_policy,
    rollout_risk_parity,
)
from sp500rl.eval.metrics import compare_rollouts

processed = ROOT / CFG["paths"]["processed_dir"] / "panel_sandbox.parquet"
if processed.exists():
    panel = pd.read_parquet(processed)
    panel["date"] = pd.to_datetime(panel["date"])
else:
    panel = build_panel_from_file(None, cfg=CFG, universe="sandbox", use_synthetic=True)
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
        md(f"# {title}\n\nSame protocol as notebook 01: sandbox universe, YAML dates, custom `(f,n,t)` extractor. Default `MlpPolicy`/`CnnPolicy` are not used."),
        code(PREAMBLE),
        code("""\
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC

from sp500rl.data.pipeline import build_panel_from_file
from sp500rl.env.extractors import sb3_policy_kwargs
from sp500rl.env.make_env import make_poe
from sp500rl.baselines.simple import rollout, rollout_equal_weight, rollout_risk_parity
from sp500rl.eval.metrics import compare_rollouts

processed = ROOT / CFG["paths"]["processed_dir"] / "panel_sandbox.parquet"
if processed.exists():
    panel = pd.read_parquet(processed)
    panel["date"] = pd.to_datetime(panel["date"])
else:
    panel = build_panel_from_file(None, cfg=CFG, universe="sandbox", use_synthetic=True)
print("panel", panel.shape)
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
        md("# 04 — Full-universe template\n\nParameterised scaffold. No results. Change `UNIVERSE_RULE` and rebuild the panel; dates and seed still come from YAML."),
        code(PREAMBLE),
        code("""\
# Parameters (not dates — dates stay in configs/default.yaml)
UNIVERSE_RULE = "full_window"  # sandbox | full_window | top_n
INPUT_CSV = ROOT / CFG["paths"]["raw_dir"] / "synthetic_canonical.csv"
ADAPTER = CFG.get("adapter", "generic")

print("universe rule", UNIVERSE_RULE)
print("seed", SEED)
print("dates", CFG["dates"])
"""),
        code("""\
from sp500rl.data.pipeline import build_panel_from_file
from sp500rl.env.make_env import make_poe

panel = build_panel_from_file(
    INPUT_CSV if INPUT_CSV.exists() else None,
    cfg=CFG,
    adapter=ADAPTER,
    universe=UNIVERSE_RULE,
    use_synthetic=not INPUT_CSV.exists(),
)
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
    nb_sb3("02_sandbox_10_stocks_SB3_PPO.ipynb", "PPO", "02 — Sandbox 10 stocks, SB3 PPO")
    nb_sb3("03_sandbox_10_stocks_SB3_SAC.ipynb", "SAC", "03 — Sandbox 10 stocks, SB3 SAC")
    nb04()
