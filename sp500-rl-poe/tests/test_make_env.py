from __future__ import annotations

import numpy as np
import pytest

from sp500rl.config import load_config
from sp500rl.data.pipeline import build_panel_from_file
from sp500rl.env.make_env import make_poe
from sp500rl.finrl_bootstrap import import_pg_stack, import_poe


def test_finrl_poe_and_agents_import():
    POE = import_poe()
    DRLAgent, EIIE, GPM, PolicyGradient = import_pg_stack()
    assert POE.__name__ == "PortfolioOptimizationEnv"
    assert DRLAgent.__name__ == "DRLAgent"
    assert EIIE.__name__ == "EIIE"
    assert GPM.__name__ == "GPM"
    assert PolicyGradient.__name__ == "PolicyGradient"


def test_make_poe_box_and_dict_obs(tmp_path):
    cfg = load_config()
    panel = build_panel_from_file(
        tmp_path / "missing.csv",
        cfg=cfg,
        universe="sandbox",
        use_synthetic=True,
    )
    env_box = make_poe(panel, cfg, mode="test", return_last_action=False, new_gym_api=False)
    obs = env_box.reset()
    assert not isinstance(obs, dict)
    f = len(cfg["poe"]["features"])
    n = env_box.portfolio_size
    t = int(cfg["poe"]["time_window"])
    assert obs.shape == (f, n, t)

    env_dict = make_poe(panel, cfg, mode="test", return_last_action=True, new_gym_api=False)
    obs_d = env_dict.reset()
    assert set(obs_d) == {"state", "last_action"}
    assert obs_d["state"].shape == (f, n, t)
    assert obs_d["last_action"].shape == (n + 1,)

    action = np.zeros(n + 1, dtype=np.float32)
    action[1:] = 1.0 / n
    nxt, reward, done, info = env_box.step(action)
    assert nxt.shape == (f, n, t)
    assert "trf_mu" in info or done
