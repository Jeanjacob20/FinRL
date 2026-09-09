from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _load_ppo_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "finrl"
        / "agents"
        / "portfolio_optimization"
        / "ppo.py"
    )
    spec = importlib.util.spec_from_file_location("poe_ppo", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ppo = _load_ppo_module()


class DummyPOE:
    """Minimal POE stand-in: Box obs, simplex action, log-return reward."""

    def __init__(self, n_assets=2, n_features=3, time_window=4, episode_length=8):
        self.portfolio_size = n_assets
        self.episode_length = episode_length
        self.action_space = SimpleNamespace(shape=(n_assets + 1,))
        self.observation_space = SimpleNamespace(
            shape=(n_features, n_assets, time_window)
        )
        self._t = 0
        self._price_rel = np.array([1.0, 1.05, 1.0], dtype=np.float32)

    def reset(self):
        self._t = 0
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def step(self, action):
        weights = np.asarray(action, dtype=np.float64)
        if not (np.isclose(weights.sum(), 1.0, atol=1e-4) and np.min(weights) >= -1e-6):
            weights = ppo.logits_to_weights(weights)
        growth = float(np.dot(weights, self._price_rel))
        reward = float(np.log(max(growth, 1e-8)))
        self._t += 1
        done = self._t >= self.episode_length
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {"price_variation": self._price_rel, "trf_mu": 1.0}
        return obs, reward, done, info


def test_logits_to_weights_are_a_poe_portfolio_vector():
    weights = ppo.logits_to_weights(np.array([0.0, 0.58, 0.42]))
    assert weights.shape == (3,)
    assert weights.sum() == pytest.approx(1.0)
    assert np.min(weights) >= 0


def test_flatten_concatenates_last_action_for_poe_state():
    obs = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    last = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    flat = ppo.flatten_poe_state(obs, last)
    assert flat.shape == (15,)
    np.testing.assert_array_equal(flat[-3:], last)


def test_gae_matches_the_summary_one_step_advantage():
    # r=0.93, γ=0.99, V(s)=105, V(s')=106 → A = 0.87 when the episode continues
    advantages, returns = ppo.compute_gae(
        rewards=np.array([0.93]),
        values=np.array([105.0]),
        dones=np.array([False]),
        next_value=106.0,
        gamma=0.99,
        gae_lambda=1.0,
    )
    assert advantages[0] == pytest.approx(0.87)
    assert returns[0] == pytest.approx(0.93 + 0.99 * 106.0)


def test_act_samples_simplex_weights_and_log_prob():
    torch.manual_seed(0)
    agent = ppo.PPO(DummyPOE(), n_steps=4, n_epochs=1, minibatch_size=4, device="cpu")
    obs = agent.env.reset()
    last = agent._initial_weights()
    weights, logits, log_prob, value = agent.act(obs, last)
    assert weights.shape == (3,)
    assert weights.sum() == pytest.approx(1.0, abs=1e-5)
    assert logits.shape == (3,)
    assert np.isfinite(log_prob)
    assert np.isfinite(value)
    assert last[0] == pytest.approx(1.0)


def test_train_keeps_ppo_update_and_discards_rollout():
    torch.manual_seed(1)
    np.random.seed(1)
    agent = ppo.PPO(
        DummyPOE(episode_length=6),
        n_steps=4,
        n_epochs=2,
        minibatch_size=4,
        lr=1e-3,
        device="cpu",
    )
    agent.train(episodes=2)
    assert agent.last_update
    assert "actor_loss" in agent.last_update
    assert "critic_loss" in agent.last_update
    assert np.isfinite(agent.last_update["actor_loss"])
    assert np.isfinite(agent.last_update["critic_loss"])
    # Step 7: on-policy batch is thrown away after the update
    assert agent.rollout["rewards"] == []


def test_greedy_test_does_not_learn_online():
    torch.manual_seed(2)
    agent = ppo.PPO(DummyPOE(episode_length=5), n_steps=8, device="cpu")
    before = [p.detach().clone() for p in agent.actor_critic.parameters()]
    agent.test(DummyPOE(episode_length=5))
    after = list(agent.actor_critic.parameters())
    for b, a in zip(before, after):
        assert torch.equal(b, a)
