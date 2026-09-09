"""Gymnasium wrapper around FinRL's gym.Env PortfolioOptimizationEnv.

POE subclasses ``gym.Env``. Stable-Baselines3 >= 2.0 requires
``gymnasium.Env``. Setting ``new_gym_api=True`` only changes the tuple
lengths; the class is still old Gym. This wrapper converts spaces and
delegates ``reset`` / ``step``.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces as gym_spaces


def convert_space(space):
    """Map a ``gym.spaces`` object onto ``gymnasium.spaces``."""

    if isinstance(space, gym_spaces.Space):
        return space
    cls = space.__class__.__name__
    if cls == "Box":
        return gym_spaces.Box(
            low=np.array(space.low, copy=True),
            high=np.array(space.high, copy=True),
            shape=tuple(space.shape),
            dtype=space.dtype,
        )
    if cls == "Dict":
        return gym_spaces.Dict({k: convert_space(v) for k, v in space.spaces.items()})
    if cls == "Discrete":
        return gym_spaces.Discrete(int(space.n))
    raise TypeError(f"Cannot convert space {type(space)}")


class GymnasiumPOE(gym.Env):
    """Adapter: inner POE + Gymnasium API.

    Observation / action shapes match the inner env (see ``docs/poe_contract.md``).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.env._new_gym_api = True
        self.observation_space = convert_space(env.observation_space)
        self.action_space = convert_space(env.action_space)
        self.portfolio_size = env.portfolio_size
        self.episode_length = env.episode_length

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None and hasattr(self.env, "_seed"):
            self.env._seed(seed)
        out = self.env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
            return obs, info if isinstance(info, dict) else {}
        return out, {}

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            return obs, float(reward), bool(terminated), bool(truncated), info
        obs, reward, done, info = out
        return obs, float(reward), bool(done), False, info

    def render(self):
        return self.env.render()

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)
