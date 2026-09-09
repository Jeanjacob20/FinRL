"""SB3 feature extractors that keep POE's ``(f, n, t)`` tensor structure.

Default ``MlpPolicy`` / ``CnnPolicy`` are excluded: they flatten the
observation or assume HWC images and destroy the (features, assets, time)
layout. This extractor is a small per-asset 1-D conv stack in the EIIE
spirit (Jiang et al., kernels of height 1).

When ``return_last_action=True`` the observation is a Dict
``{"state": (f,n,t), "last_action": (n+1,)}``. Use SB3 ``MultiInputPolicy``
with ``features_extractor_class=EIIEFeaturesExtractor``.
"""

from __future__ import annotations

import gymnasium
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class EIIEFeaturesExtractor(BaseFeaturesExtractor):
    """Per-asset conv encoder on a POE ``(f, n, t)`` state.

    Parameters
    ----------
    observation_space
        ``Box(f, n, t)`` or ``Dict`` with ``state`` / ``last_action``.
    features_dim
        Output width fed to the SB3 MLP policy head. Shape ``(features_dim,)``.
    """

    def __init__(
        self,
        observation_space: gymnasium.Space,
        features_dim: int = 64,
        conv_mid: int = 2,
        conv_final: int = 20,
        k_size: int = 3,
    ):
        super().__init__(observation_space, features_dim)
        if isinstance(observation_space, spaces.Dict):
            state_space = observation_space.spaces["state"]
            self._is_dict = True
            last_dim = int(np.prod(observation_space.spaces["last_action"].shape))
        else:
            state_space = observation_space
            self._is_dict = False
            last_dim = 0

        n_features, n_assets, time_window = (int(x) for x in state_space.shape)
        if time_window < k_size:
            raise ValueError(
                f"time_window={time_window} < k_size={k_size}; increase poe.time_window"
            )
        n_size = time_window - k_size + 1
        self.conv = nn.Sequential(
            nn.Conv2d(n_features, conv_mid, kernel_size=(1, k_size)),
            nn.ReLU(),
            nn.Conv2d(conv_mid, conv_final, kernel_size=(1, n_size)),
            nn.ReLU(),
        )
        conv_out = conv_final * n_assets
        self.linear = nn.Linear(conv_out + last_dim, features_dim)

    def forward(self, observations) -> torch.Tensor:
        """Encode a batch of POE observations.

        Parameters
        ----------
        observations
            Tensor ``(B, f, n, t)`` or dict of tensors.

        Returns
        -------
        torch.Tensor
            Shape ``(B, features_dim)``.
        """

        if self._is_dict:
            state = observations["state"]
            last = observations["last_action"]
        else:
            state = observations
            last = None
        x = self.conv(state).flatten(start_dim=1)
        if last is not None:
            x = torch.cat([x, last.reshape(last.shape[0], -1)], dim=1)
        return self.linear(x)


def sb3_policy_kwargs(features_dim: int = 64) -> dict:
    """``policy_kwargs`` for PPO/SAC/TD3 ``MultiInputPolicy``."""

    return {
        "features_extractor_class": EIIEFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": features_dim},
    }
