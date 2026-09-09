from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from sp500rl.env.extractors import EIIEFeaturesExtractor


def test_eiie_extractor_keeps_tensor_layout():
    f, n, t = 3, 10, 50
    space = spaces.Dict(
        {
            "state": spaces.Box(-np.inf, np.inf, shape=(f, n, t), dtype=np.float32),
            "last_action": spaces.Box(0, 1, shape=(n + 1,), dtype=np.float32),
        }
    )
    ext = EIIEFeaturesExtractor(space, features_dim=32)
    obs = {
        "state": torch.zeros(2, f, n, t),
        "last_action": torch.zeros(2, n + 1),
    }
    out = ext(obs)
    assert out.shape == (2, 32)
    assert not torch.isnan(out).any()
