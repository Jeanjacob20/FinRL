from __future__ import annotations

import numpy as np
import pandas as pd

import pytest

from sp500rl.eval.metrics import max_drawdown, sharpe, turnover


def test_sharpe_constant_returns():
    r = np.full(252, 0.001)
    s = sharpe(r)
    assert s > 0


def test_max_drawdown():
    v = np.array([100.0, 110.0, 90.0, 95.0])
    assert max_drawdown(v) == pytest.approx((90 / 110) - 1)


def test_turnover_zero_when_constant():
    w = np.tile(np.array([0.0, 0.5, 0.5]), (5, 1))
    assert turnover(w) == 0.0
