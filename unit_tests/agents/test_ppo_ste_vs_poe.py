from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_comparison_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "finrl"
        / "agents"
        / "portfolio_optimization"
        / "ppo_ste_vs_poe.py"
    )
    spec = importlib.util.spec_from_file_location("ppo_ste_vs_poe", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


comparison = _load_comparison_module()


def test_original_ppo_matches_asset_allocation_summary():
    result = comparison.original_ppo_transition()
    assert result.reward == pytest.approx(0.93)
    assert result.advantage == pytest.approx(0.87)
    assert result.portfolio_value == pytest.approx(100.934, rel=1e-6)
    np.testing.assert_allclose(result.holdings_value, [60.32, 40.614], rtol=1e-6)
    assert result.changed_original_ppo_steps == ()
    assert result.log_prob is not None
    assert result.critic_loss is not None


def test_ste_does_not_rebalance_when_cash_is_zero():
    result = comparison.ste_transition()
    # Buys cannot fill: the 50/50 book just rides A+4% / B-3.3%.
    expected_value = 50.0 * (1.04 + (1.0 - 0.033))
    assert result.portfolio_value == pytest.approx(expected_value)
    assert result.portfolio_value == pytest.approx(100.35)
    # 0.58 * 100 = 57.999... which StockTradingEnv truncates with astype(int).
    np.testing.assert_allclose(result.action_used, [57.0, 42.0])
    assert result.reward == pytest.approx(
        (expected_value - 100.0) * comparison.STE_REWARD_SCALING
    )
    # Same critic numbers as the summary, tiny STE reward → advantage flips sign.
    assert result.advantage < 0
    assert comparison.original_ppo_transition().advantage > 0
    assert 2 in result.changed_original_ppo_steps


def test_ste_integer_rounding_can_zero_the_trade():
    result = comparison.ste_transition(hmax=1)
    np.testing.assert_array_equal(result.action_used, [0.0, 0.0])
    assert result.portfolio_value == pytest.approx(100.35)


def test_poe_rebalances_and_uses_log_return():
    result = comparison.poe_env_transition()
    np.testing.assert_allclose(result.action_used, [0.0, 0.58, 0.42])
    np.testing.assert_allclose(result.holdings_value, [60.32, 40.614], rtol=1e-6)
    assert result.portfolio_value == pytest.approx(100.934, rel=1e-6)
    expected_reward = float(np.log(100.934 / 100.0))
    assert result.reward == pytest.approx(expected_reward)
    assert result.reward != pytest.approx(0.93)
    assert result.advantage is None
    assert 1 in result.changed_original_ppo_steps
    assert 2 in result.changed_original_ppo_steps


def test_poe_softmax_when_weights_do_not_sum_to_one():
    raw = np.array([0.2, 0.58, 0.42])
    assert not np.isclose(np.sum(raw), 1.0)
    weights = comparison.softmax(raw)
    assert weights.sum() == pytest.approx(1.0)
    assert np.min(weights) >= 0


def test_jiang_pg_replaces_ppo_steps_three_through_seven():
    pg = comparison.poe_policy_gradient_update()
    growth = 0.58 * 1.04 + 0.42 * (1.0 - 0.033)
    assert pg.actor_loss == pytest.approx(-np.log(growth))
    assert pg.advantage is None
    assert pg.critic_loss is None
    assert pg.log_prob is None
    assert pg.changed_original_ppo_steps == (1, 2, 3, 4, 5, 6, 7)


def test_ppo_on_poe_keeps_clipping_but_advantage_changes():
    original = comparison.original_ppo_transition()
    poe_ppo = comparison.ppo_on_poe_transition()
    assert poe_ppo.log_prob == pytest.approx(original.log_prob)
    assert poe_ppo.advantage != pytest.approx(original.advantage)
    assert poe_ppo.advantage < 0
    # Clipped surrogate is still the original PPO formula, with the new A.
    assert poe_ppo.actor_loss == pytest.approx(
        comparison.ppo_clipped_actor_loss(1.05, poe_ppo.advantage)
    )
    assert 3 not in poe_ppo.changed_original_ppo_steps
    assert 5 not in poe_ppo.changed_original_ppo_steps


def test_every_original_ppo_step_is_changed_by_finrl_poe_agent():
    changes = comparison.ppo_step_changes()
    assert [row.step for row in changes] == [1, 2, 3, 4, 5, 6, 7]
    assert all(row.poe_changes_original_ppo for row in changes)
    # STE keeps the PPO update (steps 3, 5, 6, 7) and only rewrites the MDP.
    ste = comparison.ste_transition()
    assert 5 not in ste.changed_original_ppo_steps
    assert 6 not in ste.changed_original_ppo_steps
    assert 7 not in ste.changed_original_ppo_steps


def test_walkthrough_report_mentions_the_fork_points():
    report = comparison.format_walkthrough(comparison.run_asset_allocation_example())
    assert "CHANGES original PPO" in report
    assert "0.87" in report
    assert "log-return" in report
    assert "No critic" in report or "no critic" in report
    assert "hmax" in report


def test_asset_returns_match_the_summary_book():
    rebalanced = 100.0 * comparison.ACTION * (1.0 + comparison.ASSET_RETURNS)
    assert rebalanced[0] == pytest.approx(60.32)
    assert rebalanced[1] == pytest.approx(40.614)
    assert float(rebalanced.sum()) == pytest.approx(100.934)
