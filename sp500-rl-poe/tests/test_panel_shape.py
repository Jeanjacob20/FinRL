from __future__ import annotations

from sp500rl.config import load_config
from sp500rl.data.datasets import build_panel
from sp500rl.data.features import add_features
from sp500rl.data.panel import assert_balanced_panel, build_panel as assemble
from sp500rl.data.synthetic import make_synthetic_canonical
from sp500rl.data.universe import select_universe


def test_panel_balanced_no_nan_sorted():
    raw = make_synthetic_canonical(start="2019-01-02", end="2020-06-30", seed=7)
    cfg = load_config()
    featured = add_features(raw)
    panel = assemble(
        featured,
        feature_cols=list(cfg["poe"]["features"]),
        missing_policy="drop_ticker",
    )
    assert_balanced_panel(panel, feature_cols=cfg["poe"]["features"])
    n_tic = panel["tic"].nunique()
    n_date = panel["date"].nunique()
    assert len(panel) == n_tic * n_date
    assert panel["date"].is_monotonic_increasing
    assert str(panel["date"].dtype).startswith("datetime64")
    assert not panel[cfg["poe"]["features"]].isna().any().any()


def test_ab_finrl_list_filters_to_ten():
    raw = make_synthetic_canonical(seed=8)
    cfg = load_config()
    out = select_universe(raw, cfg, rule="ab_finrl")
    assert out["tic"].nunique() == 10


def test_build_panel_from_canonical_prices():
    cfg = load_config()
    prices = make_synthetic_canonical(seed=9)
    panel = build_panel(prices, cfg, universe="ab_finrl")
    assert panel["tic"].nunique() == 10
    assert set(["date", "tic", *cfg["poe"]["features"]]).issubset(panel.columns)
