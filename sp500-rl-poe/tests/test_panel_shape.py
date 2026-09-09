from __future__ import annotations

from sp500rl.data.features import add_features
from sp500rl.data.panel import assert_balanced_panel, build_panel
from sp500rl.data.pipeline import build_panel_from_file
from sp500rl.data.synthetic import make_synthetic_canonical
from sp500rl.data.universe import select_universe
from sp500rl.config import load_config


def test_panel_balanced_no_nan_sorted():
    raw = make_synthetic_canonical(start="2019-01-02", end="2020-06-30", seed=7)
    cfg = load_config()
    featured = add_features(raw)
    panel = build_panel(
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


def test_sandbox_universe_filters_to_ten():
    raw = make_synthetic_canonical(seed=8)
    cfg = load_config()
    out = select_universe(raw, cfg, rule="sandbox")
    assert out["tic"].nunique() == 10


def test_pipeline_synthetic(tmp_path, monkeypatch):
    cfg = load_config()
    panel = build_panel_from_file(
        tmp_path / "does_not_exist.csv",
        cfg=cfg,
        adapter="generic",
        universe="sandbox",
        use_synthetic=True,
    )
    assert panel["tic"].nunique() == 10
    assert set(["date", "tic", *cfg["poe"]["features"]]).issubset(panel.columns)
