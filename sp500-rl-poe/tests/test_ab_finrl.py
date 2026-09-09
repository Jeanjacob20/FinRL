from __future__ import annotations

from pathlib import Path

import pandas as pd

from sp500rl.config import load_config
from sp500rl.data.ab_finrl import load_ab_extract, load_wrds_tickers
from sp500rl.data.pipeline import build_panel_from_file
from sp500rl.data.schema import validate
from sp500rl.data.synthetic import (
    make_synthetic_canonical,
    to_wrds_processed,
    to_wrds_tickers,
    write_synthetic,
)


def test_ab_extract_joins_tickers_and_adjusts_crsp(tmp_path: Path):
    canonical = make_synthetic_canonical(
        start="2020-01-02", end="2020-03-31", tickers=["AAPL", "MSFT"], seed=11
    )
    processed = to_wrds_processed(canonical)
    # Negative PRC on the first AAPL row (bid/ask midpoint).
    processed.loc[processed.index[0], "prc"] = -abs(float(processed.loc[processed.index[0], "prc"]))
    p_path = tmp_path / "wrds_processed.csv"
    t_path = tmp_path / "wrds_tickers.csv"
    processed.to_csv(p_path, index=False)
    to_wrds_tickers(canonical).to_csv(t_path, index=False)

    names = load_wrds_tickers(t_path)
    assert set(names["ticker"]) == {"AAPL", "MSFT"}

    out = load_ab_extract(p_path, t_path)
    report = validate(out)
    assert report.ok, report.summary()
    assert set(out["tic"]) == {"AAPL", "MSFT"}
    assert (out["close"] > 0).all()


def test_pipeline_native_wrds_filenames(tmp_path: Path, monkeypatch):
    cfg = load_config()
    cfg = {**cfg, "paths": {**cfg["paths"], "raw_dir": str(tmp_path),
                            "wrds_processed": str(tmp_path / "wrds_processed.csv"),
                            "wrds_tickers": str(tmp_path / "wrds_tickers.csv")}}
    write_synthetic(tmp_path, start="2019-06-03", end="2020-12-31", seed=12)
    panel = build_panel_from_file(
        tmp_path / "wrds_processed.csv",
        cfg=cfg,
        universe="sandbox",
        tickers_path=tmp_path / "wrds_tickers.csv",
    )
    assert panel["tic"].nunique() == 10
    assert {"date", "tic", "close"}.issubset(panel.columns)


def test_universe_rule_wrds_tickers(tmp_path: Path):
    cfg = load_config()
    cfg["paths"]["wrds_tickers"] = str(tmp_path / "wrds_tickers.csv")
    canonical = make_synthetic_canonical(
        start="2020-01-02", end="2020-06-30", tickers=["AAPL", "MSFT", "JPM"], seed=13
    )
    to_wrds_tickers(canonical).to_csv(tmp_path / "wrds_tickers.csv", index=False)
    # Restrict the tickers file to two names.
    names = pd.read_csv(tmp_path / "wrds_tickers.csv")
    names = names.loc[names["ticker"].isin(["AAPL", "MSFT"])]
    names.to_csv(tmp_path / "wrds_tickers.csv", index=False)

    from sp500rl.data.features import add_features
    from sp500rl.data.universe import select_universe

    cfg["universe"]["rule"] = "wrds_tickers"
    selected = select_universe(canonical, cfg, rule="wrds_tickers")
    assert set(selected["tic"].str.upper()) <= {"AAPL", "MSFT"}
    featured = add_features(selected)
    assert featured["tic"].nunique() == 2


def test_ab_finrl_universe_defaults_to_ten_selected_names():
    from sp500rl.data.universe import AB_FINRL_TICKERS, list_rules, select_universe

    raw = make_synthetic_canonical(start="2020-01-02", end="2020-06-30", seed=14)
    cfg = load_config()
    out = select_universe(raw, cfg, rule="ab_finrl")
    assert set(out["tic"].str.upper()) == set(AB_FINRL_TICKERS)
    assert "ab_finrl" in list_rules()


def test_ab_finrl_yaml_list_independent_of_sandbox(tmp_path):
    from sp500rl.data.universe import select_universe

    raw = make_synthetic_canonical(
        start="2020-01-02",
        end="2020-03-31",
        tickers=["AAPL", "MSFT", "JPM", "XOM"],
        seed=15,
    )
    cfg = load_config()
    cfg["paths"]["ab_finrl_tickers"] = str(tmp_path / "absent.csv")
    cfg["paths"]["wrds_tickers"] = str(tmp_path / "absent_wrds.csv")
    cfg["universe"]["sandbox_tickers"] = ["AAPL", "MSFT"]
    cfg["universe"]["ab_finrl_tickers"] = ["JPM", "XOM"]
    sandbox = select_universe(raw, cfg, rule="sandbox")
    ab = select_universe(raw, cfg, rule="ab_finrl")
    assert set(sandbox["tic"].str.upper()) == {"AAPL", "MSFT"}
    assert set(ab["tic"].str.upper()) == {"JPM", "XOM"}


def test_ab_finrl_sidecar_csv(tmp_path):
    from sp500rl.data.universe import select_universe

    raw = make_synthetic_canonical(
        start="2020-01-02",
        end="2020-03-31",
        tickers=["AAPL", "MSFT", "JPM", "XOM"],
        seed=16,
    )
    side = tmp_path / "ab_finrl_tickers.csv"
    pd.DataFrame({"ticker": ["JPM", "XOM"]}).to_csv(side, index=False)
    cfg = load_config()
    cfg["paths"]["ab_finrl_tickers"] = str(side)
    cfg["paths"]["wrds_tickers"] = str(tmp_path / "absent.csv")
    out = select_universe(raw, cfg, rule="ab_finrl")
    assert set(out["tic"].str.upper()) == {"JPM", "XOM"}


def test_ab_finrl_selected_flag_on_wrds_tickers(tmp_path):
    from sp500rl.data.universe import select_universe

    canonical = make_synthetic_canonical(
        start="2020-01-02",
        end="2020-03-31",
        tickers=["AAPL", "MSFT", "JPM"],
        seed=17,
    )
    names = to_wrds_tickers(canonical)
    names["selected"] = names["ticker"].isin(["AAPL", "MSFT"]).astype(int)
    names.to_csv(tmp_path / "wrds_tickers.csv", index=False)
    cfg = load_config()
    cfg["paths"]["ab_finrl_tickers"] = str(tmp_path / "absent.csv")
    cfg["paths"]["wrds_tickers"] = str(tmp_path / "wrds_tickers.csv")
    out = select_universe(canonical, cfg, rule="ab_finrl")
    assert set(out["tic"].str.upper()) == {"AAPL", "MSFT"}
