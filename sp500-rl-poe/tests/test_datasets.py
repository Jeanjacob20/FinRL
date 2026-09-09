"""The four questions: what / how / where / use — through the registry."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sp500rl.config import load_config
from sp500rl.data import (
    DatasetNotReady,
    fetch,
    get_dataset,
    get_panel,
    list_datasets,
    load_prices,
    status,
)
from sp500rl.data.panel import assert_balanced_panel
from sp500rl.data.schema import validate
from sp500rl.data.universe import AB_FINRL_TICKERS


def _cfg_in(tmp_path: Path) -> dict:
    """Config whose processed dir lives in tmp so tests never touch data/."""

    cfg = load_config()
    cfg["paths"]["processed_dir"] = str(tmp_path / "processed")
    cfg["dates"] = {
        "start": "2020-01-02",
        "end": "2020-12-31",
        "train_start": "2020-04-01",
        "train_end": "2020-09-30",
        "test_start": "2020-10-01",
        "test_end": "2020-12-31",
    }
    return cfg


def _redirect(monkeypatch, tmp_path: Path) -> None:
    """Point every dataset's files under tmp_path."""

    import sp500rl.data.datasets as mod

    def _resolve(rel):
        p = Path(rel)
        return p if p.is_absolute() else tmp_path / p

    monkeypatch.setattr(mod, "_resolve", _resolve)


def test_what_registry_lists_datasets():
    names = {ds.name for ds in list_datasets()}
    assert {"ab_wrds", "yahoo", "synthetic"} <= names
    ab = get_dataset("ab_wrds")
    assert ab.how == "manual"
    assert ab.format == "wrds_crsp"
    assert set(ab.files) == {"prices", "tickers"}
    assert ab.what


def test_where_status_reports_missing(monkeypatch, tmp_path):
    _redirect(monkeypatch, tmp_path)
    rep = status("ab_wrds")["ab_wrds"]
    assert rep["ready"] is False
    assert rep["files"]["prices"]["path"].endswith("data/raw/ab_wrds/wrds_processed.csv")
    assert rep["files"]["tickers"]["path"].endswith("data/raw/ab_wrds/wrds_tickers.csv")


def test_how_manual_dataset_explains_itself(monkeypatch, tmp_path):
    _redirect(monkeypatch, tmp_path)
    with pytest.raises(DatasetNotReady) as exc:
        fetch("ab_wrds", _cfg_in(tmp_path))
    msg = str(exc.value)
    assert "wrds_processed.csv" in msg and "wrds_tickers.csv" in msg
    assert "AB_finRL" in msg


def test_how_synthetic_fetches_itself_and_loads(monkeypatch, tmp_path):
    _redirect(monkeypatch, tmp_path)
    cfg = _cfg_in(tmp_path)
    written = fetch("synthetic", cfg)
    assert written["prices"].exists()
    prices = load_prices("synthetic", cfg)
    assert validate(prices).ok
    assert set(prices["tic"]) == set(AB_FINRL_TICKERS)


def test_how_fake_writes_dataset_in_its_own_format(monkeypatch, tmp_path):
    _redirect(monkeypatch, tmp_path)
    cfg = _cfg_in(tmp_path)
    written = fetch("ab_wrds", cfg, fake=True)
    raw = pd.read_csv(written["prices"])
    assert {"permno", "prc", "cfacpr"} <= set(raw.columns)
    assert written["tickers"].exists()
    prices = load_prices("ab_wrds", cfg)
    assert validate(prices).ok
    assert "permno" in prices.columns


def test_use_get_panel_ab_finrl_and_cache(monkeypatch, tmp_path):
    _redirect(monkeypatch, tmp_path)
    cfg = _cfg_in(tmp_path)
    fetch("ab_wrds", cfg, fake=True)
    panel = get_panel("ab_wrds", "ab_finrl", cfg)
    assert_balanced_panel(panel, feature_cols=cfg["poe"]["features"])
    assert set(panel["tic"]) == set(AB_FINRL_TICKERS)
    cached = tmp_path / "processed" / "ab_wrds__ab_finrl.parquet"
    assert cached.exists()
    again = get_panel("ab_wrds", "ab_finrl", cfg)
    assert len(again) == len(panel)


def test_use_universe_listed_and_lists_are_independent(monkeypatch, tmp_path):
    _redirect(monkeypatch, tmp_path)
    cfg = _cfg_in(tmp_path)
    cfg["universe"]["lists"]["sandbox"] = ["AAPL", "MSFT"]
    fetch("ab_wrds", cfg, fake=True)
    small = get_panel("ab_wrds", "sandbox", cfg, cache=False)
    assert set(small["tic"]) == {"AAPL", "MSFT"}
    listed = get_panel("ab_wrds", "listed", cfg, cache=False)
    assert set(listed["tic"]) == set(AB_FINRL_TICKERS)
    ab = get_panel("ab_wrds", "ab_finrl", cfg, cache=False)
    assert set(ab["tic"]) == set(AB_FINRL_TICKERS)


def test_use_default_dataset_from_config(monkeypatch, tmp_path):
    _redirect(monkeypatch, tmp_path)
    cfg = _cfg_in(tmp_path)
    cfg["dataset"] = "synthetic"
    panel = get_panel(cfg=cfg, cache=False)
    assert panel["tic"].nunique() == 10
