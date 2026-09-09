from __future__ import annotations

from pathlib import Path

import pandas as pd

from sp500rl.data.loaders import load_csv
from sp500rl.data.schema import validate
from sp500rl.data.synthetic import make_synthetic_canonical, to_wide_close, to_yahoo_format


def test_loader_wide_format_csv(tmp_path: Path):
    canonical = make_synthetic_canonical(
        start="2020-01-02", end="2020-02-28", tickers=["AAPL", "MSFT", "JPM"], seed=4
    )
    wide = to_wide_close(canonical)
    path = tmp_path / "wide.csv"
    wide.to_csv(path, index=False)
    loaded = load_csv(path, adapter="generic")
    report = validate(loaded)
    assert report.ok, report.summary()
    assert set(loaded["tic"].unique()) == {"AAPL", "MSFT", "JPM"}
    assert loaded["close"].notna().all()
    # Wide files only carry close; OHL are filled from close.
    assert (loaded["open"] == loaded["close"]).all()


def test_loader_long_odd_names_column_map(tmp_path: Path):
    canonical = make_synthetic_canonical(
        start="2020-01-02", end="2020-02-28", tickers=["AAPL", "JNJ"], seed=5
    )
    odd = pd.DataFrame(
        {
            "datadate": canonical["date"].dt.strftime("%Y%m%d"),
            "symbol": canonical["tic"],
            "o": canonical["open"],
            "h": canonical["high"],
            "l": canonical["low"],
            "px": canonical["close"],
            "v": canonical["volume"],
        }
    )
    path = tmp_path / "odd.csv"
    odd.to_csv(path, index=False)
    loaded = load_csv(
        path,
        adapter="generic",
        column_map={
            "datadate": "date",
            "symbol": "tic",
            "o": "open",
            "h": "high",
            "l": "low",
            "px": "close",
            "v": "volume",
        },
    )
    assert validate(loaded).ok
    assert loaded["tic"].nunique() == 2
    pd.testing.assert_series_equal(
        loaded.sort_values(["tic", "date"])["close"].reset_index(drop=True),
        canonical.sort_values(["tic", "date"])["close"].reset_index(drop=True),
        check_names=False,
    )


def test_loader_yahoo_format(tmp_path: Path):
    canonical = make_synthetic_canonical(
        start="2020-01-02", end="2020-02-28", tickers=["AAPL", "XOM"], seed=6
    )
    path = tmp_path / "yahoo.csv"
    to_yahoo_format(canonical).to_csv(path, index=False)
    loaded = load_csv(path, adapter="yahoo")
    assert validate(loaded).ok, validate(loaded).summary()
    assert set(loaded["tic"]) <= set(canonical["tic"])
    assert "adj_close" in loaded.columns
