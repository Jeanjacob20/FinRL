from __future__ import annotations

from pathlib import Path

import pandas as pd

from sp500rl.data.ab_finrl import listed_tickers, load_ab_extract, load_wrds_tickers
from sp500rl.data.schema import validate
from sp500rl.data.synthetic import make_synthetic_canonical, to_wrds_processed, to_wrds_tickers


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
    assert listed_tickers(t_path) == ["AAPL", "MSFT"]

    out = load_ab_extract(p_path, t_path)
    report = validate(out)
    assert report.ok, report.summary()
    assert set(out["tic"]) == {"AAPL", "MSFT"}
    assert (out["close"] > 0).all()


def test_ab_extract_membership_window_filters_rows(tmp_path: Path):
    canonical = make_synthetic_canonical(
        start="2020-01-02", end="2020-03-31", tickers=["AAPL"], seed=12
    )
    to_wrds_processed(canonical).to_csv(tmp_path / "wrds_processed.csv", index=False)
    names = to_wrds_tickers(canonical)
    names["ending"] = "2020-02-15"
    names.to_csv(tmp_path / "wrds_tickers.csv", index=False)

    out = load_ab_extract(tmp_path / "wrds_processed.csv", tmp_path / "wrds_tickers.csv")
    assert out["date"].max() <= pd.Timestamp("2020-02-15")
