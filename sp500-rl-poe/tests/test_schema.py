from __future__ import annotations

import pandas as pd
import pytest

from sp500rl.data.schema import validate
from sp500rl.data.synthetic import make_synthetic_canonical


def test_canonical_synthetic_validates():
    df = make_synthetic_canonical(start="2020-01-02", end="2020-03-31", seed=1)
    report = validate(df)
    assert report.ok, report.summary()
    assert report.n_tickers == 10


def test_validate_reports_negative_price_without_fixing():
    df = make_synthetic_canonical(
        start="2020-01-02", end="2020-01-10", tickers=["AAPL"], seed=2
    )
    dirty = df.copy()
    dirty.loc[dirty.index[0], "close"] = -1.0
    report = validate(dirty)
    assert not report.ok
    assert any(i.code == "non_positive_price" for i in report.issues)
    assert dirty.loc[dirty.index[0], "close"] == -1.0


def test_validate_reports_duplicate_date_tic():
    df = make_synthetic_canonical(
        start="2020-01-02", end="2020-01-10", tickers=["AAPL"], seed=3
    )
    doubled = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    report = validate(doubled)
    assert any(i.code == "duplicate_date_tic" for i in report.issues)
