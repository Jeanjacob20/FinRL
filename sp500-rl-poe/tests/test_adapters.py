from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sp500rl.data.adapters.wrds_crsp import to_canonical
from sp500rl.data.loaders import load_csv
from sp500rl.data.schema import validate


def test_crsp_adapter_negative_prc_and_split(tmp_path: Path):
    """Negative PRC (bid/ask midpoint) and a 2-for-1 split via cfacpr/cfacshr."""

    fixture = pd.DataFrame(
        {
            "date": ["2020-01-02", "2020-01-03", "2020-01-02", "2020-01-03"],
            "permno": [10107, 10107, 14593, 14593],
            "TICKER": ["MSFT", "MSFT", "AAPL", "AAPL"],
            "prc": [-100.0, 50.0, 200.0, 100.0],
            "openprc": [-99.0, 49.5, 198.0, 99.0],
            "askhi": [101.0, 51.0, 202.0, 101.0],
            "bidlo": [98.0, 48.0, 197.0, 98.0],
            "vol": [1000.0, 2000.0, 3000.0, 4000.0],
            "cfacpr": [1.0, 2.0, 1.0, 2.0],
            "cfacshr": [1.0, 2.0, 1.0, 2.0],
        }
    )
    path = tmp_path / "crsp.csv"
    fixture.to_csv(path, index=False)
    loaded = load_csv(path, adapter="wrds_crsp")
    report = validate(loaded)
    assert report.ok, report.summary()

    msft = loaded.loc[loaded["tic"] == "MSFT"].sort_values("date")
    # Day 1: abs(-100)/1 = 100; day 2: abs(50)/2 = 25
    assert msft.iloc[0]["close"] == pytest.approx(100.0)
    assert msft.iloc[1]["close"] == pytest.approx(25.0)
    # Volume on split day: 2000 * 2
    assert msft.iloc[1]["volume"] == pytest.approx(4000.0)


def test_crsp_to_canonical_direct():
    df = pd.DataFrame(
        {
            "caldt": ["2020-06-01"],
            "permno": [1],
            "prc": [-10.0],
            "vol": [5.0],
        }
    )
    out = to_canonical(df)
    assert out.iloc[0]["close"] == 10.0
    assert out.iloc[0]["tic"] == "1"
    assert out.iloc[0]["open"] == 10.0
