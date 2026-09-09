"""Shared synthetic frames for unit tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sp500rl.data.synthetic import make_synthetic_canonical, to_wide_close, to_yahoo_format

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def canonical_small() -> pd.DataFrame:
    """Two years is overkill for loader tests; ~80 business days, 3 tickers."""

    return make_synthetic_canonical(
        start="2020-01-02",
        end="2020-04-30",
        tickers=["AAPL", "MSFT", "JPM"],
        seed=0,
    )


@pytest.fixture
def tmp_csv_dir(tmp_path: Path) -> Path:
    return tmp_path
