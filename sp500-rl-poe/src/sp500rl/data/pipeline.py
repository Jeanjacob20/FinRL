"""End-to-end: raw file → canonical → universe → features → POE panel."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from sp500rl.config import load_config, project_root
from sp500rl.data.features import add_features
from sp500rl.data.loaders import load_table
from sp500rl.data.panel import panel_from_config
from sp500rl.data.synthetic import write_synthetic
from sp500rl.data.universe import select_universe


def build_panel_from_file(
    input_path: str | Path | None,
    cfg: dict[str, Any] | None = None,
    adapter: str | None = None,
    column_map: Mapping[str, str] | None = None,
    universe: str | None = None,
    use_synthetic: bool = False,
) -> pd.DataFrame:
    """Run the full data pipeline.

    Parameters
    ----------
    input_path
        CSV/parquet. If missing and ``use_synthetic`` is True, a fake
        canonical file is written first.
    cfg
        YAML config dict. Loaded from ``configs/default.yaml`` when None.
    adapter
        Source adapter name; defaults to ``cfg['adapter']``.
    column_map
        Optional source→canonical rename.
    universe
        Override ``cfg['universe']['rule']``.
    use_synthetic
        Generate synthetic data when the input file is absent.

    Returns
    -------
    pd.DataFrame
        Balanced POE panel. Columns ``date, tic, <poe.features...>``.
        Shape ``(D * n, 2 + f)``.
    """

    cfg = cfg if cfg is not None else load_config()
    root = project_root()
    adapter = adapter or cfg.get("adapter") or "generic"
    column_map = column_map if column_map is not None else cfg.get("column_map")
    dates = cfg.get("dates", {})

    if input_path is None:
        input_path = root / cfg.get("paths", {}).get("raw_dir", "data/raw") / "synthetic_canonical.csv"
    input_path = Path(input_path)
    if not input_path.is_absolute():
        input_path = (root / input_path).resolve()

    if not input_path.exists():
        if not use_synthetic:
            raise FileNotFoundError(
                f"{input_path} not found. Pass --input or set USE_SYNTHETIC=1."
            )
        raw_dir = input_path.parent
        write_synthetic(
            raw_dir,
            start=str(dates.get("start", "2019-01-02")),
            end=str(dates.get("end", "2023-12-29")),
            seed=int(cfg.get("seed", 42)),
        )
        if not input_path.exists():
            input_path = raw_dir / "synthetic_canonical.csv"

    prices = load_table(input_path, adapter=adapter, column_map=column_map)
    prices = select_universe(prices, cfg, rule=universe)
    feat_cfg = cfg.get("features", {})
    featured = add_features(
        prices,
        indicators=feat_cfg.get("indicators", ("macd", "rsi_30", "cci_30", "dx_30")),
        log_return=bool(feat_cfg.get("log_return", True)),
        rolling_vol_window=int(feat_cfg.get("rolling_vol_window", 20)),
    )
    return panel_from_config(featured, cfg)
