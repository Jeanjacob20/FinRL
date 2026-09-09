"""End-to-end: raw file → canonical → universe → features → POE panel.

Native AB_finRL path: ``wrds_processed.csv`` + ``wrds_tickers.csv``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from sp500rl.config import load_config, project_root
from sp500rl.data.ab_finrl import load_ab_extract
from sp500rl.data.features import add_features
from sp500rl.data.loaders import load_table
from sp500rl.data.panel import panel_from_config
from sp500rl.data.synthetic import write_synthetic
from sp500rl.data.universe import select_universe


def _raw_dir(cfg: dict[str, Any], root: Path) -> Path:
    return root / cfg.get("paths", {}).get("raw_dir", "data/raw")


def resolve_ab_paths(cfg: dict[str, Any], root: Path | None = None) -> tuple[Path, Path]:
    """Default locations of ``wrds_processed`` and ``wrds_tickers``."""

    root = root or project_root()
    paths = cfg.get("paths", {})
    raw = _raw_dir(cfg, root)
    processed = Path(paths.get("wrds_processed", raw / "wrds_processed.csv"))
    tickers = Path(paths.get("wrds_tickers", raw / "wrds_tickers.csv"))
    if not processed.is_absolute():
        processed = (root / processed).resolve()
    if not tickers.is_absolute():
        tickers = (root / tickers).resolve()
    return processed, tickers


def _looks_like_ab(path: Path) -> bool:
    name = path.name.lower()
    return "wrds_processed" in name or name in {"wrds_processed.csv", "wrds_processed.parquet"}


def build_panel_from_file(
    input_path: str | Path | None,
    cfg: dict[str, Any] | None = None,
    adapter: str | None = None,
    column_map: Mapping[str, str] | None = None,
    universe: str | None = None,
    use_synthetic: bool = False,
    tickers_path: str | Path | None = None,
) -> pd.DataFrame:
    """Run the full data pipeline.

    Parameters
    ----------
    input_path
        Price CSV/parquet. ``None`` prefers ``data/raw/wrds_processed.csv``,
        then ``synthetic_canonical.csv``.
    cfg
        YAML config dict. Loaded from ``configs/default.yaml`` when None.
    adapter
        Source adapter name; defaults to ``cfg['adapter']``. Ignored when
        ingesting an AB_finRL ``wrds_processed`` file (always CRSP adapter
        + ticker join).
    column_map
        Optional source→canonical rename (non-AB files).
    universe
        Override ``cfg['universe']['rule']``.
    use_synthetic
        Generate synthetic AB-shaped CSVs when the input file is absent.
    tickers_path
        AB_finRL ``wrds_tickers`` file. Defaults to config / ``data/raw/wrds_tickers.csv``.

    Returns
    -------
    pd.DataFrame
        Balanced POE panel. Columns ``date, tic, <poe.features...>``.
        Shape ``(D * n, 2 + f)``.
    """

    cfg = cfg if cfg is not None else load_config()
    root = project_root()
    adapter = adapter or cfg.get("adapter") or "wrds_crsp"
    column_map = column_map if column_map is not None else cfg.get("column_map")
    dates = cfg.get("dates", {})
    ab_processed, ab_tickers = resolve_ab_paths(cfg, root)
    if tickers_path is not None:
        ab_tickers = Path(tickers_path)
        if not ab_tickers.is_absolute():
            ab_tickers = (root / ab_tickers).resolve()

    if input_path is None:
        if ab_processed.exists() or use_synthetic:
            input_path = ab_processed
        else:
            input_path = _raw_dir(cfg, root) / "synthetic_canonical.csv"
    input_path = Path(input_path)
    if not input_path.is_absolute():
        input_path = (root / input_path).resolve()

    if not input_path.exists():
        if not use_synthetic:
            raise FileNotFoundError(
                f"{input_path} not found. Drop wrds_processed.csv in data/raw/ "
                "or set USE_SYNTHETIC=1."
            )
        raw_dir = input_path.parent
        write_synthetic(
            raw_dir,
            start=str(dates.get("start", "2019-01-02")),
            end=str(dates.get("end", "2023-12-29")),
            seed=int(cfg.get("seed", 42)),
        )
        if not input_path.exists():
            fallback = raw_dir / "wrds_processed.csv"
            input_path = fallback if fallback.exists() else raw_dir / "synthetic_canonical.csv"

    if _looks_like_ab(input_path) or adapter in {"ab_finrl"}:
        tickers_arg = ab_tickers if ab_tickers.exists() else None
        prices = load_ab_extract(input_path, tickers_arg)
    else:
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
