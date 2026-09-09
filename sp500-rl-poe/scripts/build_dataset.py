#!/usr/bin/env python3
"""CLI: config + CSV → POE-ready parquet panel.

Example
-------
python scripts/build_dataset.py --config configs/default.yaml \\
    --input data/raw/wrds_processed.csv --tickers data/raw/wrds_tickers.csv \\
    --universe sandbox
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sp500rl.config import load_config, project_root  # noqa: E402
from sp500rl.data.pipeline import build_panel_from_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a POE-ready parquet panel.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument(
        "--input",
        default=None,
        help="Price CSV/parquet. Default: data/raw/wrds_processed.csv (AB_finRL extract).",
    )
    parser.add_argument(
        "--tickers",
        default=None,
        help="AB_finRL wrds_tickers.csv (PERMNO ↔ ticker universe).",
    )
    parser.add_argument("--adapter", default=None, help="generic | wrds_crsp | yahoo | ab_finrl")
    parser.add_argument(
        "--universe",
        default=None,
        help="sandbox | ab_finrl | full_window | top_n | wrds_tickers",
    )
    parser.add_argument("--output", default=None, help="Output parquet path")
    parser.add_argument(
        "--use-synthetic",
        action="store_true",
        help="Write a fake canonical CSV if --input is missing (or set USE_SYNTHETIC=1).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    use_synthetic = args.use_synthetic or os.environ.get("USE_SYNTHETIC", "0") == "1"
    panel = build_panel_from_file(
        args.input,
        cfg=cfg,
        adapter=args.adapter,
        universe=args.universe,
        use_synthetic=use_synthetic,
        tickers_path=args.tickers,
    )

    processed = project_root() / cfg.get("paths", {}).get("processed_dir", "data/processed")
    processed.mkdir(parents=True, exist_ok=True)
    rule = args.universe or cfg.get("universe", {}).get("rule", "panel")
    out = Path(args.output) if args.output else processed / f"panel_{rule}.parquet"
    if not out.is_absolute():
        out = project_root() / out
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out, index=False)
    print(f"wrote {out} shape={panel.shape} tickers={sorted(panel['tic'].unique())}")
    print(f"date range {panel['date'].min().date()} → {panel['date'].max().date()}")


if __name__ == "__main__":
    main()
