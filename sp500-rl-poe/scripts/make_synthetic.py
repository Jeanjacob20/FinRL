#!/usr/bin/env python3
"""Write synthetic canonical / Yahoo / wide CSVs under data/raw/.

Respects USE_SYNTHETIC=1 as a no-op-if-exists flag when --force is not set.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sp500rl.config import load_config, project_root  # noqa: E402
from sp500rl.data.synthetic import write_synthetic  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic price CSVs.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dates = cfg.get("dates", {})
    out_dir = Path(args.out_dir) if args.out_dir else project_root() / cfg.get("paths", {}).get("raw_dir", "data/raw")
    target = out_dir / "synthetic_canonical.csv"
    if target.exists() and not args.force:
        print(f"{target} already exists (pass --force to overwrite)")
        return
    paths = write_synthetic(
        out_dir,
        start=str(dates.get("start", "2019-01-02")),
        end=str(dates.get("end", "2023-12-29")),
        seed=int(cfg.get("seed", 42)),
    )
    for kind, path in paths.items():
        print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
