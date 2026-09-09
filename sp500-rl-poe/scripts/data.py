#!/usr/bin/env python3
"""The data CLI. Four questions, four subcommands.

    python scripts/data.py list                         # what datasets exist
    python scripts/data.py fetch <name> [--fake]        # how to retrieve one
    python scripts/data.py status [<name>]              # where files go / are they there
    python scripts/data.py build <name> [--universe X]  # use: write the POE panel parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sp500rl.config import load_config  # noqa: E402
from sp500rl.data.datasets import (  # noqa: E402
    DatasetNotReady,
    fetch,
    get_panel,
    list_datasets,
    panel_path,
    status,
)
from sp500rl.data.universe import list_rules  # noqa: E402


def cmd_list(_: argparse.Namespace, cfg: dict) -> int:
    active = cfg.get("dataset")
    print(f"{'name':<12} {'how':<10} {'format':<10} what")
    for ds in list_datasets():
        mark = "*" if ds.name == active else " "
        print(f"{mark}{ds.name:<11} {ds.how:<10} {ds.format:<10} {ds.what}")
    print(f"\n* = configs/default.yaml dataset. Universes: {', '.join(list_rules(cfg))}")
    return 0


def cmd_status(args: argparse.Namespace, _: dict) -> int:
    rc = 0
    for name, info in status(args.name).items():
        flag = "ready" if info["ready"] else "MISSING"
        print(f"{name} [{flag}]  how={info['how']}")
        for role, f in info["files"].items():
            print(f"    {'ok ' if f['exists'] else '-- '}{role}: {f['path']}")
        rc = rc or (0 if info["ready"] else 1)
    return rc


def cmd_fetch(args: argparse.Namespace, cfg: dict) -> int:
    try:
        written = fetch(args.name, cfg, force=args.force, fake=args.fake)
    except DatasetNotReady as exc:
        print(exc)
        return 1
    for role, path in written.items():
        print(f"{role}: {path}")
    return 0


def cmd_build(args: argparse.Namespace, cfg: dict) -> int:
    try:
        panel = get_panel(args.name, args.universe, cfg, refresh=True)
    except DatasetNotReady as exc:
        print(exc)
        return 1
    rule = args.universe or cfg["universe"]["rule"]
    name = args.name or cfg["dataset"]
    print(f"wrote {panel_path(name, rule, cfg)} shape={panel.shape}")
    print(f"tickers {sorted(panel['tic'].unique())}")
    print(f"dates {panel['date'].min().date()} → {panel['date'].max().date()}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="what datasets exist").set_defaults(fn=cmd_list)

    p = sub.add_parser("status", help="where files go and whether they exist")
    p.add_argument("name", nargs="?")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("fetch", help="retrieve a dataset (or print how to)")
    p.add_argument("name", nargs="?", help="default: configs/default.yaml dataset")
    p.add_argument("--force", action="store_true", help="re-download / regenerate")
    p.add_argument("--fake", action="store_true", help="write fake files in this dataset's format")
    p.set_defaults(fn=cmd_fetch)

    p = sub.add_parser("build", help="write the POE panel parquet")
    p.add_argument("name", nargs="?", help="default: configs/default.yaml dataset")
    p.add_argument("--universe", default=None, help="list name | full_window | top_n | listed")
    p.set_defaults(fn=cmd_build)

    args = parser.parse_args()
    cfg = load_config(args.config)
    return args.fn(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
