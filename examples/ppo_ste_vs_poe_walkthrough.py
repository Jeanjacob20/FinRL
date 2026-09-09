"""Walk through original PPO vs PPO-STE vs PPO-POE on the two-asset toy example.

Run from the repo root:

    python examples/ppo_ste_vs_poe_walkthrough.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_comparison_module():
    """Load the comparison module without importing ``finrl.__init__``.

    ``finrl/__init__.py`` pulls in gymnasium and the train/test stack. This
    walkthrough is pure numpy so it stays runnable with that extra install.
    """

    path = (
        Path(__file__).resolve().parents[1]
        / "finrl"
        / "agents"
        / "portfolio_optimization"
        / "ppo_ste_vs_poe.py"
    )
    spec = importlib.util.spec_from_file_location("ppo_ste_vs_poe", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    comparison = _load_comparison_module()
    print(comparison.format_walkthrough(comparison.run_asset_allocation_example()))


if __name__ == "__main__":
    main()
