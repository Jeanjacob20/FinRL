"""Import FinRL portfolio-optimization modules without running ``finrl/__init__.py``.

``finrl/__init__.py`` imports the train/trade stack (Alpaca, etc.), which this
project does not need. This helper registers namespace packages from either:

1. an already-installed ``finrl`` distribution
2. ``FINRL_ROOT``
3. the parent of this subdirectory project (the FinRL fork checkout)
"""

from __future__ import annotations

import sys
import types
from pathlib import Path


def _parent_finrl_root() -> Path | None:
    # sp500rl/__file__ → src/sp500rl → src → sp500-rl-poe → FinRL repo root
    candidate = Path(__file__).resolve().parents[3]
    if (candidate / "finrl" / "meta" / "env_portfolio_optimization").is_dir():
        return candidate
    return None


def bootstrap_finrl(finrl_root: str | Path | None = None) -> Path | None:
    """Make ``finrl.meta...`` and ``finrl.agents...`` importable.

    Returns the root used, or None if ``finrl`` is already a real package.
    """

    if finrl_root is None:
        import os

        env_root = os.environ.get("FINRL_ROOT")
        finrl_root = env_root if env_root else _parent_finrl_root()

    # If a proper installation is already on sys.path, leave it.
    if "finrl" in sys.modules and getattr(sys.modules["finrl"], "__file__", None):
        file_ = Path(sys.modules["finrl"].__file__ or "")
        if file_.name == "__init__.py":
            # Real package; still may pull Alpaca. Prefer namespace overlay
            # only when we have a local root.
            pass

    if finrl_root is None:
        return None

    root = Path(finrl_root)
    packages = {
        "finrl": root / "finrl",
        "finrl.meta": root / "finrl/meta",
        "finrl.meta.preprocessor": root / "finrl/meta/preprocessor",
        "finrl.meta.env_portfolio_optimization": root
        / "finrl/meta/env_portfolio_optimization",
        "finrl.agents": root / "finrl/agents",
        "finrl.agents.portfolio_optimization": root
        / "finrl/agents/portfolio_optimization",
    }
    for name, path in packages.items():
        if not path.is_dir():
            continue
        existing = sys.modules.get(name)
        if existing is not None and getattr(existing, "__path__", None):
            continue
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        module.__package__ = name
        sys.modules[name] = module
    return root


def import_poe():
    """Return :class:`PortfolioOptimizationEnv`."""

    bootstrap_finrl()
    from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import (
        PortfolioOptimizationEnv,
    )

    return PortfolioOptimizationEnv


def import_pg_stack():
    """Return ``(DRLAgent, EIIE, GPM, PolicyGradient)``."""

    bootstrap_finrl()
    from finrl.agents.portfolio_optimization.architectures import EIIE, GPM
    from finrl.agents.portfolio_optimization.algorithms import PolicyGradient
    from finrl.agents.portfolio_optimization.models import DRLAgent

    return DRLAgent, EIIE, GPM, PolicyGradient
