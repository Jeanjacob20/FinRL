"""Load YAML configs and resolve paths relative to the project root.

Outputs
-------
dict
    Nested configuration. Path values remain strings; use ``project_root()``
    to resolve them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def package_root() -> Path:
    """Return ``.../sp500-rl-poe`` (the directory that contains ``src/``)."""

    return Path(__file__).resolve().parents[2]


def project_root() -> Path:
    """Alias for :func:`package_root`."""

    return package_root()


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config.

    Parameters
    ----------
    path
        Config file. Defaults to ``<project>/configs/default.yaml``.

    Returns
    -------
    dict
        Parsed YAML. Expected keys include ``dates``, ``universe``, ``poe``.
    """

    cfg_path = Path(path) if path is not None else project_root() / "configs" / "default.yaml"
    with cfg_path.open() as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {cfg_path} did not parse to a mapping.")
    return cfg
