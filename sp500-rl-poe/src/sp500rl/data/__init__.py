"""Data layer public surface.

    from sp500rl.data import get_panel, load_prices, fetch, status, list_datasets

Datasets are declared in ``configs/datasets.yaml``; universes in
``configs/default.yaml``. See :mod:`sp500rl.data.datasets`.
"""

from __future__ import annotations

from sp500rl.data.datasets import (
    Dataset,
    DatasetNotReady,
    build_panel,
    fetch,
    get_dataset,
    get_panel,
    list_datasets,
    load_prices,
    panel_path,
    status,
)

__all__ = [
    "Dataset",
    "DatasetNotReady",
    "build_panel",
    "fetch",
    "get_dataset",
    "get_panel",
    "list_datasets",
    "load_prices",
    "panel_path",
    "status",
]
