"""Dataset registry: the one place that answers the four data questions.

* **What** dataset?      → :func:`list_datasets` / ``configs/datasets.yaml``
* **How** to retrieve?   → :func:`fetch` (runs it, or prints ``how_to``)
* **Where** to place?    → :func:`status` (expected paths, present / missing)
* **Where** to use?      → :func:`get_panel` (POE-ready panel, cached parquet)

Every dataset is one YAML entry (``what``, ``how``, ``how_to``, ``files``,
``format``, optional ``column_map``). Nothing downstream (universe, features,
panel, env, agents) knows which dataset it is looking at.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from sp500rl.config import load_config, project_root
from sp500rl.data.ab_finrl import listed_tickers, load_ab_extract
from sp500rl.data.features import add_features
from sp500rl.data.loaders import load_table
from sp500rl.data.panel import panel_from_config
from sp500rl.data.synthetic import write_fake
from sp500rl.data.universe import list_tickers, select_universe, ticker_lists

FORMAT_TO_ADAPTER: dict[str, str] = {
    "canonical": "generic",
    "generic": "generic",
    "yahoo": "yahoo",
    "wrds_crsp": "wrds_crsp",
    "ab_finrl": "wrds_crsp",
}
AUTO_HOW: frozenset[str] = frozenset({"synthetic", "yfinance", "command"})


@dataclass(frozen=True)
class Dataset:
    """One ``datasets.yaml`` entry with paths resolved to absolute ``Path``."""

    name: str
    what: str
    how: str
    how_to: str
    files: dict[str, Path]
    format: str
    column_map: dict[str, str] | None = None
    command: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def prices(self) -> Path:
        return self.files["prices"]

    @property
    def tickers(self) -> Path | None:
        return self.files.get("tickers")

    def missing(self) -> list[Path]:
        return [p for p in self.files.values() if not p.exists()]

    def is_ready(self) -> bool:
        return not self.missing()


class DatasetNotReady(FileNotFoundError):
    """Files are missing and cannot be fetched automatically."""


# --------------------------------------------------------------------------- registry


def registry_path() -> Path:
    return project_root() / "configs" / "datasets.yaml"


def _resolve(rel: str | Path) -> Path:
    path = Path(rel)
    return path if path.is_absolute() else project_root() / path


def load_registry(path: str | Path | None = None) -> dict[str, Dataset]:
    """Parse ``configs/datasets.yaml`` into :class:`Dataset` objects."""

    reg_path = Path(path) if path is not None else registry_path()
    with reg_path.open() as fh:
        raw = yaml.safe_load(fh) or {}
    entries = raw.get("datasets") or {}
    out: dict[str, Dataset] = {}
    for name, spec in entries.items():
        spec = dict(spec or {})
        files_raw = spec.pop("files", None) or {}
        if "prices" not in files_raw:
            raise ValueError(f"datasets.yaml: {name!r} needs files.prices")
        files = {role: _resolve(p) for role, p in files_raw.items()}
        fmt = str(spec.pop("format", "canonical")).lower()
        if fmt not in FORMAT_TO_ADAPTER:
            raise ValueError(
                f"datasets.yaml: {name!r} format {fmt!r} not in {sorted(FORMAT_TO_ADAPTER)}"
            )
        out[str(name)] = Dataset(
            name=str(name),
            what=str(spec.pop("what", "")).strip(),
            how=str(spec.pop("how", "manual")).lower(),
            how_to=str(spec.pop("how_to", "")).strip(),
            files=files,
            format=fmt,
            column_map=spec.pop("column_map", None),
            command=spec.pop("command", None),
            extra=spec,
        )
    return out


def list_datasets() -> list[Dataset]:
    """All registered datasets, in YAML order."""

    return list(load_registry().values())


def get_dataset(name: str | None = None, cfg: Mapping[str, Any] | None = None) -> Dataset:
    """Look up a dataset. ``None`` → ``cfg['dataset']`` (``configs/default.yaml``)."""

    reg = load_registry()
    if name is None:
        cfg = cfg if cfg is not None else load_config()
        name = cfg.get("dataset")
        if not name:
            raise KeyError("configs/default.yaml has no `dataset:`; pass a name.")
    if name not in reg:
        raise KeyError(f"Unknown dataset {name!r}. Known: {sorted(reg)}")
    return reg[name]


def status(name: str | None = None) -> dict[str, dict[str, Any]]:
    """Where the files go and whether they are there.

    Returns ``{dataset: {"ready": bool, "files": {role: {"path", "exists"}}}}``
    for one dataset or, with ``None``, for all of them.
    """

    datasets = [get_dataset(name)] if name else list_datasets()
    report: dict[str, dict[str, Any]] = {}
    for ds in datasets:
        report[ds.name] = {
            "what": ds.what,
            "how": ds.how,
            "ready": ds.is_ready(),
            "files": {
                role: {"path": str(path), "exists": path.exists()} for role, path in ds.files.items()
            },
        }
    return report


# --------------------------------------------------------------------------- fetch


def _fetch_tickers(cfg: Mapping[str, Any]) -> list[str]:
    rule = str(cfg.get("universe", {}).get("rule", "ab_finrl")).lower()
    lists = ticker_lists(dict(cfg))
    if rule in lists:
        return lists[rule]
    return list_tickers("ab_finrl", dict(cfg))


def _fetch_yfinance(ds: Dataset, cfg: Mapping[str, Any]) -> dict[str, Path]:
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise DatasetNotReady(f"{ds.name}: pip install yfinance  ({ds.how_to})") from exc

    dates = cfg.get("dates", {})
    tickers = _fetch_tickers(cfg)
    raw = yf.download(
        tickers,
        start=dates.get("start"),
        end=dates.get("end"),
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=True,
    )
    frames = []
    for tic in tickers:
        sub = raw[tic] if isinstance(raw.columns, pd.MultiIndex) else raw
        sub = sub.dropna(how="all").reset_index()
        sub["tic"] = tic
        frames.append(sub)
    out = pd.concat(frames, ignore_index=True)
    ds.prices.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(ds.prices, index=False)
    return {"prices": ds.prices}


def fetch(
    name: str | None = None,
    cfg: Mapping[str, Any] | None = None,
    *,
    force: bool = False,
    fake: bool = False,
) -> dict[str, Path]:
    """Retrieve a dataset's files into their ``files`` paths.

    * ``how: synthetic`` → generate
    * ``how: yfinance``  → download via ``yfinance`` for the active universe list
    * ``how: command``   → run ``command`` in the project root
    * ``how: manual``    → raise :class:`DatasetNotReady` with ``how_to`` + paths
    * ``fake=True``      → write fake files in this dataset's own ``format``
      (lets the ``ab_wrds`` path run before the real CSVs exist)

    Returns role → path for the files written. No-op when already present
    unless ``force``.
    """

    cfg = dict(cfg) if cfg is not None else load_config()
    ds = get_dataset(name, cfg)
    if ds.is_ready() and not force:
        return dict(ds.files)

    dates = cfg.get("dates", {})
    if fake or ds.how == "synthetic":
        return write_fake(
            ds.files,
            fmt=ds.format,
            start=str(dates.get("start", "2019-01-02")),
            end=str(dates.get("end", "2023-12-29")),
            tickers=_fetch_tickers(cfg) if ds.how != "synthetic" else None,
            seed=int(cfg.get("seed", 42)),
        )
    if ds.how == "yfinance":
        return _fetch_yfinance(ds, cfg)
    if ds.how == "command":
        if not ds.command:
            raise ValueError(f"{ds.name}: how=command needs a `command:` line")
        ds.prices.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(shlex.split(ds.command), cwd=project_root(), check=True)
        missing = ds.missing()
        if missing:
            raise DatasetNotReady(f"{ds.name}: command ran but did not write {missing}")
        return dict(ds.files)

    lines = [f"Dataset {ds.name!r} is not on disk and must be placed by hand."]
    if ds.how_to:
        lines.append(ds.how_to)
    lines.append("Expected files:")
    lines += [f"  {role}: {path}" for role, path in ds.files.items()]
    lines.append(f"Or run `python scripts/data.py fetch {ds.name} --fake` to test the path.")
    raise DatasetNotReady("\n".join(lines))


# --------------------------------------------------------------------------- use


def load_prices(
    name: str | None = None,
    cfg: Mapping[str, Any] | None = None,
    *,
    auto_fetch: bool = True,
) -> pd.DataFrame:
    """Load one dataset as the canonical frame ``date, tic, open, high, low, close, volume``.

    Missing files are fetched when the dataset is auto-retrievable; manual
    datasets raise :class:`DatasetNotReady` with the instructions.
    """

    cfg = dict(cfg) if cfg is not None else load_config()
    ds = get_dataset(name, cfg)
    if not ds.is_ready():
        if ds.how in AUTO_HOW and not auto_fetch:
            raise DatasetNotReady(f"{ds.name}: files missing {ds.missing()}; run fetch first.")
        # Manual datasets raise DatasetNotReady here with how_to and the expected paths.
        fetch(ds.name, cfg)
    if ds.format in {"wrds_crsp", "ab_finrl"} and ds.tickers is not None:
        return load_ab_extract(ds.prices, ds.tickers)
    return load_table(ds.prices, adapter=FORMAT_TO_ADAPTER[ds.format], column_map=ds.column_map)


def panel_path(name: str, universe: str, cfg: Mapping[str, Any] | None = None) -> Path:
    """``data/processed/<dataset>__<universe>.parquet``."""

    cfg = dict(cfg) if cfg is not None else load_config()
    processed = _resolve(cfg.get("paths", {}).get("processed_dir", "data/processed"))
    return processed / f"{name}__{universe}.parquet"


def build_panel(
    prices: pd.DataFrame,
    cfg: Mapping[str, Any],
    universe: str | None = None,
    listed: list[str] | None = None,
) -> pd.DataFrame:
    """Canonical prices → universe → features → balanced POE panel (no I/O)."""

    cfg = dict(cfg)
    selected = select_universe(prices, cfg, rule=universe, listed=listed)
    feat_cfg = cfg.get("features", {})
    featured = add_features(
        selected,
        indicators=feat_cfg.get("indicators", ("macd", "rsi_30", "cci_30", "dx_30")),
        log_return=bool(feat_cfg.get("log_return", True)),
        rolling_vol_window=int(feat_cfg.get("rolling_vol_window", 20)),
    )
    return panel_from_config(featured, cfg)


def get_panel(
    name: str | None = None,
    universe: str | None = None,
    cfg: Mapping[str, Any] | None = None,
    *,
    refresh: bool = False,
    cache: bool = True,
) -> pd.DataFrame:
    """The POE-ready panel for ``(dataset, universe)``.

    Parameters
    ----------
    name
        Dataset name; ``None`` → ``cfg['dataset']``.
    universe
        Universe rule / list name; ``None`` → ``cfg['universe']['rule']``.
    cfg
        YAML config dict; ``None`` → ``configs/default.yaml``.
    refresh
        Rebuild even if the cached parquet exists.
    cache
        Write / read ``data/processed/<dataset>__<universe>.parquet``.

    Returns
    -------
    pd.DataFrame
        Columns ``date, tic, <poe.features...>``. Shape ``(D * n, 2 + f)``.
    """

    cfg = dict(cfg) if cfg is not None else load_config()
    ds = get_dataset(name, cfg)
    rule = (universe or cfg.get("universe", {}).get("rule") or "ab_finrl").lower()
    out = panel_path(ds.name, rule, cfg)

    if cache and out.exists() and not refresh:
        panel = pd.read_parquet(out)
        panel["date"] = pd.to_datetime(panel["date"])
        return panel

    prices = load_prices(ds.name, cfg)
    listed = listed_tickers(ds.tickers) if ds.tickers is not None and ds.tickers.exists() else None
    panel = build_panel(prices, cfg, universe=rule, listed=listed)
    if cache:
        out.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(out, index=False)
    return panel
