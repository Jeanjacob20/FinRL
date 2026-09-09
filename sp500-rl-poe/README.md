# sp500-rl-poe

Data-source-agnostic pipeline into FinRL's **PortfolioOptimizationEnv (POE)**.
Any well-formed CSV/parquet that satisfies the canonical schema can be turned
into a balanced panel and handed to every agent (FinRL EIIE / PolicyGradient
or SB3 PPO / SAC). Adding a source never requires touching the environment,
agents, or notebooks — drop a file in `data/raw/` or write a small adapter.

This directory is a self-contained project inside the FinRL fork. Existing
FinRL sources are not modified.

## Setup

Python ≥ 3.10. From `sp500-rl-poe/`:

```bash
pip install -e ".[dev]"
# FinRL: either this parent checkout, or:
pip install git+https://github.com/AI4Finance-Foundation/FinRL.git@2334a5fe6d30629157f13c3b0319e1637e15e123
```

### Pinned FinRL / gym / SB3

| Piece | Pin | Notes |
|---|---|---|
| FinRL (upstream) | `2334a5fe6d30629157f13c3b0319e1637e15e123` | `AI4Finance-Foundation/FinRL` |
| FinRL (this fork) | `a91d888290915ea5be9dcbe1d6a4c826a4c6b8e6` | POE source used to write `docs/poe_contract.md` |
| gym | `0.26.2` | POE subclasses `gym.Env` |
| gymnasium | `1.0.0` | SB3 2.x |
| stable-baselines3 | `2.9.0` | Gymnasium; use `MultiInputPolicy` + `EIIEFeaturesExtractor` |
| torch-geometric | `2.8.x` | Required to import `architectures.EIIE` / `GPM` (module-level import) |
| quantstats | any recent | POE imports it at module load |

Import POE via submodules (`sp500rl.finrl_bootstrap`) so `finrl/__init__.py`
does not pull Alpaca/train. Contract details: [`docs/poe_contract.md`](docs/poe_contract.md).

There is **no live WRDS client** in this repo.

## Data: four questions

Every data source is one entry in [`configs/datasets.yaml`](configs/datasets.yaml).
The entry says what it is, how to get it, where the files go, and how the
columns are read. Nothing downstream knows which dataset it is looking at.

| Question | Command | Python |
|---|---|---|
| **What** dataset should I retrieve? | `python scripts/data.py list` | `list_datasets()` |
| **How** do I retrieve it? | `python scripts/data.py fetch <name>` | `fetch(name)` |
| **Where** do I place it? | `python scripts/data.py status` | `status()` |
| **Where** do I use it? | `python scripts/data.py build <name> --universe ab_finrl` | `get_panel(name, universe)` |

`fetch` downloads / generates when it can (`yfinance`, `synthetic`, `command`).
For a manual dataset it prints the instruction and the exact paths. `build` /
`get_panel` write `data/processed/<dataset>__<universe>.parquet`, the object
every notebook hands to POE.

```python
from sp500rl.data import get_panel
panel = get_panel("ab_wrds", universe="ab_finrl")   # POE-ready, cached
```

Registered datasets:

| Name | What | How | Where (under `sp500-rl-poe/`) | Format |
|---|---|---|---|---|
| `ab_wrds` | AB_finRL WRDS/CRSP extract | manual: copy the two CSVs | `data/raw/ab_wrds/wrds_processed.csv`, `data/raw/ab_wrds/wrds_tickers.csv` | `wrds_crsp` |
| `yahoo` | Yahoo Finance daily OHLCV for the active universe list | `fetch yahoo` (yfinance) | `data/raw/yahoo/prices.csv` | `yahoo` |
| `synthetic` | Fake GBM prices | `fetch synthetic` | `data/raw/synthetic/prices.csv` | `canonical` |

`configs/default.yaml` has one knob, `dataset:`, that notebooks read. It is
`synthetic` until the AB CSVs are in place; then set it to `ab_wrds`.
`data/raw/` is gitignored — real CRSP files never get committed.
`fetch ab_wrds --fake` writes CRSP-shaped fakes at the AB paths so that path
can be exercised before the real files arrive. Column contract for the AB
files: [`docs/ab_finrl_contract.md`](docs/ab_finrl_contract.md).

### Adding a dataset

Add a block to `configs/datasets.yaml`:

```yaml
  berkeley:
    what: Daily prices from the Berkeley vendor feed.
    how: manual                  # or yfinance | synthetic | command (+ command: ...)
    how_to: Export from ... and save as the path below.
    files:
      prices: data/raw/berkeley/prices.csv
    format: canonical            # canonical | yahoo | wrds_crsp
    column_map: {Datum: date, Symbol: tic}   # only if the headers are unusual
```

That is the whole change. If the file needs cleaning no `column_map` can
express, write `src/sp500rl/data/adapters/<name>.py` with
`to_canonical(df) -> df`, register it in `ADAPTERS`, and use that name as
`format`.

## Canonical schema

Required columns (long format): `date`, `tic`, `open`, `high`, `low`, `close`,
`volume`. Optional: `adj_close`, `market_cap`, `sector`, `permno`.

`sp500rl.data.schema.validate` **reports** problems (duplicates, non-positive
prices, bad dtypes). It does not silently fix them. Adapters are the place
where CRSP negatives and split factors become canonical.

See [`src/sp500rl/data/schema.py`](src/sp500rl/data/schema.py).

## Universe selection (survivorship bias)

POE needs a **fixed ticker set on every date**. That is not a point-in-time
S&P 500. `universe.rule` in `configs/default.yaml`:

| Rule | Behaviour | Bias |
|---|---|---|
| a name in `universe.lists` (`ab_finrl`, `sandbox`, …) | Fixed tickers. `ab_finrl` = the 10 names selected in AB_finRL: AAPL, MSFT, JNJ, JPM, XOM, PG, HD, UNH, CAT, DIS | None beyond the list |
| `full_window` | Names present on every date in the window | Survivorship: leavers/joiners dropped |
| `top_n` | Top-N by `market_cap` on the first training date (else avg dollar volume), then **held fixed** | Look-ahead / survivorship |
| `listed` | Names in the dataset's `tickers` file ∩ prices, then full-window survivors | Survivorship |

Add a list by adding a key under `universe.lists`. **This bias must be
reported** in any AB comparison of RL vs equal-weight / risk parity. It is a
property of the POE test bed, not of a particular agent.

## Pipeline

```
datasets.yaml entry → fetch → loaders (+ column_map) → adapter.to_canonical → validate
  → universe → features (no look-ahead) → balanced panel → POE
```

```bash
python scripts/data.py fetch synthetic
python scripts/data.py build synthetic --universe ab_finrl
pytest tests/
```

Missing sessions: `missing_days.policy` is `drop_ticker` or `ffill` with
`ffill_max_gap`. Never silent.

## Notebooks

From `notebooks/`. Each reads `dataset:`, `universe.rule`, dates and the seed
from `configs/default.yaml` — do not hard-code them — and gets its panel with
`get_panel(DATASET, UNIVERSE, CFG)`.

| Notebook | What it does |
|---|---|
| `00_data_pipeline_check.ipynb` | The four questions live: list, status, fetch, load, panel, plot |
| `01_sandbox_10_stocks_EIIE.ipynb` | POE + FinRL `PolicyGradient` + `EIIE` vs equal-weight and risk parity |
| `02_sandbox_10_stocks_SB3_PPO.ipynb` | SB3 PPO + `EIIEFeaturesExtractor` / `MultiInputPolicy` |
| `03_sandbox_10_stocks_SB3_SAC.ipynb` | Same protocol, SAC |
| `04_full_universe_template.ipynb` | Parameterised scaffold (dataset, rule), no results |

FinRL `PolicyGradient` needs a **Box** observation (`return_last_action=False`,
old gym 4-tuple). SB3 needs a **Dict** observation (`return_last_action=True`)
plus the Gymnasium wrapper.

## How to add an agent

All agents train **inside POE** (`sp500rl.env.make_env.make_poe`).

* **FinRL architecture** (EIIE, GPM, …): `DRLAgent.get_model("pg", model_kwargs={"policy": YourNet, "policy_kwargs": {...}})` with `return_last_action=False`.
* **SB3** (PPO / SAC / TD3): `make_poe(..., wrap_gymnasium=True, return_last_action=True)` and `MultiInputPolicy` with `sp500rl.env.extractors.EIIEFeaturesExtractor`. Do not use `MlpPolicy` / `CnnPolicy` — they flatten `(f, n, t)`.
* Roll equal-weight / risk parity / buy-and-hold through a **fresh copy of the same POE** (`sp500rl.baselines.simple`) so `comission_fee_pct` is identical.
* Report Sharpe, max DD, turnover, and TRF cost drag from `sp500rl.eval.metrics` (no pyfolio).

## Transaction costs

`poe.comission_fee_model: trf` (Jiang et al. transaction remainder factor).
`poe.comission_fee_pct: 0.001` is a **placeholder**. AllianceBernstein has not
supplied a figure.

## Layout

```
sp500-rl-poe/
├── configs/default.yaml       # dataset:, universe, dates, POE kwargs
├── configs/datasets.yaml      # what / how / where / format per dataset
├── docs/poe_contract.md
├── docs/ab_finrl_contract.md
├── src/sp500rl/data/          # datasets (registry), schema, loaders, adapters, universe, features, panel
├── src/sp500rl/env/           # make_poe, Gymnasium wrapper, SB3 extractor
├── src/sp500rl/baselines/
├── src/sp500rl/eval/
├── notebooks/
├── scripts/data.py            # list | fetch | status | build
└── tests/
```
