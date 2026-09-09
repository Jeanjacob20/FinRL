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

Copy [`.env.example`](.env.example) to `.env` if you want to override paths.
There is **no live WRDS client** in this repo.

## WRDS / CRSP data (AB_finRL extract)

This repo does **not** talk to WRDS. The native input is the two CSVs from the
AB_finRL pipeline. Column contract: [`docs/ab_finrl_contract.md`](docs/ab_finrl_contract.md).

| File | What it is | Path |
|---|---|---|
| `wrds_tickers` | Full ticker / PERMNO universe (S&P 500 membership + names) | `data/raw/wrds_tickers.csv` |
| `wrds_processed` | Daily CRSP prices after the AB extract | `data/raw/wrds_processed.csv` |
| `ab_finrl_tickers` | The 10 names selected in AB_finRL | `configs/ab_finrl_tickers.csv` |

```bash
# After you copy the two CSVs into data/raw/:
python scripts/build_dataset.py --config configs/default.yaml \
  --input data/raw/wrds_processed.csv \
  --tickers data/raw/wrds_tickers.csv \
  --adapter wrds_crsp \
  --universe ab_finrl

# Same command with no --input: those filenames are the default.
python scripts/build_dataset.py --config configs/default.yaml --universe ab_finrl
```

The AB_finRL git repo is not cloned here (and its database was unreadable).
Place the exported CSVs at the paths above. Real CRSP files stay gitignored.

Without an extract, generate **the same two filenames** in CRSP-shaped columns:

```bash
python scripts/make_synthetic.py
# or
USE_SYNTHETIC=1 python scripts/build_dataset.py --config configs/default.yaml --universe sandbox
```

That also writes Yahoo-format and wide-format CSVs so notebook `00` can prove
source-agnosticism. `universe.rule: wrds_tickers` keeps names listed in
`wrds_tickers.csv` that survive the full window (report the survivorship bias).
`universe.rule: ab_finrl` keeps the 10 names selected in AB_finRL
(`AAPL, MSFT, JNJ, JPM, XOM, PG, HD, UNH, CAT, DIS`). Override that list in
`configs/ab_finrl_tickers.csv`, `universe.ab_finrl_tickers` in YAML, or with a
`selected` flag on `wrds_tickers.csv`.

## Canonical schema

Required columns (long format): `date`, `tic`, `open`, `high`, `low`, `close`,
`volume`. Optional: `adj_close`, `market_cap`, `sector`, `permno`.

`sp500rl.data.schema.validate` **reports** problems (duplicates, non-positive
prices, bad dtypes). It does not silently fix them. Adapters are the place
where CRSP negatives and split factors become canonical.

See [`src/sp500rl/data/schema.py`](src/sp500rl/data/schema.py).

## Universe selection (survivorship bias)

POE needs a **fixed ticker set on every date**. That is not a point-in-time
S&P 500. Config `universe.rule`:

| Rule | Behaviour | Bias |
|---|---|---|
| `sandbox` | 10 hand-picked liquid names across sectors (`universe.sandbox_tickers`) | None beyond the list |
| `ab_finrl` | The 10 names selected in AB_finRL (`configs/ab_finrl_tickers.csv`) | None beyond the list |
| `full_window` | Names present on every date in the window | Survivorship: leavers/joiners dropped |
| `top_n` | Top-N by `market_cap` on the first training date (else avg dollar volume), then **held fixed** | Look-ahead / survivorship: names are chosen with information from the start of train and never replaced |
| `wrds_tickers` | Names in `wrds_tickers.csv` ∩ prices, then full-window survivors | Survivorship: leavers/joiners dropped |

**This bias must be reported** in any AB comparison of RL vs equal-weight /
risk parity. It is a property of the POE test bed, not of a particular agent.

## Pipeline

```
CSV/parquet → loaders (+ column_map) → adapter.to_canonical → validate
  → universe → features (no look-ahead) → balanced panel → POE
```

```bash
python scripts/build_dataset.py --config configs/default.yaml \
  --input data/raw/synthetic_yahoo.csv --adapter yahoo --universe sandbox
pytest tests/
```

Missing sessions: `missing_days.policy` is `drop_ticker` or `ffill` with
`ffill_max_gap`. Never silent.

## Notebooks

From `notebooks/`, with `data/processed/panel_sandbox.parquet` already built
(or the first cell of `00` will build it). Dates and the seed come from
`configs/default.yaml` — do not hard-code them.

| Notebook | What it does |
|---|---|
| `00_data_pipeline_check.ipynb` | Load CSV, validation report, panel plots; second cell loads a Yahoo-format CSV |
| `01_sandbox_10_stocks_EIIE.ipynb` | POE + FinRL `PolicyGradient` + `EIIE` vs equal-weight and risk parity |
| `02_sandbox_10_stocks_SB3_PPO.ipynb` | SB3 PPO + `EIIEFeaturesExtractor` / `MultiInputPolicy` |
| `03_sandbox_10_stocks_SB3_SAC.ipynb` | Same protocol, SAC |
| `04_full_universe_template.ipynb` | Parameterised scaffold (rule, dates, seed), no results |

FinRL `PolicyGradient` needs a **Box** observation (`return_last_action=False`,
old gym 4-tuple). SB3 needs a **Dict** observation (`return_last_action=True`)
plus the Gymnasium wrapper.

## How to add a data source

1. Write `src/sp500rl/data/adapters/<name>.py` with `to_canonical(df) -> df`.
2. Register it in `ADAPTERS` inside `src/sp500rl/data/adapters/__init__.py`.
3. Point `scripts/build_dataset.py --adapter <name>` at your file.
4. If columns are unusual, pass `column_map` in YAML instead of editing code.

The adapter output must pass `validate`.

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
├── configs/default.yaml
├── docs/poe_contract.md
├── src/sp500rl/data/          # schema, loaders, adapters, universe, features, panel
├── src/sp500rl/env/           # make_poe, Gymnasium wrapper, SB3 extractor
├── src/sp500rl/baselines/
├── src/sp500rl/eval/
├── notebooks/
├── scripts/build_dataset.py
├── scripts/make_synthetic.py
└── tests/
```
