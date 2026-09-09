# AB_finRL extract contract

The previous `AB_finRL` WRDS pipeline is **not in this checkout** (private /
corrupted store). This project is wired to its **CSV outputs**, not to live
WRDS. Drop the two files below into `data/raw/` and the rest of `sp500rl`
(universe → features → POE) is unchanged.

| File | Role | Default path |
|---|---|---|
| `wrds_tickers` | Full S&P 500 (or study) ticker / PERMNO universe, optionally with membership dates | `data/raw/wrds_tickers.csv` |
| `wrds_processed` | Daily CRSP prices after the AB extract (one row per PERMNO × date) | `data/raw/wrds_processed.csv` |

```
python scripts/build_dataset.py --config configs/default.yaml \
  --input data/raw/wrds_processed.csv \
  --tickers data/raw/wrds_tickers.csv \
  --adapter wrds_crsp \
  --universe sandbox
```

If those paths are missing, `USE_SYNTHETIC=1` writes **the same two filenames**
in CRSP-shaped columns so the native path still runs.

Do not commit real CRSP extracts (license). They stay gitignored under `data/raw/`.

## `wrds_tickers` columns

Long or one-row-per-PERMNO. Required: a PERMNO and a ticker.

| Column (any alias) | Meaning |
|---|---|
| `permno` / `PERMNO` | CRSP security id |
| `ticker` / `TICKER` / `tic` | Symbol |
| `start` / `start_date` / `namedt` / `mbrstartdt` / `start` | Membership / name start (optional) |
| `ending` / `end` / `end_date` / `nameenddt` / `mbrenddt` | Membership / name end (optional) |
| `comnam` | Company name (kept, ignored downstream) |

This is the join of `crsp.dsp500list` / `dsp500list_v2` with
`crsp.stocknames` / `dsenames` in the AB extract. Extra columns are kept.

## `wrds_processed` columns

Daily CRSP stock file (`crsp.dsf` / `dsf_v2`) plus a ticker when the AB
pipeline already mapped PERMNO.

| Column (any alias) | Meaning |
|---|---|
| `date` / `caldt` / `dlycaldt` | Session date |
| `permno` | CRSP id |
| `TICKER` / `ticker` | Symbol if already joined |
| `prc` | Close; **negative = bid/ask midpoint** |
| `openprc` | Open |
| `askhi` | High |
| `bidlo` | Low |
| `vol` | Volume |
| `cfacpr` / `cfacshr` | Cumulative split factors |
| `ret` | CRSP return (optional, not required by POE) |

`sp500rl.data.adapters.wrds_crsp` then: `abs(prc)`, split-adjust with
`cfacpr`/`cfacshr`, `prc`→`close`, `permno`→`tic` via `wrds_tickers` when the
price file has no ticker.

## Point-in-time vs POE

`wrds_tickers` membership dates are a **point-in-time S&P 500**. POE still
needs a fixed ticker set. After ingest:

* `universe.rule: wrds_tickers` — keep names listed in the tickers file that
  also appear in `wrds_processed`, then take the **full-window survivors**
  (survivorship-biased; report it).
* `sandbox` / `top_n` still work on the canonical frame.

## What this repo does **not** do

No `wrds` Python package, no `db.describe_table`, no live CRSP pull. That
stays in AB_finRL. Only the two CSVs cross the boundary.
