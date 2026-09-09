# AB_finRL extract contract (dataset `ab_wrds`)

The `AB_finRL` WRDS pipeline is **not in this checkout**. This project reads
its **two CSV outputs**; nothing talks to WRDS here.

| Question | Answer |
|---|---|
| What | Daily CRSP OHLCV for the S&P 500 universe + the PERMNO ↔ ticker membership file |
| How | Run the WRDS export in AB_finRL (or ask for the two files); no credentials needed here |
| Where | `data/raw/ab_wrds/wrds_processed.csv` and `data/raw/ab_wrds/wrds_tickers.csv` (gitignored) |
| Use | `python scripts/data.py build ab_wrds --universe ab_finrl` or `get_panel("ab_wrds", "ab_finrl")` |

`python scripts/data.py status ab_wrds` shows the two paths and whether they
are present. `python scripts/data.py fetch ab_wrds --fake` writes CRSP-shaped
fakes at those paths so the path can be tested before the real files arrive.

## `wrds_tickers` columns

Long or one-row-per-PERMNO. Required: a PERMNO and a ticker.

| Column (any alias) | Meaning |
|---|---|
| `permno` / `PERMNO` | CRSP security id |
| `ticker` / `TICKER` / `tic` | Symbol |
| `start` / `start_date` / `namedt` / `mbrstartdt` | Membership / name start (optional) |
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
price file has no ticker. Rows outside a PERMNO's membership window are
dropped when the tickers file carries dates.

## Point-in-time vs POE

`wrds_tickers` membership dates are a **point-in-time S&P 500**. POE still
needs a fixed ticker set (`universe.rule` in `configs/default.yaml`):

* `ab_finrl` — the 10 names selected in AB_finRL:
  `AAPL, MSFT, JNJ, JPM, XOM, PG, HD, UNH, CAT, DIS` (`universe.lists.ab_finrl`).
* `listed` — names in `wrds_tickers` that also appear in `wrds_processed`, then
  the **full-window survivors** (survivorship-biased; report it).
* `full_window` / `top_n` — see the README.

## What this repo does **not** do

No `wrds` Python package, no `db.describe_table`, no live CRSP pull. That
stays in AB_finRL. Only the two CSVs cross the boundary.
