# PortfolioOptimizationEnv contract

This document is copied from the **installed** FinRL source, not from memory.
Inspected file:

`finrl/meta/env_portfolio_optimization/env_portfolio_optimization.py`

Pinned FinRL commit (this fork, used for the inspection and smoke test):

`a91d888290915ea5be9dcbe1d6a4c826a4c6b8e6` (`github.com/Jeanjacob20/FinRL`)

Upstream `AI4Finance-Foundation/FinRL` HEAD at scaffold time:

`2334a5fe6d30629157f13c3b0319e1637e15e123`

Install:

```
pip install git+https://github.com/AI4Finance-Foundation/FinRL.git@2334a5fe6d30629157f13c3b0319e1637e15e123
```

This subdirectory project also runs against the parent checkout of this fork
(commit `a91d8882`), which contains the same `PortfolioOptimizationEnv` plus
local PPO-POE work. Prefer submodule imports so `finrl/__init__.py` does not
pull the Alpaca/train stack:

```python
from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import (
    PortfolioOptimizationEnv,
)
```

## Gym vs Gymnasium

- `class PortfolioOptimizationEnv(gym.Env)` — **OpenAI Gym**, `import gym` /
  `from gym import spaces`.
- Pinned here: `gym==0.26.2`.
- Flag `new_gym_api=False` (default): `reset()` → `obs`; `step()` →
  `(obs, reward, terminal, info)` (4-tuple).
- Flag `new_gym_api=True`: `reset()` → `(obs, info)`; `step()` →
  `(obs, reward, terminal, truncated, info)` (Gymnasium 5-tuple). The class
  still **subclasses `gym.Env`**, so Stable-Baselines3 2.x still needs a
  Gymnasium wrapper (`sp500rl.env.gymnasium_wrapper.GymnasiumPOE`).
- SB3 pin: `stable-baselines3==2.9.0` (Gymnasium). Compatible `gymnasium==1.0.0`.
- FinRL `PolicyGradient` / `EIIE` call `reset()` / `step()` with the **old**
  4-tuple API and treat `obs` as a numpy tensor. Use
  `return_last_action=False` and `new_gym_api=False` for those agents.
- SB3 PPO/SAC use `return_last_action=True` (Dict obs) + `new_gym_api=True` +
  the Gymnasium wrapper + `MultiInputPolicy`.

POE imports `quantstats` at module load. Pin `quantstats` even though this
repo implements metrics in `sp500rl.eval.metrics`.

## Constructor signature

```python
def __init__(
    self,
    df,
    initial_amount,
    order_df=True,
    return_last_action=False,
    normalize_df="by_previous_time",
    reward_scaling=1,
    comission_fee_model="trf",
    comission_fee_pct=0,
    features=["close", "high", "low"],
    valuation_feature="close",
    time_column="date",
    time_format="%Y-%m-%d",
    tic_column="tic",
    tics_in_portfolio="all",
    time_window=1,
    cwd="./",
    new_gym_api=False,
):
```

### Parameters we expose through `configs/default.yaml`

| Config key | POE arg | Notes |
|---|---|---|
| `poe.features` | `features` | Column names in `df` that form the observation channels |
| `poe.valuation_feature` | `valuation_feature` | Price used for portfolio value (`"close"`) |
| `poe.time_window` | `time_window` | Lookback `t` |
| `poe.normalize_df` | `normalize_df` | `"by_previous_time"` (default), `"by_fist_time_window_value"`, `"by_COLUMN"`, callable, or `None` |
| `poe.comission_fee_model` | `comission_fee_model` | `"trf"` = Jiang et al. transaction remainder factor |
| `poe.comission_fee_pct` | `comission_fee_pct` | Placeholder `0.001`; AB has not supplied a figure |
| `poe.return_last_action` | `return_last_action` | Previous weights in the observation |
| `poe.initial_amount` | `initial_amount` | Cash at `t0` |
| `poe.cwd` | `cwd` | POE writes `results/rl/*.png` here |

## DataFrame format POE expects

Long format, one row per `(date, tic)`. Required columns:

- `date` — parsed with `time_format` (default `"%Y-%m-%d"`) then
  `pd.to_datetime`.
- `tic` — ticker symbol.
- Every name in `features` (default `close`, `high`, `low`), numeric,
  cast to `float32` inside POE.

Example from the class docstring:

```
date        high            low             close           tic
2020-12-23  0.157414        0.127420        0.136394        ADA-USD
2020-12-23  34.381519       30.074295       31.097898       BNB-USD
```

**Balanced panel.** `_get_state_and_info_from_time_index` loops
`self._tic_list` (unique tickers in `df` order) and stacks each ticker's
`time_window` rows of `features`. If a ticker is missing on a date, the
`(f, n, t)` tensor is ragged or wrong. `sp500rl.data.panel` must guarantee:

- every date has every ticker
- no NaNs in feature columns after warm-up
- `date` dtype `datetime64`
- rows sorted by `date`, then `tic`

POE sorts by `[tic, date]` when `order_df=True`.

## Observation tensor

Shape **`(f, n, t)`** = `(len(features), n_tickers, time_window)`,
channels-first, built as:

```
state = stack_over_tics(features.T)  # (f, t, n) then transpose to (f, n, t)
```

### `return_last_action=False` (Box)

`observation_space = Box(-inf, inf, shape=(f, n, t))`.

`reset` / `step` return that array.

Use this for FinRL `PolicyGradient` + `EIIE`: the policy does
`np.expand_dims(obs, axis=0)` and feeds a tensor into `Conv2d`. Last weights
come from FinRL's `PVM`, not from the env observation.

### `return_last_action=True` (Dict)

```python
{
    "state": Box(shape=(f, n, t)),
    "last_action": Box(low=0, high=1, shape=(n + 1,)),
}
```

`last_action` is the **submitted** weight vector from `_actions_memory`
(cash + assets), not the post-price-move holdings in `_final_weights`.

Use this for SB3 `MultiInputPolicy`. Do **not** use default `MlpPolicy` or
`CnnPolicy`: they flatten / assume image layout and destroy `(f, n, t)`.

## Action space

`Box(low=0, high=1, shape=(n + 1,))` — cash at index 0, then one weight per
ticker in `_tic_list` order.

If `sum(actions)` is not ~1 or any weight is negative, POE applies softmax.

## Reward and costs

- Reward is log-return `ln(V_t / V_{t-1})` times `reward_scaling`.
- `comission_fee_model="trf"`: Jiang et al. transaction remainder factor
  `mu`; `info["trf_mu"]` is the fraction of value kept after the rebalance.
- Initial allocation is 100% cash: `[1, 0, ..., 0]`.
- Price variation for cash is inserted as `1` at index 0.

## Smoke-tested FinRL portfolio-optimization imports

On the pinned commit, with `torch-geometric` installed (required because
`architectures.py` imports it at module load):

- `finrl.agents.portfolio_optimization.models.DRLAgent`
- `finrl.agents.portfolio_optimization.architectures.EIIE`
- `finrl.agents.portfolio_optimization.architectures.GPM`
- `finrl.agents.portfolio_optimization.algorithms.PolicyGradient`
