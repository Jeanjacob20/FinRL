"""Compare original PPO with FinRL PPO-STE and PPO-POE.

The functions in this module replay the seven-step PPO asset-allocation
summary (actor sample → reward → critic → advantage → clipped update →
critic fit → discard) on a two-asset toy portfolio.

PPO-STE is original PPO running in ``StockTradingEnv``: the actor still
stores ``log_prob`` and the clipped surrogate is unchanged, but the
environment treats the action as a share-count delta, not as target
weights.

PPO-POE is how FinRL actually trains on ``PortfolioOptimizationEnv``:
the environment rewrites the action and reward, and the registered POE
agent (Jiang policy gradient) replaces critic / advantage / clipping /
on-policy discard. Those replacements are the places POE changes
original PPO behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Toy portfolio from the PPO asset-allocation summary
# ---------------------------------------------------------------------------

INITIAL_VALUE = 100.0
PRICES_S0 = np.array([50.0, 30.0], dtype=np.float64)
WEIGHTS_S0 = np.array([0.5, 0.5], dtype=np.float64)
MU = np.array([0.6, 0.4], dtype=np.float64)
SIGMA = np.array([0.1, 0.1], dtype=np.float64)
ACTION = np.array([0.58, 0.42], dtype=np.float64)
ASSET_RETURNS = np.array([0.04, -0.033], dtype=np.float64)
GAMMA = 0.99
V_S0 = 105.0
V_S1 = 106.0
CLIP_LOW = 0.8
CLIP_HIGH = 1.2
STE_HMAX = 100
STE_REWARD_SCALING = 1e-4
POE_COMMISSION_PCT = 0.0


@dataclass(frozen=True)
class PPOStepChange:
    """One row of the original-PPO vs STE vs POE behaviour map."""

    step: int
    name: str
    original_ppo: str
    ppo_ste: str
    ppo_poe: str
    poe_changes_original_ppo: bool
    what_poe_changes: str


@dataclass
class TransitionResult:
    """Numeric outcome of one PPO step on the toy portfolio."""

    name: str
    action_used: np.ndarray
    cash: float
    holdings_value: np.ndarray
    portfolio_value: float
    reward: float
    log_prob: Optional[float]
    value_s0: Optional[float]
    value_s1: Optional[float]
    td_error: Optional[float]
    advantage: Optional[float]
    actor_loss: Optional[float]
    critic_target: Optional[float]
    critic_loss: Optional[float]
    changed_original_ppo_steps: tuple
    notes: tuple = field(default_factory=tuple)


def gaussian_log_prob(action: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Independent Gaussian log π(a|s), matching original PPO step 1."""

    action = np.asarray(action, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    z = (action - mu) / sigma
    return float(np.sum(-0.5 * np.log(2.0 * np.pi) - np.log(sigma) - 0.5 * z**2))


def ppo_clipped_actor_loss(ratio: float, advantage: float) -> float:
    """Original PPO step 5: -min(ratio A, clip(ratio) A)."""

    clipped = float(np.clip(ratio, CLIP_LOW, CLIP_HIGH))
    return float(-min(ratio * advantage, clipped * advantage))


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = np.asarray(x, dtype=np.float64) - np.max(x)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def td_error(
    reward: float, gamma: float, value_s1: float, value_s0: float
) -> float:
    """Original PPO step 4 one-step advantage δ = r + γV(s') - V(s)."""

    return float(reward + gamma * value_s1 - value_s0)


def _initial_shares() -> np.ndarray:
    allocated = INITIAL_VALUE * WEIGHTS_S0
    return allocated / PRICES_S0


def original_ppo_transition(
    ratio: float = 1.05,
) -> TransitionResult:
    """Replay the seven-step PPO summary on the toy allocation example.

    The actor samples target weights, the book is rebalanced immediately,
    and the reward is the percent return used in the summary (0.93).
    Critic / GAE / clipping are the original PPO formulas.
    """

    log_prob = gaussian_log_prob(ACTION, MU, SIGMA)
    holdings = INITIAL_VALUE * ACTION
    price_rel = 1.0 + ASSET_RETURNS
    holdings_after = holdings * price_rel
    portfolio_value = float(np.sum(holdings_after))
    # Summary uses +0.93% as the scalar 0.93 (not 0.0093).
    reward = 0.93
    delta = td_error(reward, GAMMA, V_S1, V_S0)
    actor_loss = ppo_clipped_actor_loss(ratio, delta)
    critic_target = float(reward + GAMMA * V_S1)
    critic_loss = float((V_S0 - critic_target) ** 2)
    return TransitionResult(
        name="original_ppo",
        action_used=ACTION.copy(),
        cash=0.0,
        holdings_value=holdings_after,
        portfolio_value=portfolio_value,
        reward=reward,
        log_prob=log_prob,
        value_s0=V_S0,
        value_s1=V_S1,
        td_error=delta,
        advantage=delta,
        actor_loss=actor_loss,
        critic_target=critic_target,
        critic_loss=critic_loss,
        changed_original_ppo_steps=(),
        notes=(
            "Action is the sampled target weight vector.",
            "Reward is the summary's +0.93% written as 0.93.",
            "Advantage, clipped surrogate, and critic MSE are original PPO.",
        ),
    )


def ste_transition(
    hmax: int = STE_HMAX,
    reward_scaling: float = STE_REWARD_SCALING,
    ratio: float = 1.05,
) -> TransitionResult:
    """Original PPO actor/critic running inside StockTradingEnv mechanics.

    Step 1 still stores Gaussian log_prob. Steps 3–7 keep original PPO.
    What changes is step 2: the sampled vector is a share-count delta,
    rounded through ``hmax``, not a portfolio-weight command.
    """

    log_prob = gaussian_log_prob(ACTION, MU, SIGMA)
    cash = 0.0
    shares = _initial_shares()
    begin_value = cash + float(np.sum(shares * PRICES_S0))

    # StockTradingEnv.step: scale, then drop the fractional share.
    share_delta = (ACTION * hmax).astype(int)

    prices = PRICES_S0.copy()
    # Sell first (negative deltas), then buy. Both toy actions are buys.
    for index, delta in enumerate(share_delta):
        if delta < 0:
            sold = min(-delta, shares[index])
            cash += sold * prices[index]
            shares[index] -= sold
    for index, delta in enumerate(share_delta):
        if delta > 0:
            affordable = np.floor(cash / prices[index]) if prices[index] > 0 else 0
            bought = min(delta, affordable)
            cash -= bought * prices[index]
            shares[index] += bought

    prices_s1 = prices * (1.0 + ASSET_RETURNS)
    end_value = cash + float(np.sum(shares * prices_s1))
    dollar_pnl = end_value - begin_value
    reward = dollar_pnl * reward_scaling

    # Critic values from the summary are dollar-scale remaining return.
    # STE still *uses* those PPO formulas; only r (and therefore A) change.
    delta = td_error(reward, GAMMA, V_S1, V_S0)
    actor_loss = ppo_clipped_actor_loss(ratio, delta)
    critic_target = float(reward + GAMMA * V_S1)
    critic_loss = float((V_S0 - critic_target) ** 2)

    no_fill = cash == 0.0 and np.all(share_delta >= 0) and np.allclose(
        shares, _initial_shares()
    )
    notes = [
        f"STE interprets a={ACTION.tolist()} as share deltas {share_delta.tolist()} "
        f"(hmax={hmax}, truncated toward 0).",
        "Starting cash is 0 because the book is already 50/50 invested, so buy orders do not fill.",
        f"Reward is dollar PnL {dollar_pnl:.4f} scaled by {reward_scaling} → {reward:.6g}.",
        "Actor log_prob, critic, GAE, clipping, and on-policy discard stay original PPO.",
    ]
    if not np.allclose(share_delta, ACTION * hmax):
        notes.append(
            f"astype(int) truncated {ACTION * hmax} to {share_delta.tolist()}."
        )
    if np.all(share_delta == 0):
        notes.append("Integer rounding of |a| < 1/hmax produces a zero trade.")
    if no_fill:
        notes.append("Market is taken on the original 50/50 holdings, not on [0.58, 0.42].")

    return TransitionResult(
        name="ppo_ste",
        action_used=share_delta.astype(np.float64),
        cash=float(cash),
        holdings_value=shares * prices_s1,
        portfolio_value=end_value,
        reward=float(reward),
        log_prob=log_prob,
        value_s0=V_S0,
        value_s1=V_S1,
        td_error=delta,
        advantage=delta,
        actor_loss=actor_loss,
        critic_target=critic_target,
        critic_loss=critic_loss,
        changed_original_ppo_steps=(1, 2, 4),
        notes=tuple(notes),
    )


def poe_env_transition(
    commission_pct: float = POE_COMMISSION_PCT,
) -> TransitionResult:
    """POE environment step: this is where POE first changes original PPO.

    Changes versus the summary:
    * action dimension becomes cash + assets (n+1)
    * weights that do not already sit on the simplex are softmax-normalised
    * the book is rebalanced to those weights, then multiplied by price
      relatives (cash relative is 1)
    * reward is ln(V_t / V_{t-1}), not percent return or dollar PnL
    * optional transaction-remainder factor μ shrinks V before the move
    """

    raw = np.concatenate(([0.0], ACTION))  # cash + two assets
    if np.isclose(np.sum(raw), 1.0, atol=1e-6) and np.min(raw) >= 0:
        weights = raw
        normalised = False
    else:
        weights = softmax(raw)
        normalised = True

    mu = 1.0
    if commission_pct > 0:
        # Closed form of POE's TRF iteration when the previous book is
        # fully invested and the new cash weight is 0.
        mu = 1.0 - 2.0 * commission_pct + commission_pct**2

    value_after_fee = INITIAL_VALUE * mu
    price_rel = np.concatenate(([1.0], 1.0 + ASSET_RETURNS))
    holdings = value_after_fee * (weights * price_rel)
    portfolio_value = float(np.sum(holdings))
    reward = float(np.log(portfolio_value / INITIAL_VALUE))

    notes = [
        f"POE action is (cash, *assets) = {weights.tolist()}.",
        "Rebalance happens in weight space; there is no integer-share rounding.",
        f"Reward is log-return ln(V'/V) = {reward:.6f}, not +0.93%.",
    ]
    if normalised:
        notes.append("Raw weights did not sum to 1, so POE applied softmax.")
    if commission_pct > 0:
        notes.append(f"TRF commission μ={mu:.6f} reduced the book before the market move.")

    return TransitionResult(
        name="ppo_poe_env",
        action_used=weights,
        cash=float(holdings[0]),
        holdings_value=holdings[1:],
        portfolio_value=portfolio_value,
        reward=reward,
        log_prob=None,
        value_s0=None,
        value_s1=None,
        td_error=None,
        advantage=None,
        actor_loss=None,
        critic_target=None,
        critic_loss=None,
        changed_original_ppo_steps=(1, 2),
        notes=tuple(notes),
    )


def poe_policy_gradient_update(
    env_result: Optional[TransitionResult] = None,
) -> TransitionResult:
    """Jiang PG update used by FinRL's POE agent — this replaces PPO steps 3–7.

    ``PolicyGradient._gradient_ascent`` maximises ln(μ (W · P)) directly.
    There is no critic, no advantage, no probability ratio, no clipping,
    and the sequential batch is reused rather than thrown away as
    on-policy PPO data.
    """

    env_result = env_result or poe_env_transition()
    weights = env_result.action_used
    price_rel = np.concatenate(([1.0], 1.0 + ASSET_RETURNS))
    trf_mu = 1.0 if POE_COMMISSION_PCT == 0 else (
        1.0 - 2.0 * POE_COMMISSION_PCT + POE_COMMISSION_PCT**2
    )
    growth = float(np.sum(weights * price_rel * trf_mu))
    pg_loss = float(-np.log(growth))
    return TransitionResult(
        name="ppo_poe_pg",
        action_used=weights,
        cash=env_result.cash,
        holdings_value=env_result.holdings_value.copy(),
        portfolio_value=env_result.portfolio_value,
        reward=env_result.reward,
        log_prob=None,
        value_s0=None,
        value_s1=None,
        td_error=None,
        advantage=None,
        actor_loss=pg_loss,
        critic_target=None,
        critic_loss=None,
        changed_original_ppo_steps=(1, 2, 3, 4, 5, 6, 7),
        notes=(
            "EIIE policy is deterministic softmax; it does not sample or store log_prob.",
            "No critic network: the batch log-wealth Σ ln(μ (W · P)) is the value.",
            "No advantage and no PPO ratio. Loss = -mean(log(sum(W * price_rel * μ))).",
            "Sequential replay buffer is reused; test-time online learning continues.",
            f"PG loss on this step = -ln({growth:.6f}) = {pg_loss:.6f}.",
        ),
    )


def ppo_on_poe_transition(
    ratio: float = 1.05,
) -> TransitionResult:
    """What original PPO *would* do if only the POE environment changed.

    Keeps Gaussian log_prob, critic, GAE, clipping, and on-policy discard.
    Feeds them POE's simplex action and log-return reward. This isolates
    environment-driven changes (steps 1–2, and therefore 4) from the
    algorithm replacement in ``poe_policy_gradient_update``.
    """

    env_result = poe_env_transition()
    log_prob = gaussian_log_prob(ACTION, MU, SIGMA)
    delta = td_error(env_result.reward, GAMMA, V_S1, V_S0)
    actor_loss = ppo_clipped_actor_loss(ratio, delta)
    critic_target = float(env_result.reward + GAMMA * V_S1)
    critic_loss = float((V_S0 - critic_target) ** 2)
    return TransitionResult(
        name="ppo_on_poe",
        action_used=env_result.action_used,
        cash=env_result.cash,
        holdings_value=env_result.holdings_value.copy(),
        portfolio_value=env_result.portfolio_value,
        reward=env_result.reward,
        log_prob=log_prob,
        value_s0=V_S0,
        value_s1=V_S1,
        td_error=delta,
        advantage=delta,
        actor_loss=actor_loss,
        critic_target=critic_target,
        critic_loss=critic_loss,
        changed_original_ppo_steps=(1, 2, 4),
        notes=(
            "Same clipped PPO update as the summary, but r is POE log-return.",
            "Because r ≈ 0.009 vs V ∈ {105, 106}, the critic units no longer match.",
            "Advantage sign/magnitude therefore differ from the summary's A=0.87.",
        ),
    )


def ppo_step_changes() -> tuple[PPOStepChange, ...]:
    """Where PPO-STE and PPO-POE diverge from the original seven PPO steps."""

    return (
        PPOStepChange(
            step=1,
            name="Actor decides",
            original_ppo=(
                "State s=[prices, weights]; actor outputs Gaussian (μ, σ); "
                "sample target weights a=[0.58, 0.42]; store log_prob."
            ),
            ppo_ste=(
                "State is the STE vector [cash, prices, share counts, indicators]. "
                "Gaussian is over buy/sell fractions in [-1, 1], later × hmax. "
                "log_prob is still stored (original PPO)."
            ),
            ppo_poe=(
                "State is a (features, tickers, time_window) tensor plus last weights. "
                "Action dim is n_assets+1 (cash). EIIE emits a deterministic softmax "
                "portfolio vector; no sample and no log_prob."
            ),
            poe_changes_original_ppo=True,
            what_poe_changes=(
                "Replaces the stochastic Gaussian actor with a deterministic simplex "
                "policy, adds a cash weight, and folds last_action into the state."
            ),
        ),
        PPOStepChange(
            step=2,
            name="Execute and get reward",
            original_ppo=(
                "Rebalance to a=[0.58, 0.42], market moves A+4% / B-3.3%, "
                "V=$100.93, r=+0.93%."
            ),
            ppo_ste=(
                "a is scaled by hmax then truncated to an integer share delta "
                "(0.58×100 → 57). With cash=0 both buys fail, so the original "
                "50/50 book rides the market. r = (V' - V) * 1e-4 (dollar PnL), "
                "not percent return."
            ),
            ppo_poe=(
                "Softmax-normalise if needed, optionally apply TRF commission μ, "
                "then V' = sum(V * w * price_rel). r = ln(V'/V)."
            ),
            poe_changes_original_ppo=True,
            what_poe_changes=(
                "Target-weight rebalance (with cash and fees) and log-return "
                "reward replace dollar/percent PnL. This is the first numeric fork."
            ),
        ),
        PPOStepChange(
            step=3,
            name="Critic estimates values",
            original_ppo="Neural critic: V(s0)=105, V(s1)=106.",
            ppo_ste="Same critic network; V(s) predicts discounted dollar return.",
            ppo_poe=(
                "No critic network. FinRL's POE agent uses batch log-wealth "
                "Σ ln(μ (W · P)) as the value function."
            ),
            poe_changes_original_ppo=True,
            what_poe_changes="Drops the critic entirely; portfolio growth stands in for V(s).",
        ),
        PPOStepChange(
            step=4,
            name="Compute advantage",
            original_ppo=(
                "δ = r + γV(s') - V(s) = 0.93 + 0.99×106 - 105 = 0.87. "
                "A>0 reinforces the sampled weights."
            ),
            ppo_ste=(
                "Same formula, but r is scaled dollar PnL (~3.5e-5 here), so A "
                "is dominated by γV(s')-V(s) and is no longer 0.87."
            ),
            ppo_poe="Advantage is not computed. Policy is updated on log-growth directly.",
            poe_changes_original_ppo=True,
            what_poe_changes=(
                "Removes A = r + γV(s') - V(s) and GAE. There is no baseline "
                "subtraction and no 'better than expected' signal."
            ),
        ),
        PPOStepChange(
            step=5,
            name="PPO actor update",
            original_ppo=(
                "ratio=π_new/π_old; loss=-min(ratio A, clip(ratio, 0.8, 1.2) A). "
                "A>0 raises the probability of [0.58, 0.42]."
            ),
            ppo_ste="Identical clipped surrogate; only the advantage input changed.",
            ppo_poe=(
                "loss = -mean(log(sum(W * price_rel * μ))). No probability ratio "
                "and no clipping."
            ),
            poe_changes_original_ppo=True,
            what_poe_changes=(
                "Replaces the clipped likelihood-ratio surrogate with a "
                "deterministic log-wealth gradient. PPO's stability clip is gone."
            ),
        ),
        PPOStepChange(
            step=6,
            name="Train critic",
            original_ppo="target = GAE returns; loss = (V(s) - target)^2.",
            ppo_ste="Same MSE critic update (original PPO).",
            ppo_poe="Skipped: there is no critic to train.",
            poe_changes_original_ppo=True,
            what_poe_changes="Removes critic MSE. Value learning is implicit in log-wealth.",
        ),
        PPOStepChange(
            step=7,
            name="Discard and repeat",
            original_ppo="Throw away the batch (on-policy) and recast under π_new.",
            ppo_ste="Same on-policy discard as original PPO / SB3 PPO.",
            ppo_poe=(
                "Sequential buffer of length batch_size is gradient-ascented, "
                "including leftover steps at episode end. At test time the policy "
                "keeps learning online."
            ),
            poe_changes_original_ppo=True,
            what_poe_changes=(
                "Drops strict on-policy discard and adds test-time online updates."
            ),
        ),
    )


def run_asset_allocation_example(ratio: float = 1.05) -> dict:
    """Run the summary's toy trade under original PPO, PPO-STE, and PPO-POE."""

    original = original_ppo_transition(ratio=ratio)
    ste = ste_transition(ratio=ratio)
    poe_env = poe_env_transition()
    poe_pg = poe_policy_gradient_update(poe_env)
    ppo_on_poe = ppo_on_poe_transition(ratio=ratio)
    return {
        "original_ppo": original,
        "ppo_ste": ste,
        "ppo_poe_env": poe_env,
        "ppo_poe_pg": poe_pg,
        "ppo_on_poe": ppo_on_poe,
        "step_changes": ppo_step_changes(),
    }


def _fmt_array(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{x:.4f}" for x in np.asarray(values, dtype=np.float64)) + "]"


def _fmt_optional(value: Optional[float], digits: int = 6) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def format_walkthrough(results: Optional[dict] = None) -> str:
    """Pretty-print the seven-step comparison and the numeric toy example."""

    results = results or run_asset_allocation_example()
    lines = [
        "PPO STE vs PPO POE — where POE changes original PPO",
        "=" * 72,
        "",
        "Setup (from the PPO asset-allocation summary)",
        f"  Portfolio: ${INITIAL_VALUE:.0f} | prices A,B={PRICES_S0.tolist()} | "
        f"weights={WEIGHTS_S0.tolist()}",
        f"  Actor μ={MU.tolist()}, σ={SIGMA.tolist()} | sampled a={ACTION.tolist()}",
        f"  Market: A {ASSET_RETURNS[0]:+.1%}, B {ASSET_RETURNS[1]:+.1%}",
        f"  Critic V(s0)={V_S0}, V(s1)={V_S1}, γ={GAMMA}",
        "",
        "Step-by-step behaviour map",
        "-" * 72,
    ]
    for change in results["step_changes"]:
        flag = "CHANGES original PPO" if change.poe_changes_original_ppo else "keeps original PPO"
        lines.extend(
            [
                f"Step {change.step}: {change.name}  [{flag}]",
                f"  Original PPO : {change.original_ppo}",
                f"  PPO STE      : {change.ppo_ste}",
                f"  PPO POE      : {change.ppo_poe}",
            ]
        )
        if change.poe_changes_original_ppo:
            lines.append(f"  POE change   : {change.what_poe_changes}")
        lines.append("")

    lines.extend(["Numeric toy transition", "-" * 72])
    for key in ("original_ppo", "ppo_ste", "ppo_poe_env", "ppo_on_poe", "ppo_poe_pg"):
        result = results[key]
        lines.extend(
            [
                f"{result.name}",
                f"  action used     : {_fmt_array(result.action_used)}",
                f"  holdings value  : {_fmt_array(result.holdings_value)}  "
                f"cash={result.cash:.4f}",
                f"  portfolio value : {result.portfolio_value:.4f}",
                f"  reward          : {_fmt_optional(result.reward)}",
                f"  log_prob        : {_fmt_optional(result.log_prob)}",
                f"  δ / advantage   : {_fmt_optional(result.advantage)}",
                f"  actor loss      : {_fmt_optional(result.actor_loss)}",
                f"  critic loss     : {_fmt_optional(result.critic_loss)}",
                f"  PPO steps changed: {result.changed_original_ppo_steps or 'none'}",
            ]
        )
        for note in result.notes:
            lines.append(f"    - {note}")
        lines.append("")

    orig = results["original_ppo"]
    ste = results["ppo_ste"]
    poe_env = results["ppo_poe_env"]
    ppo_on_poe = results["ppo_on_poe"]
    poe_pg = results["ppo_poe_pg"]
    lines.extend(
        [
            "Takeaways",
            "-" * 72,
            f"  Original PPO reward={orig.reward:.2f}, A={orig.advantage:.2f} (summary).",
            f"  PPO STE never rebalanced to [0.58, 0.42]; V={ste.portfolio_value:.4f}, "
            f"r={ste.reward:.6f}, A={ste.advantage:.4f}.",
            f"  POE *did* rebalance; V={poe_env.portfolio_value:.4f} matches the summary "
            f"book, but r={poe_env.reward:.6f} is log-return, so PPO-on-POE A="
            f"{ppo_on_poe.advantage:.4f} ≠ 0.87.",
            f"  FinRL's POE agent then drops steps 3–7: actor loss={poe_pg.actor_loss:.6f} "
            "with no critic, ratio, clip, or on-policy discard.",
        ]
    )
    return "\n".join(lines) + "\n"
