"""Original PPO with only the adapters PortfolioOptimizationEnv needs.

The seven PPO steps from the asset-allocation summary are unchanged:

1. Actor samples ``a ~ N(μ(s), σ)`` and stores ``log_prob``
2. Environment executes and returns a reward
3. Critic estimates ``V(s)`` and ``V(s')``
4. Advantage ``A = r + γV(s') - V(s)`` (GAE)
5. Clipped surrogate ``-min(ratio A, clip(ratio) A)``
6. Critic MSE to GAE returns
7. Discard the on-policy batch and recast

What changes is the MDP interface, not those formulas:

* observation is POE's ``(features, tickers, time_window)`` tensor plus the
  last portfolio vector
* sampled logits are softmaxed into a cash + assets weight vector
* the reward that enters GAE is POE's log-return ``ln(V_t / V_{t-1})``
* the critic is trained in that same log-return units
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.optim import AdamW
from tqdm import tqdm


def unpack_reset(env):
    """POE reset is ``obs`` or ``(obs, info)`` depending on ``new_gym_api``."""

    out = env.reset()
    if isinstance(out, tuple):
        return out[0]
    return out


def unpack_step(env, action):
    """POE step is a 4-tuple or a gymnasium 5-tuple."""

    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, float(reward), bool(terminated or truncated), info
    obs, reward, done, info = out
    return obs, float(reward), bool(done), info


def split_observation(obs, last_action):
    """Return ``(state_array, last_weights)`` from a Box or Dict POE obs."""

    if isinstance(obs, dict):
        state = np.asarray(obs["state"], dtype=np.float32)
        last = np.asarray(obs.get("last_action", last_action), dtype=np.float32)
        return state, last
    return np.asarray(obs, dtype=np.float32), np.asarray(last_action, dtype=np.float32)


def flatten_poe_state(obs, last_action):
    """Concatenate the POE market tensor with the last portfolio vector."""

    state, last = split_observation(obs, last_action)
    return np.concatenate([state.ravel(), last.ravel()]).astype(np.float32)


def logits_to_weights(logits):
    """Map unconstrained actor logits onto the portfolio simplex (cash + assets)."""

    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return (exp / np.sum(exp)).astype(np.float32)


def observation_size(env):
    space = env.observation_space
    if hasattr(space, "spaces"):
        return int(np.prod(space["state"].shape))
    return int(np.prod(space.shape))


def compute_gae(rewards, values, dones, next_value, gamma, gae_lambda):
    """Step 4: generalised advantage estimation, walking the batch backwards."""

    advantages = np.zeros_like(rewards, dtype=np.float64)
    gae = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        bootstrap = next_value if t == len(rewards) - 1 else values[t + 1]
        mask = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * bootstrap * mask - values[t]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)


class ActorCritic(nn.Module):
    """Gaussian actor + scalar critic on a flattened POE observation."""

    def __init__(self, obs_dim, action_dim, hidden_sizes=(64, 64)):
        super().__init__()
        layers = []
        last = obs_dim
        for size in hidden_sizes:
            layers.extend([nn.Linear(last, size), nn.Tanh()])
            last = size
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        feature_dim = last
        self.actor_mean = nn.Linear(feature_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(feature_dim, 1)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, obs):
        features = self.backbone(obs)
        mean = self.actor_mean(features)
        std = self.log_std.exp().expand_as(mean)
        value = self.critic(features).squeeze(-1)
        return mean, std, value

    def distribution(self, obs):
        mean, std, value = self.forward(obs)
        return Normal(mean, std), value


class PPO:
    """Clipped PPO for ``PortfolioOptimizationEnv``.

    Same actor-critic loop as original PPO / PPO-STE. Only the observation
    flattening, softmax action, and log-return reward are POE-specific.
    """

    def __init__(
        self,
        env,
        policy=None,
        policy_kwargs=None,
        validation_env=None,
        lr=3e-4,
        n_steps=128,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        n_epochs=4,
        minibatch_size=64,
        max_grad_norm=0.5,
        optimizer=AdamW,
        device="cpu",
        **unused,
    ):
        del policy, unused  # EIIE/PG kwargs are ignored; PPO has its own actor-critic
        self.env = env
        self.validation_env = validation_env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.minibatch_size = minibatch_size
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)

        policy_kwargs = {} if policy_kwargs is None else dict(policy_kwargs)
        hidden_sizes = tuple(policy_kwargs.get("hidden_sizes", (64, 64)))

        self.action_dim = int(env.action_space.shape[0])
        self.obs_dim = observation_size(env) + self.action_dim
        self.actor_critic = ActorCritic(
            self.obs_dim, self.action_dim, hidden_sizes=hidden_sizes
        ).to(self.device)
        self.optimizer = optimizer(self.actor_critic.parameters(), lr=lr)
        self.last_update = {}
        self._clear_rollout()

    def _clear_rollout(self):
        """Step 7: throw away the on-policy batch."""

        self.rollout = defaultdict(list)

    def _initial_weights(self):
        weights = np.zeros(self.action_dim, dtype=np.float32)
        weights[0] = 1.0
        return weights

    def _tensor_obs(self, obs, last_action):
        flat = flatten_poe_state(obs, last_action)
        return torch.as_tensor(flat, dtype=torch.float32, device=self.device).unsqueeze(
            0
        )

    def act(self, obs, last_action, deterministic=False):
        """Step 1: sample (or take the mean) and store log_prob.

        The Gaussian lives in unconstrained logit space. Softmax turns the
        sample into the cash+asset vector POE expects.
        """

        obs_t = self._tensor_obs(obs, last_action)
        dist, value = self.actor_critic.distribution(obs_t)
        logits = dist.mean if deterministic else dist.sample()
        log_prob = dist.log_prob(logits).sum(-1)
        weights = logits_to_weights(logits.squeeze(0).detach().cpu().numpy())
        return (
            weights,
            logits.squeeze(0).detach().cpu().numpy(),
            float(log_prob.item()),
            float(value.item()),
        )

    def _bootstrap_value(self, obs, last_action, done):
        if done:
            return 0.0
        with torch.no_grad():
            _, value = self.actor_critic.distribution(self._tensor_obs(obs, last_action))
        return float(value.item())

    def _update(self, next_obs, next_last_action, done):
        """Steps 3–6, then step 7 discards the collected batch."""

        rewards = np.asarray(self.rollout["rewards"], dtype=np.float64)
        values = np.asarray(self.rollout["values"], dtype=np.float64)
        dones = np.asarray(self.rollout["dones"], dtype=np.bool_)
        next_value = self._bootstrap_value(next_obs, next_last_action, done)
        advantages, returns = compute_gae(
            rewards, values, dones, next_value, self.gamma, self.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs = torch.as_tensor(
            np.stack(self.rollout["obs"]), dtype=torch.float32, device=self.device
        )
        logits = torch.as_tensor(
            np.stack(self.rollout["logits"]), dtype=torch.float32, device=self.device
        )
        old_log_prob = torch.as_tensor(
            self.rollout["log_probs"], dtype=torch.float32, device=self.device
        )
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        batch_size = obs.shape[0]
        minibatch = min(self.minibatch_size, batch_size)
        actor_losses = []
        critic_losses = []
        for _ in range(self.n_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch):
                mb = indices[start : start + minibatch]
                dist, value = self.actor_critic.distribution(obs[mb])
                new_log_prob = dist.log_prob(logits[mb]).sum(-1)
                ratio = (new_log_prob - old_log_prob[mb]).exp()
                clipped = ratio.clamp(1.0 - self.clip_range, 1.0 + self.clip_range)
                # Step 5: original PPO clipped surrogate
                actor_loss = -torch.min(
                    ratio * advantages_t[mb], clipped * advantages_t[mb]
                ).mean()
                entropy = dist.entropy().sum(-1).mean()
                # Step 6: critic MSE to GAE returns
                critic_loss = torch.nn.functional.mse_loss(value, returns_t[mb])
                loss = (
                    actor_loss
                    + self.vf_coef * critic_loss
                    - self.ent_coef * entropy
                )
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                actor_losses.append(float(actor_loss.item()))
                critic_losses.append(float(critic_loss.item()))

        self.last_update = {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "n_steps": batch_size,
            "advantage_mean": float(advantages.mean()),
        }
        self._clear_rollout()
        return self.last_update

    def _maybe_update(self, next_obs, next_last_action, done, force=False):
        if not self.rollout["rewards"]:
            return None
        if force or len(self.rollout["rewards"]) >= self.n_steps:
            return self._update(next_obs, next_last_action, done)
        return None

    def train(self, episodes=100):
        """Collect on-policy rollouts in POE and apply the seven PPO steps."""

        for episode in tqdm(range(1, episodes + 1)):
            obs = unpack_reset(self.env)
            last_action = self._initial_weights()
            done = False
            while not done:
                # Step 1
                weights, logits, log_prob, value = self.act(obs, last_action)
                # Step 2: POE rebalances to `weights` and returns log-return
                next_obs, reward, done, _info = unpack_step(self.env, weights)
                self.rollout["obs"].append(flatten_poe_state(obs, last_action))
                self.rollout["logits"].append(np.asarray(logits, dtype=np.float32))
                self.rollout["log_probs"].append(log_prob)
                self.rollout["values"].append(value)
                self.rollout["rewards"].append(reward)
                self.rollout["dones"].append(done)
                obs = next_obs
                last_action = weights
                self._maybe_update(obs, last_action, done)
            self._maybe_update(obs, last_action, True, force=True)
            if self.last_update:
                tqdm.write(
                    "episode {0}: actor_loss={1:.4f} critic_loss={2:.4f}".format(
                        episode,
                        self.last_update["actor_loss"],
                        self.last_update["critic_loss"],
                    )
                )
            if self.validation_env is not None:
                self.test(self.validation_env)

    def test(
        self,
        env,
        policy=None,
        online_training_period=10,
        lr=None,
        optimizer=None,
    ):
        """Greedy evaluation. Online learning is disabled so PPO stays on-policy."""

        del policy, online_training_period, lr, optimizer
        obs = unpack_reset(env)
        last_action = self._initial_weights()
        done = False
        while not done:
            weights, _logits, _log_prob, _value = self.act(
                obs, last_action, deterministic=True
            )
            obs, _reward, done, _info = unpack_step(env, weights)
            last_action = weights
