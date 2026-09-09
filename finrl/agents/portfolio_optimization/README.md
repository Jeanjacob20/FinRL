# Portfolio Optimization Agents

This directory contains architectures and algorithms commonly used in portfolio optimization agents.

To instantiate the model, it's necessary to have an instance of [PortfolioOptimizationEnv](/finrl/meta/env_portfolio_optimization/). In the example below, we use the `DRLAgent` class to instantiate a policy gradient ("pg") model. With the dictionary `model_kwargs`, we can set the `PolicyGradient` class parameters and, whith the dictionary `policy_kwargs`, it's possible to change the parameters of the chosen architecture.

```python
from finrl.agents.portfolio_optimization.models import DRLAgent
from finrl.agents.portfolio_optimization.architectures import EIIE

# set PolicyGradient algorithm arguments
model_kwargs = {
    "lr": 0.01,
    "policy": EIIE,
}

# set EIIE architecture arguments
policy_kwargs = {
    "k_size": 4
}

model = DRLAgent(train_env).get_model("pg", model_kwargs, policy_kwargs)
```

In the example below, the model is trained in 5 episodes (we define an episode as a complete period of the used environment).

```python
DRLAgent.train_model(model, episodes=5)
```

It's important that the architecture and the environment have the same `time_window` defined. By default, both of them use 50 timesteps as `time_window`. For more details about what is a time window, check this [article](https://doi.org/10.5753/bwaif.2023.231144).

### Where POE changes original PPO

FinRL's stock-trading PPO (`PPO` + `StockTradingEnv`, "PPO STE") is the original clipped actor-critic algorithm. The POE stack changes that behaviour in two layers, mapped onto the usual seven PPO steps:

1. **Actor** — EIIE emits a deterministic softmax portfolio vector of size `n_assets + 1` (cash). There is no Gaussian sample and no stored `log_prob`.
2. **Execute / reward** — the env rebalances to those weights (softmax if needed, optional TRF fees) and returns `ln(V_t / V_{t-1})`, not dollar PnL.
3. **Critic** — dropped. Batch log-wealth `Σ ln(μ (W · P))` stands in for `V(s)`.
4. **Advantage** — dropped. No `A = r + γV(s') - V(s)` and no GAE.
5. **Actor loss** — `-mean(log(sum(W * price_rel * μ)))` instead of the clipped ratio surrogate.
6. **Critic loss** — skipped.
7. **On-policy discard** — sequential minibatches are reused, and the policy keeps learning at test time.

A numeric walkthrough of the two-asset PPO summary (the `$100`, `a=[0.58, 0.42]`, A+4% / B−3.3% example) lives in [`ppo_ste_vs_poe.py`](ppo_ste_vs_poe.py) and can be printed with:

```bash
python examples/ppo_ste_vs_poe_walkthrough.py
```

That script also shows *PPO-on-POE*: original PPO formulas fed only POE's action and log-return. The advantage already diverges from the summary's `A = 0.87` before Jiang PG removes steps 3–7.

### Keeping the PPO steps on POE

To keep the original seven PPO steps and only change what POE actually requires, use the `"ppo"` agent in this package. It still samples a Gaussian, stores `log_prob`, runs a critic, GAE, the clipped surrogate, critic MSE, and on-policy discard. The POE adapters are:

* flatten `(features, tickers, time_window)` and concatenate the last weights
* softmax the sampled logits into a cash + assets vector (action dim `n+1`)
* feed POE's `ln(V_t / V_{t-1})` into GAE so the critic lives in log-return units

```python
from finrl.agents.portfolio_optimization.models import DRLAgent

model_kwargs = {
    "lr": 3e-4,
    "n_steps": 128,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,  # ratio clip to [0.8, 1.2], same as the summary
}

policy_kwargs = {"hidden_sizes": (64, 64)}

model = DRLAgent(train_env).get_model("ppo", model_kwargs, policy_kwargs)
DRLAgent.train_model(model, episodes=5)
```

The 10-ticker Brazilian portfolio from `FinRL_PortfolioOptimizationEnv_Demo.ipynb` is wired to this PPO agent in:

```bash
python examples/ppo_poe_10ticker_sandbox.py --episodes 40
```

and in the notebook `examples/FinRL_PortfolioOptimizationEnv_PPO_Demo.ipynb`.

Do not pass `"pg"` / EIIE if you want PPO: that path replaces steps 3–7 with Jiang log-wealth gradient ascent.

### Policy Gradient Algorithm

The class `PolicyGradient` implements the Policy Gradient algorithm used in *Jiang et al* paper. This algorithm is inspired by DDPG (deep deterministic policy gradient), but there are a couple of differences:
- DDPG is an actor-critic algorithm, so it has an actor and a critic neural network. The algorithm below, however, doesn't have a critic neural network and uses the portfolio value as value function: the policy will be updated to maximize the portfolio value.
- DDPG usually makes use of a noise parameter in the action during training to create an exploratory behavior. PG algorithm, on the other hand, has a full-exploit approach.
- DDPG randomly samples experiences from its replay buffer. The implemented policy gradient, however, samples a sequential batch of experiences in time, to make it possible to calculate the variation of the portfolio value in the batch and use it as value function.

The algorithm was implemented as follows:
1. Initializes policy network and replay buffer;
2. For each episode, do the following:
    1. For each period of `batch_size` timesteps, do the following:
        1. For each timestep, define an action to be performed, simulate the timestep and save the experiences in the replay buffer.
        2. After `batch_size` timesteps are simulated, sample the replay buffer.
        4. Calculate the value function: $V = \sum\limits_{t=1}^{batch\_size} ln(\mu_{t}(W_{t} \cdot P_{t}))$, where $W_{t}$ is the action performed at timestep t, $P_{t}$ is the price variation vector at timestep t and $\mu_{t}$ is the transaction remainder factor at timestep t. Check *Jiang et al* paper for more details.
        5. Perform gradient ascent in the policy network.
    2. If, in the and of episode, there is sequence of remaining experiences in the replay buffer, perform steps 1 to 5 with the remaining experiences.

### References

If you are using one of them in your research, you can use the following references.

#### EIIE Architecture and Policy Gradient algorithm

[A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem](https://doi.org/10.48550/arXiv.1706.10059)
```
@misc{jiang2017deep,
      title={A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem},
      author={Zhengyao Jiang and Dixing Xu and Jinjun Liang},
      year={2017},
      eprint={1706.10059},
      archivePrefix={arXiv},
      primaryClass={q-fin.CP}
}
```

#### EI3 Architecture

[A Multi-Scale Temporal Feature Aggregation Convolutional Neural Network for Portfolio Management](https://doi.org/10.1145/3357384.3357961)
```
@inproceedings{shi2018multiscale,
               author = {Shi, Si and Li, Jianjun and Li, Guohui and Pan, Peng},
               title = {A Multi-Scale Temporal Feature Aggregation Convolutional Neural Network for Portfolio Management},
               year = {2019},
               isbn = {9781450369763},
               publisher = {Association for Computing Machinery},
               address = {New York, NY, USA},
               url = {https://doi.org/10.1145/3357384.3357961},
               doi = {10.1145/3357384.3357961},
               booktitle = {Proceedings of the 28th ACM International Conference on Information and Knowledge Management},
               pages = {1613–1622},
               numpages = {10},
               keywords = {portfolio management, reinforcement learning, inception network, convolution neural network},
               location = {Beijing, China},
               series = {CIKM '19} }
```
