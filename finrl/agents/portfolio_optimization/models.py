"""
DRL models to solve the portfolio optimization task with reinforcement learning.
This agent was developed to work with environments like PortfolioOptimizationEnv.
"""

from __future__ import annotations

from .algorithms import PolicyGradient
from .ppo import PPO

MODELS = {"pg": PolicyGradient, "ppo": PPO}


class DRLAgent:
    """Implementation for DRL algorithms for portfolio optimization.

    Note:
        Jiang policy gradient ("pg") continues to learn during testing.
        The PPO agent ("ppo") does not: evaluation is greedy and on-policy.

    Attributes:
        env: Gym environment class.
    """

    def __init__(self, env):
        """Agent initialization.

        Args:
            env: Gym environment to be used in training.
        """
        self.env = env

    def get_model(
        self, model_name, device="cpu", model_kwargs=None, policy_kwargs=None
    ):
        """Setups DRL model.

        Args:
            model_name: Name of the model according to MODELS list.
            device: Device used to instantiate neural networks.
            model_kwargs: Arguments to be passed to model class.
            policy_kwargs: Arguments to be passed to policy class.

        Note:
            model_kwargs and policy_kwargs are dictionaries. The keys must be strings
            with the same names as the class arguments. Example for model_kwargs::

            { "lr": 0.01, "policy": EIIE }

        Returns:
            An instance of the model.
        """
        if model_name not in MODELS:
            raise NotImplementedError("The model requested was not implemented.")

        model = MODELS[model_name]
        model_kwargs = {} if model_kwargs is None else model_kwargs
        policy_kwargs = {} if policy_kwargs is None else policy_kwargs

        # add device settings
        model_kwargs["device"] = device
        policy_kwargs["device"] = device

        # add policy_kwargs inside model_kwargs
        model_kwargs["policy_kwargs"] = policy_kwargs

        return model(self.env, **model_kwargs)

    @staticmethod
    def train_model(model, episodes=100):
        """Trains portfolio optimization model.

        Args:
            model: Instance of the model.
            episoded: Number of episodes.

        Returns:
            An instance of the trained model.
        """
        model.train(episodes)
        return model

    @staticmethod
    def DRL_validation(
        model,
        test_env,
        policy=None,
        online_training_period=10,
        learning_rate=None,
        optimizer=None,
    ):
        """Evaluates a model.

        Args:
            model: Instance of the model.
            test_env: Gym environment to be used in testing.
            policy: Policy architecture to be used. If None, it will use the training
            architecture.
            online_training_period: Used by Jiang PG only. PPO ignores this and
            runs a greedy rollout.
            batch_size: Batch size to train neural network. If None, it will use the
            training batch size.
            lr: Policy neural network learning rate. If None, it will use the training
            learning rate
            optimizer: Optimizer of neural network. If None, it will use the training
            optimizer
        """
        model.test(test_env, policy, online_training_period, learning_rate, optimizer)
