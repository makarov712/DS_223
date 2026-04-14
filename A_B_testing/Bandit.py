"""
Bayesian A/B Testing - Multi-Armed Bandit Experiment
Implements two bandit strategies:
  - Epsilon-Greedy (with decaying epsilon = 1/t)
  - Thompson Sampling (with known precision / Gaussian prior)

Usage:
    python Bandit.py

:author: Davit_Ter-Harutyunyan
:date: 2026
"""

from abc import ABC, abstractmethod
from loguru import logger

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

BANDIT_REWARDS   = [1, 2, 3, 4]
NUM_TRIALS       = 20_000
EPSILON_INITIAL  = 1.0
PRECISION        = 1.0
OUTPUT_CSV       = "results.csv"
OUTPUT_PLOT1     = "plot1_learning_curves.png"
OUTPUT_PLOT2     = "plot2_comparison.png"


class Bandit(ABC):
    """
    Abstract base class for all multi-armed bandit strategies.

    :param p: True mean reward (used to simulate pulls).
    :type p: float
    """

    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        pass


class Visualization:
    """
    Plotting helpers shared by both algorithms.

    :param eg_rewards: Per-trial rewards from Epsilon-Greedy.
    :param ts_rewards: Per-trial rewards from Thompson Sampling.
    :param eg_selected: Bandit index chosen each trial (E-Greedy).
    :param ts_selected: Bandit index chosen each trial (Thompson Sampling).
    """

    def __init__(self, eg_rewards, ts_rewards, eg_selected, ts_selected):
        self.eg_rewards  = eg_rewards
        self.ts_rewards  = ts_rewards
        self.eg_selected = eg_selected
        self.ts_selected = ts_selected

    def plot1(self):
        """
        Visualise the learning process on linear and log scale for each algorithm.
        Saves to plot1_learning_curves.png.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Learning Process: Estimated Reward of Chosen Bandit",
                     fontsize=14, fontweight="bold")

        trials = np.arange(1, NUM_TRIALS + 1)
        eg_cum_avg = np.cumsum(self.eg_rewards) / trials
        ts_cum_avg = np.cumsum(self.ts_rewards) / trials

        ax = axes[0, 0]
        ax.plot(trials, eg_cum_avg, label="E-Greedy", color="steelblue", lw=1)
        ax.axhline(max(BANDIT_REWARDS), color="red", ls="--", lw=0.8,
                   label=f"Optimal = {max(BANDIT_REWARDS)}")
        ax.set_title("E-Greedy - Linear Scale")
        ax.set_xlabel("Trial"); ax.set_ylabel("Cumulative Avg Reward")
        ax.legend(); ax.grid(alpha=0.3)

        ax = axes[0, 1]
        ax.plot(trials, ts_cum_avg, label="Thompson Sampling", color="darkorange", lw=1)
        ax.axhline(max(BANDIT_REWARDS), color="red", ls="--", lw=0.8,
                   label=f"Optimal = {max(BANDIT_REWARDS)}")
        ax.set_title("Thompson Sampling - Linear Scale")
        ax.set_xlabel("Trial"); ax.set_ylabel("Cumulative Avg Reward")
        ax.legend(); ax.grid(alpha=0.3)

        ax = axes[1, 0]
        ax.plot(trials, eg_cum_avg, label="E-Greedy", color="steelblue", lw=1)
        ax.axhline(max(BANDIT_REWARDS), color="red", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_title("E-Greedy - Log Scale")
        ax.set_xlabel("Trial (log)"); ax.set_ylabel("Cumulative Avg Reward")
        ax.legend(); ax.grid(alpha=0.3, which="both")

        ax = axes[1, 1]
        ax.plot(trials, ts_cum_avg, label="Thompson Sampling", color="darkorange", lw=1)
        ax.axhline(max(BANDIT_REWARDS), color="red", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_title("Thompson Sampling - Log Scale")
        ax.set_xlabel("Trial (log)"); ax.set_ylabel("Cumulative Avg Reward")
        ax.legend(); ax.grid(alpha=0.3, which="both")

        plt.tight_layout()
        plt.savefig(OUTPUT_PLOT1, dpi=150)
        plt.show()
        plt.close()
        logger.info(f"plot1 saved to {OUTPUT_PLOT1}")

    def plot2(self):
        """
        Compare E-Greedy vs Thompson Sampling cumulative rewards and regrets.
        Saves to plot2_comparison.png.
        """
        trials     = np.arange(1, NUM_TRIALS + 1)
        optimal    = max(BANDIT_REWARDS)
        eg_cum_rew = np.cumsum(self.eg_rewards)
        ts_cum_rew = np.cumsum(self.ts_rewards)
        eg_regret  = optimal * trials - eg_cum_rew
        ts_regret  = optimal * trials - ts_cum_rew

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("E-Greedy vs Thompson Sampling", fontsize=14, fontweight="bold")

        ax = axes[0]
        ax.plot(trials, eg_cum_rew, label="E-Greedy",         color="steelblue",  lw=1)
        ax.plot(trials, ts_cum_rew, label="Thompson Sampling", color="darkorange", lw=1)
        ax.set_title("Cumulative Reward")
        ax.set_xlabel("Trial"); ax.set_ylabel("Total Reward")
        ax.legend(); ax.grid(alpha=0.3)

        ax = axes[1]
        ax.plot(trials, eg_regret, label="E-Greedy",          color="steelblue",  lw=1)
        ax.plot(trials, ts_regret, label="Thompson Sampling",  color="darkorange", lw=1)
        ax.set_title("Cumulative Regret")
        ax.set_xlabel("Trial"); ax.set_ylabel("Total Regret")
        ax.legend(); ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(OUTPUT_PLOT2, dpi=150)
        plt.show()
        plt.close()
        logger.info(f"plot2 saved to {OUTPUT_PLOT2}")


class EpsilonGreedy(Bandit):
    """
    Epsilon-Greedy bandit with decaying epsilon (epsilon_t = 1/t).

    :param p: True mean reward of this bandit arm.
    :type p: float
    """

    def __init__(self, p):
        """
        :param p: True mean reward.
        :type p: float
        """
        self.p          = p
        self.p_estimate = 0.0
        self.n          = 0

    def __repr__(self):
        return f"EpsilonGreedy(true_mean={self.p}, estimate={self.p_estimate:.4f}, pulls={self.n})"

    def pull(self):
        """
        Sample reward from N(p, 1).

        :return: Observed reward.
        :rtype: float
        """
        return np.random.randn() + self.p

    def update(self, reward):
        """
        Incremental update of the sample mean.

        :param reward: Observed reward.
        :type reward: float
        """
        self.n          += 1
        self.p_estimate += (reward - self.p_estimate) / self.n

    def experiment(self):
        """
        Run the epsilon-greedy experiment over NUM_TRIALS trials.

        :return: Tuple (rewards_per_trial, selected_bandit_indices).
        :rtype: tuple[list[float], list[int]]
        """
        bandits  = [EpsilonGreedy(p) for p in BANDIT_REWARDS]
        rewards  = []
        selected = []

        for t in range(1, NUM_TRIALS + 1):
            eps = EPSILON_INITIAL / t

            if np.random.random() < eps:
                chosen = np.random.randint(len(bandits))
            else:
                chosen = np.argmax([b.p_estimate for b in bandits])

            reward = bandits[chosen].pull()
            bandits[chosen].update(reward)
            rewards.append(reward)
            selected.append(chosen)

        self._bandits  = bandits
        self._rewards  = rewards
        self._selected = selected
        return rewards, selected

    def report(self):
        """
        Store results to CSV, log cumulative reward and regret.
        """
        optimal    = max(BANDIT_REWARDS)
        cum_reward = sum(self._rewards)
        cum_regret = optimal * NUM_TRIALS - cum_reward

        df = pd.DataFrame({
            "Bandit":    [BANDIT_REWARDS[i] for i in self._selected],
            "Reward":    self._rewards,
            "Algorithm": "EpsilonGreedy",
        })
        df.to_csv(OUTPUT_CSV, index=False, mode="w")
        logger.info(f"EpsilonGreedy | cumulative reward : {cum_reward:.2f}")
        logger.info(f"EpsilonGreedy | cumulative regret : {cum_regret:.2f}")
        return df


class ThompsonSampling(Bandit):
    """
    Thompson Sampling bandit with known precision (Gaussian conjugate prior).

    :param p: True mean reward of this bandit arm.
    :type p: float
    """

    def __init__(self, p):
        """
        :param p: True mean reward.
        :type p: float
        """
        self.p            = p
        self.lambda_0     = PRECISION
        self.mu_0         = 0.0
        self.lambda_n     = PRECISION
        self.mu_n         = 0.0
        self.n            = 0
        self._sum_rewards = 0.0

    def __repr__(self):
        return (f"ThompsonSampling(true_mean={self.p}, "
                f"posterior_mean={self.mu_n:.4f}, pulls={self.n})")

    def pull(self):
        """
        Sample reward from N(p, 1).

        :return: Observed reward.
        :rtype: float
        """
        return np.random.randn() + self.p

    def update(self, reward):
        """
        Bayesian update of the Gaussian posterior.

        :param reward: Observed reward.
        :type reward: float
        """
        self.n            += 1
        self._sum_rewards += reward
        self.lambda_n      = self.lambda_0 + self.n * PRECISION
        self.mu_n          = (self.lambda_0 * self.mu_0 + PRECISION * self._sum_rewards) / self.lambda_n

    def sample(self):
        """
        Draw a sample from the current posterior N(mu_n, 1/lambda_n).

        :return: Sampled estimated mean.
        :rtype: float
        """
        return np.random.randn() / np.sqrt(self.lambda_n) + self.mu_n

    def experiment(self):
        """
        Run the Thompson Sampling experiment over NUM_TRIALS trials.

        :return: Tuple (rewards_per_trial, selected_bandit_indices).
        :rtype: tuple[list[float], list[int]]
        """
        bandits  = [ThompsonSampling(p) for p in BANDIT_REWARDS]
        rewards  = []
        selected = []

        for _ in range(NUM_TRIALS):
            chosen = np.argmax([b.sample() for b in bandits])
            reward = bandits[chosen].pull()
            bandits[chosen].update(reward)
            rewards.append(reward)
            selected.append(chosen)

        self._bandits  = bandits
        self._rewards  = rewards
        self._selected = selected
        return rewards, selected

    def report(self):
        """
        Append results to CSV, log cumulative reward and regret.
        """
        optimal    = max(BANDIT_REWARDS)
        cum_reward = sum(self._rewards)
        cum_regret = optimal * NUM_TRIALS - cum_reward

        df = pd.DataFrame({
            "Bandit":    [BANDIT_REWARDS[i] for i in self._selected],
            "Reward":    self._rewards,
            "Algorithm": "ThompsonSampling",
        })
        df.to_csv(OUTPUT_CSV, index=False, mode="a", header=False)
        logger.info(f"ThompsonSampling | cumulative reward : {cum_reward:.2f}")
        logger.info(f"ThompsonSampling | cumulative regret : {cum_regret:.2f}")
        return df


def comparison():
    """
    Run both algorithms, generate all plots, save CSV, and log summaries.
    """
    logger.info("=" * 60)
    logger.info("Starting Multi-Armed Bandit Experiment")
    logger.info(f"Bandits (true means): {BANDIT_REWARDS}")
    logger.info(f"Number of trials    : {NUM_TRIALS}")
    logger.info("=" * 60)

    logger.info("Running Epsilon-Greedy ...")
    eg = EpsilonGreedy(BANDIT_REWARDS[0])
    eg_rewards, eg_selected = eg.experiment()
    eg.report()

    logger.info("Running Thompson Sampling ...")
    ts = ThompsonSampling(BANDIT_REWARDS[0])
    ts_rewards, ts_selected = ts.experiment()
    ts.report()

    viz = Visualization(eg_rewards, ts_rewards, eg_selected, ts_selected)
    viz.plot1()
    viz.plot2()

    logger.info("Experiment complete. Check results.csv, plot1 and plot2 images.")


if __name__ == "__main__":
    comparison()
