"""
Bayesian A/B Testing — Multi-Armed Bandit Experiment
=====================================================
Implements two bandit strategies:
  - Epsilon-Greedy (with decaying epsilon = 1/t)
  - Thompson Sampling (with known precision / Gaussian prior)

Usage:
    python Bandit.py

:author: DS-223 Student
:date: 2026
"""

############################### LOGGER
from abc import ABC, abstractmethod
from loguru import logger

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# ─── Constants ────────────────────────────────────────────────────────────────
BANDIT_REWARDS   = [1, 2, 3, 4]   # true mean reward for each bandit
NUM_TRIALS       = 20_000
EPSILON_INITIAL  = 1.0            # starting epsilon for ε-greedy
PRECISION        = 1.0            # known precision (1/variance) for Thompson Sampling
OUTPUT_CSV       = "results.csv"
OUTPUT_PLOT1     = "plot1_learning_curves.png"
OUTPUT_PLOT2     = "plot2_comparison.png"


# ──────────────────────────────────────────────────────────────────────────────
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
        """Simulate pulling this bandit's arm and return a reward sample."""
        pass

    @abstractmethod
    def update(self):
        """Update internal statistics after observing a reward."""
        pass

    @abstractmethod
    def experiment(self):
        """Run the full bandit experiment over NUM_TRIALS trials."""
        pass

    @abstractmethod
    def report(self):
        """
        Summarise experiment results:
          - store rewards in CSV
          - log average reward  (f-string, logger.info)
          - log average regret  (f-string, logger.info)
        """
        pass


# ──────────────────────────────────────────────────────────────────────────────
class Visualization:
    """
    Plotting helpers shared by both algorithms.

    :param eg_rewards: Per-trial rewards from Epsilon-Greedy.
    :type eg_rewards: list[float]
    :param ts_rewards: Per-trial rewards from Thompson Sampling.
    :type ts_rewards: list[float]
    :param eg_selected: Bandit index chosen each trial (E-Greedy).
    :type eg_selected: list[int]
    :param ts_selected: Bandit index chosen each trial (Thompson Sampling).
    :type ts_selected: list[int]
    """

    def __init__(self, eg_rewards, ts_rewards, eg_selected, ts_selected):
        self.eg_rewards   = eg_rewards
        self.ts_rewards   = ts_rewards
        self.eg_selected  = eg_selected
        self.ts_selected  = ts_selected

    # ------------------------------------------------------------------
    def plot1(self):
        """
        Visualise the per-algorithm learning process on linear **and** log
        scale — i.e., the estimated reward of the chosen bandit over time.

        Saves to ``plot1_learning_curves.png``.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Learning Process: Estimated Reward of Chosen Bandit",
                     fontsize=14, fontweight="bold")

        trials = np.arange(1, NUM_TRIALS + 1)

        eg_cum_avg = np.cumsum(self.eg_rewards) / trials
        ts_cum_avg = np.cumsum(self.ts_rewards) / trials

        # ── Linear scale ──────────────────────────────────────────────
        ax = axes[0, 0]
        ax.plot(trials, eg_cum_avg, label="E-Greedy", color="steelblue", lw=1)
        ax.axhline(max(BANDIT_REWARDS), color="red", ls="--", lw=0.8,
                   label=f"Optimal = {max(BANDIT_REWARDS)}")
        ax.set_title("E-Greedy — Linear Scale")
        ax.set_xlabel("Trial"); ax.set_ylabel("Cumulative Avg Reward")
        ax.legend(); ax.grid(alpha=0.3)

        ax = axes[0, 1]
        ax.plot(trials, ts_cum_avg, label="Thompson Sampling", color="darkorange", lw=1)
        ax.axhline(max(BANDIT_REWARDS), color="red", ls="--", lw=0.8,
                   label=f"Optimal = {max(BANDIT_REWARDS)}")
        ax.set_title("Thompson Sampling — Linear Scale")
        ax.set_xlabel("Trial"); ax.set_ylabel("Cumulative Avg Reward")
        ax.legend(); ax.grid(alpha=0.3)

        # ── Log scale ─────────────────────────────────────────────────
        ax = axes[1, 0]
        ax.plot(trials, eg_cum_avg, label="E-Greedy", color="steelblue", lw=1)
        ax.axhline(max(BANDIT_REWARDS), color="red", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_title("E-Greedy — Log Scale")
        ax.set_xlabel("Trial (log)"); ax.set_ylabel("Cumulative Avg Reward")
        ax.legend(); ax.grid(alpha=0.3, which="both")

        ax = axes[1, 1]
        ax.plot(trials, ts_cum_avg, label="Thompson Sampling", color="darkorange", lw=1)
        ax.axhline(max(BANDIT_REWARDS), color="red", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_title("Thompson Sampling — Log Scale")
        ax.set_xlabel("Trial (log)"); ax.set_ylabel("Cumulative Avg Reward")
        ax.legend(); ax.grid(alpha=0.3, which="both")

        plt.tight_layout()
        plt.savefig(OUTPUT_PLOT1, dpi=150)
        plt.show()
        plt.close()
        logger.info(f"plot1 saved → {OUTPUT_PLOT1}")

    # ------------------------------------------------------------------
    def plot2(self):
        """
        Compare E-Greedy vs Thompson Sampling:
          - cumulative rewards (top panel)
          - cumulative regrets (bottom panel)

        Saves to ``plot2_comparison.png``.
        """
        trials     = np.arange(1, NUM_TRIALS + 1)
        optimal    = max(BANDIT_REWARDS)

        eg_cum_rew = np.cumsum(self.eg_rewards)
        ts_cum_rew = np.cumsum(self.ts_rewards)

        eg_regret  = optimal * trials - eg_cum_rew
        ts_regret  = optimal * trials - ts_cum_rew

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("E-Greedy vs Thompson Sampling", fontsize=14,
                     fontweight="bold")

        # Cumulative rewards
        ax = axes[0]
        ax.plot(trials, eg_cum_rew, label="E-Greedy",          color="steelblue",  lw=1)
        ax.plot(trials, ts_cum_rew, label="Thompson Sampling",  color="darkorange", lw=1)
        ax.set_title("Cumulative Reward")
        ax.set_xlabel("Trial"); ax.set_ylabel("Total Reward")
        ax.legend(); ax.grid(alpha=0.3)

        # Cumulative regrets
        ax = axes[1]
        ax.plot(trials, eg_regret, label="E-Greedy",           color="steelblue",  lw=1)
        ax.plot(trials, ts_regret, label="Thompson Sampling",   color="darkorange", lw=1)
        ax.set_title("Cumulative Regret")
        ax.set_xlabel("Trial"); ax.set_ylabel("Total Regret")
        ax.legend(); ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(OUTPUT_PLOT2, dpi=150)
        plt.show()
        plt.close()
        logger.info(f"plot2 saved → {OUTPUT_PLOT2}")


# ──────────────────────────────────────────────────────────────────────────────
class EpsilonGreedy(Bandit):
    """
    Epsilon-Greedy bandit that decays exploration: epsilon_t = 1 / t.

    Internal state tracks sample mean and number of pulls for each arm.

    :param p: True mean reward of this bandit arm.
    :type p: float
    """

    def __init__(self, p):
        """
        Initialise arm with true reward ``p`` and zero statistics.

        :param p: True mean reward.
        :type p: float
        """
        self.p          = p          # true (hidden) mean reward
        self.p_estimate = 0.0        # running sample mean
        self.n          = 0          # number of times this arm was pulled

    def __repr__(self):
        return f"EpsilonGreedy(true_mean={self.p}, estimate={self.p_estimate:.4f}, pulls={self.n})"

    def pull(self):
        """
        Sample reward from N(p, 1) — Gaussian with unit variance.

        :return: Observed reward.
        :rtype: float
        """
        return np.random.randn() + self.p

    def update(self, reward):
        """
        Incremental update of the sample mean.

        :param reward: Observed reward from the last pull.
        :type reward: float
        """
        self.n          += 1
        self.p_estimate += (reward - self.p_estimate) / self.n

    def experiment(self):
        """
        Run the epsilon-greedy experiment over ``NUM_TRIALS`` trials.

        Uses decaying epsilon = 1/t so exploration tapers off over time.

        :return: Tuple (rewards_per_trial, selected_bandit_indices).
        :rtype: tuple[list[float], list[int]]
        """
        bandits  = [EpsilonGreedy(p) for p in BANDIT_REWARDS]
        rewards  = []
        selected = []

        for t in range(1, NUM_TRIALS + 1):
            eps = EPSILON_INITIAL / t      # decaying epsilon

            if np.random.random() < eps:
                # Explore: choose a random arm
                chosen = np.random.randint(len(bandits))
            else:
                # Exploit: choose arm with highest estimated mean
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

        CSV columns: Bandit, Reward, Algorithm
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


# ──────────────────────────────────────────────────────────────────────────────
class ThompsonSampling(Bandit):
    """
    Thompson Sampling bandit with **known precision** (Gaussian likelihood,
    Gaussian conjugate prior).

    Prior: mean ~ N(0, 1/lambda_0)
    After n observations with sample mean x̄:
        posterior mean  = (lambda_0 * mu_0 + n * lambda * x̄) / (lambda_0 + n * lambda)
        posterior prec  = lambda_0 + n * lambda

    :param p: True mean reward of this bandit arm.
    :type p: float
    """

    def __init__(self, p):
        """
        Initialise arm.

        :param p: True mean reward.
        :type p: float
        """
        self.p            = p          # true mean
        self.lambda_0     = PRECISION  # prior precision
        self.mu_0         = 0.0        # prior mean
        self.lambda_n     = PRECISION  # posterior precision (updated)
        self.mu_n         = 0.0        # posterior mean (updated)
        self.n            = 0          # pull count
        self._sum_rewards = 0.0        # sum of observed rewards

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
        Bayesian update of the Gaussian posterior (known precision).

        :param reward: Observed reward.
        :type reward: float
        """
        self.n             += 1
        self._sum_rewards  += reward
        x_bar               = self._sum_rewards / self.n

        # Posterior precision and mean
        self.lambda_n = self.lambda_0 + self.n * PRECISION
        self.mu_n     = (self.lambda_0 * self.mu_0 + PRECISION * self._sum_rewards) / self.lambda_n

    def sample(self):
        """
        Draw a sample from the current posterior N(mu_n, 1/lambda_n).

        :return: Sampled estimated mean.
        :rtype: float
        """
        return np.random.randn() / np.sqrt(self.lambda_n) + self.mu_n

    def experiment(self):
        """
        Run the Thompson Sampling experiment over ``NUM_TRIALS`` trials.

        Each trial: sample from every arm's posterior, pull the arm with
        the highest sample, observe reward, update that arm's posterior.

        :return: Tuple (rewards_per_trial, selected_bandit_indices).
        :rtype: tuple[list[float], list[int]]
        """
        bandits  = [ThompsonSampling(p) for p in BANDIT_REWARDS]
        rewards  = []
        selected = []

        for _ in range(NUM_TRIALS):
            # Thompson: each arm proposes a sample from its posterior
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

        CSV columns: Bandit, Reward, Algorithm
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


# ──────────────────────────────────────────────────────────────────────────────
def comparison():
    """
    Run both algorithms, generate all plots, save CSV, and log summaries.

    Workflow:
      1. Instantiate and run EpsilonGreedy experiment.
      2. Instantiate and run ThompsonSampling experiment.
      3. Produce reports (CSV + logged metrics).
      4. Generate plot1 (learning curves) and plot2 (comparison).
    """
    logger.info("=" * 60)
    logger.info("Starting Multi-Armed Bandit Experiment")
    logger.info(f"Bandits (true means): {BANDIT_REWARDS}")
    logger.info(f"Number of trials    : {NUM_TRIALS}")
    logger.info("=" * 60)

    # ── Epsilon-Greedy ────────────────────────────────────────────────
    logger.info("Running Epsilon-Greedy …")
    eg = EpsilonGreedy(BANDIT_REWARDS[0])          # p value doesn't matter here;
    eg_rewards, eg_selected = eg.experiment()      # experiment() creates all arms
    eg.report()

    # ── Thompson Sampling ─────────────────────────────────────────────
    logger.info("Running Thompson Sampling …")
    ts = ThompsonSampling(BANDIT_REWARDS[0])
    ts_rewards, ts_selected = ts.experiment()
    ts.report()

    # ── Visualisation ─────────────────────────────────────────────────
    viz = Visualization(eg_rewards, ts_rewards, eg_selected, ts_selected)
    viz.plot1()
    viz.plot2()

    logger.info("Experiment complete. Check results.csv, plot1 and plot2 images.")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    comparison()