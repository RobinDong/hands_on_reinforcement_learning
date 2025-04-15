import torch

from abc import ABC, abstractmethod
from bandit import BernouliBandit


class Solver(ABC):
    def __init__(self, bandit: BernouliBandit):
        self.bandit = bandit
        self.counts = torch.zeros(bandit.K)
        self.actions = []
        self.regrets = []

    @abstractmethod
    def run_one_step(self) -> int:
        raise NotImplementedError

    def run(self, num_steps: int):
        regret = 0
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)

            regret += self.bandit.best_prob.item() - self.bandit.probs[k].item()
            self.regrets.append((k, regret))


class EpsilonGreedy(Solver):
    def __init__(self, bandit: BernouliBandit, epsilon=0.01):
        super().__init__(bandit)
        self.epsilon = epsilon
        self.estimates = torch.zeros(self.bandit.K)

    def run_one_step(self) -> int:
        if torch.rand(1)[0] < self.epsilon:
            k = torch.randint(0, self.bandit.K, (1,))[0]
        else:
            k = torch.argmax(self.estimates)

        reward = self.bandit.step(k)
        self.estimates[k] += 1 / (self.counts[k] + 1) * (reward - self.estimates[k])
        return k


class DecayingEpsilonGreedy(Solver):
    def __init__(self, bandit: BernouliBandit):
        super().__init__(bandit)
        self.estimates = torch.zeros(self.bandit.K)
        self.total_count = 0

    def run_one_step(self) -> int:
        self.total_count += 1
        if torch.rand(1)[0] < (1 / self.total_count):
            k = torch.randint(0, self.bandit.K, (1,))[0]
        else:
            k = torch.argmax(self.estimates)

        reward = self.bandit.step(k)
        self.estimates[k] += 1 / (self.counts[k] + 1) * (reward - self.estimates[k])
        return k


class UCB(Solver):
    def __init__(self, bandit: BernouliBandit):
        super().__init__(bandit)
        self.estimates = torch.zeros(self.bandit.K)
        self.total_count = 0

    def run_one_step(self) -> int:
        self.total_count += 1
        ucb = self.estimates + torch.sqrt(
            torch.log(torch.tensor(self.total_count)) / (2 * self.counts + 1)
        )
        k = torch.argmax(ucb)
        reward = self.bandit.step(k)
        self.estimates[k] += 1 / (self.counts[k] + 1) * (reward - self.estimates[k])
        return k


class ThompsonSampling(Solver):
    def __init__(self, bandit: BernouliBandit):
        super().__init__(bandit)
        # Can't use torch.zeros() because all-zeros doesn't satisfy beta-distribution
        self.reward_one = torch.ones(self.bandit.K)
        self.reward_zero = torch.ones(self.bandit.K)

    def run_one_step(self) -> int:
        sampling = torch.distributions.beta.Beta(
            self.reward_one, self.reward_zero
        ).sample()
        k = torch.argmax(sampling)
        reward = self.bandit.step(k)
        self.reward_one[k] += reward
        self.reward_zero[k] += 1 - reward
        return k
