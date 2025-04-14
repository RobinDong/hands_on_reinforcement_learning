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
