import torch


class BernouliBandit:
    def __init__(self, K: int):
        self.probs = torch.rand(K)
        self.best_idx = torch.argmax(self.probs)
        self.best_prob = self.probs[self.best_idx]
        self.K = K

    def step(self, k: int) -> int:
        return 1 if torch.rand(1) < self.probs[k] else 0


if __name__ == "__main__":
    bandit = BernouliBandit(10)
    print("bandit:", bandit.probs)
    # Monte Carlo method
    samples = 100
    positive = 0
    for index in range(samples):
        positive += bandit.step(0)
    print(f"Estimate expectation: {positive / samples:.2f}")
