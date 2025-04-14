import torch
import matplotlib.pyplot as plt

from bandit import BernouliBandit
from solver import EpsilonGreedy, DecayingEpsilonGreedy


def plot_results(solvers):
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(
            time_list,
            [pair[1] for pair in solver.regrets],
            label=f"{solver.__class__.__name__}-{solver.epsilon}",
        )
    plt.xlabel("Time steps")
    plt.ylabel("Cumulative regrets")
    plt.title(f"{solvers[0].bandit.K}-armed bandit")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    torch.manual_seed(1023)
    bandit = BernouliBandit(20)
    solver = DecayingEpsilonGreedy(bandit)
    solver.run(10000)
    print("Cumulative regret:", solver.regrets[-1][1])

    bandit = BernouliBandit(20)
    epsilons = [1e-4, 1e-3, 1e-2, 1e-1, 0.5]
    solvers = [EpsilonGreedy(bandit, epsilon=ep) for ep in epsilons]
    for solver in solvers:
        solver.run(10000)

    plot_results(solvers)
