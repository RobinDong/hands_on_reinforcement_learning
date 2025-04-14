import torch
import matplotlib.pyplot as plt

from bandit import BernouliBandit
from solver import EpsilonGreedy


def plot_results(solvers):
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(
            time_list,
            [pair[1] for pair in solver.regrets],
            label=solver.__class__.__name__,
        )
    plt.xlabel("Time steps")
    plt.ylabel("Cumulative regrets")
    plt.title(f"{solvers[0].bandit.K}-armed bandit")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    torch.manual_seed(1023)
    bandit = BernouliBandit(20)
    solver = EpsilonGreedy(bandit, epsilon=0.1)
    solver.run(1000)
    print("Cumulative regret:", solver.regrets[-1][1])
    plot_results([solver])
