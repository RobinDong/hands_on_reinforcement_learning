import copy

from cliff_walking import CliffWalkingEnv


class PolicyIteration:
    def __init__(self, env, theta, gamma):
        self.env = env
        self.v = [[0] * self.env.ncol for _ in range(self.env.nrow)]
        self.pi = [
            [[0.25, 0.25, 0.25, 0.25] for _ in range(self.env.ncol)]
            for _ in range(self.env.nrow)
        ]
        self.theta = theta
        self.gamma = gamma

    def policy_evaluation(self):
        while True:
            max_diff = 0
            new_v = [[0] * self.env.ncol for _ in range(self.env.nrow)]
            for row in range(self.env.nrow):
                for col in range(self.env.ncol):
                    qsa_list = []  # for all actions
                    for action in range(4):
                        prob, (next_row, next_col), reward, done = self.env.P[row][col][
                            action
                        ]
                        qsa = prob * (reward + self.gamma * self.v[next_row][next_col])
                        qsa_list.append(self.pi[row][col][action] * qsa)
                    new_v[row][col] = sum(qsa_list)  # V = Sigma(Q(s,a))
                    max_diff = max(max_diff, abs(new_v[row][col] - self.v[row][col]))
            self.v = new_v
            if max_diff < self.theta:
                break

    def policy_improvement(self):
        for row in range(self.env.nrow):
            for col in range(self.env.ncol):
                qsa_list = []
                for action in range(4):
                    prob, (next_row, next_col), reward, done = self.env.P[row][col][
                        action
                    ]
                    qsa = prob * (reward + self.gamma * self.v[next_row][next_col])
                    qsa_list.append(qsa)
                maxq = max(qsa_list)
                cntq = qsa_list.count(maxq)
                self.pi[row][col] = [1 / cntq if q == maxq else 0 for q in qsa_list]
        return self.pi

    def policy_iteration(self):
        while True:
            self.policy_evaluation()
            old_pi = copy.deepcopy(self.pi)
            new_pi = self.policy_improvement()
            if old_pi == new_pi:
                break


if __name__ == "__main__":
    env = CliffWalkingEnv()
    poi = PolicyIteration(env, 0.001, 0.9)
    poi.policy_iteration()
    [print(row) for row in poi.pi]
