# symbol:
#  R: road
#  S: start
#  G: goal
#  C: cliff
#
# cliff map:
#
#  C R R R C
#  R R R R R
#  R R R G R
#  S C C R R
#


class CliffWalkingEnv:
    def __init__(self, ncol=5, nrow=4):
        self.map = [
            "CRRGC",
            "RRRRR",
            "RRRRR",
            "SCCRR",
        ]
        self.ncol = ncol
        self.nrow = nrow
        self.P = self.createP()  # nrow * ncol * (prob, next_state, reward, done)

    def process_state(self, row: int, col: int):
        changes = [[0, -1], [0, 1], [-1, 0], [1, 0]]  # left, right, up, down
        for index, action in enumerate(changes):
            space = self.map[row][col]
            if space == "C" or space == "G":
                self.P[row][col][index] = (1, (row, col), 0, True)
                continue
            next_row = min(max(0, row + action[0]), self.nrow - 1)
            next_col = min(max(0, col + action[1]), self.ncol - 1)
            next_space = self.map[next_row][next_col]
            done = True
            reward = -1
            if next_space == "C":  # cliff
                reward = -100
            else:
                done = False
            self.P[row][col][index] = (1, (next_row, next_col), reward, done)

    def createP(self):
        self.P = [[[None] * 4 for _ in range(self.ncol)] for _ in range(self.nrow)]

        for row in range(self.nrow):
            for col in range(self.ncol):
                self.process_state(row, col)

        return self.P


if __name__ == "__main__":
    env = CliffWalkingEnv()
    [print(row) for row in env.P]
