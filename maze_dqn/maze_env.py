import numpy as np


class MazeEnv:
    def __init__(self, size=6):
        self.size = size
        self.maze = np.zeros((size, size), dtype=int)
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.walls = self._generate_walls()
        self.agent_pos = self.start
        self.action_space = 4
        self.state_space = size * size

    def _generate_walls(self):
        walls = []
        if self.size >= 5:
            walls = [(1, 1), (1, 4), (2, 3), (3, 1), (3, 2), (4, 2)]
        return [(r, c) for r, c in walls if r < self.size and c < self.size]

    def reset(self):
        self.agent_pos = self.start
        return self._get_state()

    def _get_state(self):
        r, c = self.agent_pos
        return r * self.size + c

    def step(self, action):
        r, c = self.agent_pos
        if action == 0:
            r = max(0, r - 1)
        elif action == 1:
            r = min(self.size - 1, r + 1)
        elif action == 2:
            c = max(0, c - 1)
        elif action == 3:
            c = min(self.size - 1, c + 1)

        new_pos = (r, c)
        if new_pos in self.walls:
            new_pos = self.agent_pos

        self.agent_pos = new_pos
        done = self.agent_pos == self.goal

        if done:
            reward = 10.0
        elif new_pos == self.agent_pos and (r, c) in self.walls:
            reward = -1.0
        else:
            reward = -0.1

        return self._get_state(), reward, done, {}

    def render(self):
        for r in range(self.size):
            row_str = ""
            for c in range(self.size):
                if (r, c) == self.agent_pos:
                    row_str += "A "
                elif (r, c) == self.goal:
                    row_str += "G "
                elif (r, c) in self.walls:
                    row_str += "# "
                else:
                    row_str += ". "
            print(row_str)
        print()
