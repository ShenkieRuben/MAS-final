class Cell:
    def __init__(self, is_reachable):
        self.v = 0
        self.reward = -1
        self.is_reachable = is_reachable
        self.is_visited = False


class SpecialCell(Cell):
    def __init__(self, is_reachable, reward):
        super().__init__(is_reachable)
        self.reward = reward
