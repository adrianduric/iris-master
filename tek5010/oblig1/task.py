import numpy as np

class Task:
    def __init__(self, xpos: float, ypos: float):
        self.pos = np.array([xpos, ypos])