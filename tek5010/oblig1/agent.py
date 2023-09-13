import numpy as np

class Agent:
    def __init__(self, xpos: float, ypos: float, vx: float, vy: float):
        self.pos = np.array([xpos, ypos])
        self.velocity = np.array([vx, vy])

        
