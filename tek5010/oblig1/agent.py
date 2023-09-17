from utils import *
from constants import *
import numpy as np

class Agent:
    def __init__(
            self,
            x: float = None,
            y: float = None,
            vx: float = 0.0,
            vy: float = 0.0,
            comm_dist: float = 0.0
            ):
        x = np.random.random()*1000 if x is None else x
        y = np.random.random()*1000 if y is None else y
        self.pos = np.array([x, y])
        self.velocity = np.array([vx, vy])
        self.best_pos_self = self.pos
        self.comm_dist = comm_dist

    def update_pos(self):
        """
        Updates the current position of the agent by adding the self.velocity vector to the
        self.pos vector. When updating positions, the function disallows the agent to go out
        of bounds of the square grid spanning from (0, 0) to (1000, 1000). This means if the
        agent would go out of bounds by following its trajectory at its current absolute
        velocity, it instead moves in the same direction but at a lower absolute velocity, so
        that it stops at the border of the grid.
        """

        self.pos = self.pos + self.velocity
        self.pos = np.minimum(self.pos, 1000)
        self.pos = np.maximum(self.pos, 0)

    def update_velocity_random(self, abs_velocity: float = None):
        """
        Makes the agent's movement random by changing the velocity in each direction to some
        random number (absolute velocity is set to never be higher than 25). If abs_velocity
        is specified (float between 0.0 and 25.0), vx and vy are initialized so that the
        absolute velocity sums up to abs_velocity.
        """

        # Sets absolute velocity between 0 and 25 if not previously (correctly) specified
        if abs_velocity is None or abs_velocity < 0 or abs_velocity > 25:
            abs_velocity = np.random.random()*25 # abs_velocity is now between 0 and 25.
        
        # Sets velocity in each direction to random number in interval [-1, 1]
        self.velocity = (1 - (-1))*np.random.random(self.velocity.shape) - 1

        # Normalizing vector, making the norm of self.velocity equal to 1
        norm = np.linalg.norm(self.velocity)
        # To solve the unlikely case that both directional velicities are sampled as 0, we
        # randomly set vx = 1 or vy = 1 if that happens
        if norm == 0:
            self.velocity = np.array([1, 0]) if np.random.random() > 0.5 else np.array([0, 1])
            norm = np.linalg.norm(self.velocity)
        self.velocity = self.velocity / norm

        # Multiplying self.velocity (now a unit vector) with abs_velocity to achieve desired
        # (or random) velocity
        self.velocity = self.velocity * abs_velocity

        
