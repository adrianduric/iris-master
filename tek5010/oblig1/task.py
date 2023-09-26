from utils import *
from constants import *
from agent import *
import numpy as np

class Task:
    def __init__(
            self,
            x: float = None,
            y: float = None,
            task_capacity: int = 1,
            task_radius: float = 100
            ):
        x = np.random.random()*1000 if x is None else x
        y = np.random.random()*1000 if y is None else y
        self.pos = np.array([x, y])
        self.task_capacity = task_capacity
        self.task_radius = task_radius

    def sufficient_agents_in_radius(self, agents: list):
        num_agents_in_radius = 0
        for agent in agents:        
            # Checking if the agent is within the task radius, adding to
            # num_agents_in_radius
            if distance_euclid(self.pos, agent.pos) < self.task_radius:
                num_agents_in_radius += 1

            # Checking if enough rgents are close enough to task to complete it
            if num_agents_in_radius >= self.task_capacity:
                return True          
        return False