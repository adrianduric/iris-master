from utils import *
from constants import *
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