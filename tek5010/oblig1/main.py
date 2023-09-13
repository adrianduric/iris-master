import numpy as np


def distance_euclid(vec_a: np.array, vec_b: np.array):
    return np.linalg.norm(vec_b - vec_a)

class Task:
    def __init__(self, x: float, y: float, task_capacity: int, task_radius: float):
        self.pos = np.array([x, y])
        self.task_capacity = task_capacity
        self.task_radius = task_radius

    def get_pos(self):
        return self.pos

class Agent:
    def __init__(self, x: float, y: float, vx: float, vy: float, comm_dist: float):
        self.pos = np.array([x, y])
        self.velocity = np.array([vx, vy])
        self.best_pos_self = self.pos
        self.comm_dist = comm_dist

agents = []

def main():
    # Initializing task T at random position
    x = np.random.random()*1000; y = np.random.random()*1000
    task = Task(x, y, task_capacity=1, task_radius=50)

    # Initializing random position for R1:
    x = np.random.random()*1000; y = np.random.random()*1000

    # Sampling vx and vy for agent R1 so that Rv = 25:
    vx = np.random.random()*625; vy = 625 - vx
    vx = np.sqrt(vx); vy = np.sqrt(vy)
    
    # Initializing agent R1
    r1 = Agent(x, y, vx, vy, comm_dist=50)
    agents.append(r1)

    # 





if __name__ == "__main__":
    main()