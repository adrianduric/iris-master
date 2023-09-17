from utils import *
from constants import *
from agent import *
from task import *
import numpy as np

def experiment_1():
    # Initializing task T at random position
    task = Task(task_capacity=1, task_radius=50)
    
    # Initializing agent R1 at random position
    r1 = Agent()

    # Starting simulation
    completed_tasks = 0
    for i in range(NUM_EPOCHS):
        # Updating movement (velocity and position) of agent
        r1.update_velocity_random(AGENT_ABSOLUTE_VELOCITY)
        r1.update_pos()
        
        # Checking if the agent is within the task radius, adding to
        # completed tasks and creating a new one if so
        if distance_euclid(task.pos, r1.pos) < task.task_radius:
            completed_tasks += 1
            task = Task(task_capacity=1, task_radius=50)
            print("New task created at: ", task.pos)
    print(completed_tasks)


if __name__ == "__main__":
    experiment_1()