from utils import *
from constants import *
from agent import *
from task import *
import numpy as np
import matplotlib.pyplot as plt

def experiment_1():
    # Initializing task T at random position
    task = Task(task_capacity=1, task_radius=50)
    
    # Initializing agent R1 at random position
    r1 = Agent()

    result_per_epoch = []

    # Starting simulation
    completed_tasks = 0
    for i in range(NUM_RUNS):
        for j in range(NUM_EPOCHS):
            # Updating movement (velocity and position) of agent
            r1.update_velocity_random(AGENT_ABSOLUTE_VELOCITY)
            r1.update_pos()
            
            # Checking if the agent is within the task radius, adding to
            # completed tasks and creating a new one if so
            if distance_euclid(task.pos, r1.pos) < task.task_radius:
                completed_tasks += 1
                task = Task(task_capacity=1, task_radius=50)
                result_per_epoch.append(1)
            else:
                result_per_epoch.append(0)
    print(f"Tasks solved per epoch: {completed_tasks / NUM_EPOCHS}")
    
    x = np.linspace(1, NUM_RUNS, NUM_RUNS)


def experiment_2():
    # Initializing task T at random position
    task = Task(task_capacity=1, task_radius=50)
    
    # Performing experiment with different numbers of agents
    for num_agents in [3, 5, 10, 20, 30]:
        # Initializing agents at random positions
        agents = []
        for agent_i in range(num_agents):
            agent = Agent()
            agents.append(agent)

        # Starting simulation
        completed_tasks = 0
        for i in range(NUM_EPOCHS):
            # Updating movement (velocity and position) of agent
            for agent in agents:
                agent.update_velocity_random(AGENT_ABSOLUTE_VELOCITY)
                agent.update_pos()
            
                # Checking if the agent is within the task radius, adding to
                # completed tasks and creating a new one if so
                if distance_euclid(task.pos, agent.pos) < task.task_radius:
                    completed_tasks += 1
                    task = Task(task_capacity=1, task_radius=50)
        print(f"Tasks solved per epoch: {completed_tasks / NUM_EPOCHS}")

def experiment_3():
    # Initializing task T at random position
    task = Task(task_capacity=3, task_radius=50)
    
    # Performing experiment with different numbers of agents
    for num_agents in [3, 5, 10, 20, 30]:
        # Initializing agents at random positions
        agents = []
        for agent_i in range(num_agents):
            agent = Agent()
            agents.append(agent)

        # Starting simulation
        completed_tasks = 0
        for i in range(NUM_EPOCHS):
            # Updating movement (velocity and position) of agent
            num_agents_in_radius = 0
            for agent in agents:
                agent.update_velocity_random(AGENT_ABSOLUTE_VELOCITY)
                agent.update_pos()
            
                # Checking if the agent is within the task radius, adding to
                # num_agents_in_radius
                if distance_euclid(task.pos, agent.pos) < task.task_radius:
                    num_agents_in_radius += 1

                # Checking if enough rgents are close enough to task to complete it
                if num_agents_in_radius >= task.task_capacity:
                    completed_tasks += 1
                    task = Task(task_capacity=3, task_radius=50)
        print(f"Tasks solved per epoch: {completed_tasks / NUM_EPOCHS}")

if __name__ == "__main__":
    experiment_1()
    experiment_2()
    experiment_3()