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

    # Starting simulation
    results_per_epoch = []
    for i in range(NUM_RUNS):
        completed_tasks = 0
        for j in range(NUM_EPOCHS):
            # Updating movement (velocity and position) of agent
            r1.update_velocity_random(AGENT_ABSOLUTE_VELOCITY)
            r1.update_pos()
            
            # Checking if the agent is within the task radius, adding to
            # completed tasks and creating a new one if so
            if distance_euclid(task.pos, r1.pos) < task.task_radius:
                completed_tasks += 1
                task = Task(task_capacity=1, task_radius=50)
        results_per_epoch.append(completed_tasks)
    
    # Plotting results
    x = np.linspace(1, NUM_RUNS, NUM_RUNS)
    y = results_per_epoch
    plt.plot(x, y, 'o')
    plt.title("Number of tasks solved per run")
    plt.xlabel("Run #")
    plt.ylabel("# of tasks solved")
    plt.show()


def experiment_2():
    # Initializing task T at random position
    task = Task(task_capacity=1, task_radius=50)
    
    # Performing experiment with different numbers of agents
    results_per_agent_num = []
    for agent_num in NUM_AGENTS:
        # Initializing agents at random positions
        agents = []
        for agent_i in range(agent_num):
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
        results_per_agent_num.append(completed_tasks)

    # Plotting results    
    x = NUM_AGENTS
    y = results_per_agent_num
    plt.plot(x, y, 'o')
    plt.title("Number of tasks solved vs. number of agents")
    plt.xlabel("# of agents")
    plt.ylabel("# of tasks solved")
    plt.show()

def experiment_3():
    # Initializing task T at random position
    task = Task(task_capacity=3, task_radius=50)
    tasks = []
    tasks.append(task)
    
    # Performing experiment with different numbers of agents
    results_per_agent_num = []
    for agent_num in NUM_AGENTS:
        # Initializing agents at random positions
        agents = []
        for agent_i in range(agent_num):
            agent = Agent()
            agents.append(agent)

        # Starting simulation
        completed_tasks = 0
        for i in range(NUM_EPOCHS):
            # Updating movement (velocity and position) of agent
            for agent in agents:
                agent.update_velocity_random(AGENT_ABSOLUTE_VELOCITY)
                agent.update_pos()
            
            for task_i in range(len(tasks)):
                task = tasks[task_i]
                task_completed = task.sufficient_agents_in_radius(agents)
                if task_completed:
                    completed_tasks += 1
                    tasks[task_i] = Task(task_capacity=3, task_radius=50)
        results_per_agent_num.append(completed_tasks)

    # Plotting results    
    x = NUM_AGENTS
    y = results_per_agent_num
    plt.plot(x, y, 'o')
    plt.title("Number of tasks solved vs. number of agents")
    plt.xlabel("# of agents")
    plt.ylabel("# of tasks solved")
    plt.show()

def experiment_4():
    # Performing experiment with different numbers of tasks
    tasks = []
    results_per_task_num = []
    for task_num in NUM_TASKS:
        task = Task(task_capacity=3, task_radius=50)
        tasks.append(task)
        
        # Performing experiment with different numbers of agents
        for agent_num in NUM_AGENTS:
            # Initializing agents at random positions
            agents = []
            for agent_i in range(agent_num):
                agent = Agent()
                agents.append(agent)

            # Starting simulation
            completed_tasks = 0
            for i in range(NUM_EPOCHS):
                # Updating movement (velocity and position) of agent
                for agent in agents:
                    agent.update_velocity_random(AGENT_ABSOLUTE_VELOCITY)
                    agent.update_pos()
                
                for task_i in range(len(tasks)):
                    task = tasks[task_i]
                    task_completed = task.sufficient_agents_in_radius(agents)
                    if task_completed:
                        completed_tasks += 1
                        tasks[task_i] = Task(task_capacity=3, task_radius=50)
        results_per_task_num.append(completed_tasks)

    # Plotting results    
    x = NUM_TASKS
    y = results_per_task_num
    plt.plot(x, y, 'o')
    plt.title("Number of tasks solved vs. number of tasks")
    plt.xlabel("# of tasks")
    plt.ylabel("# of tasks solved")
    plt.show()

if __name__ == "__main__":
    experiment_1()
    experiment_2()
    experiment_3()
    experiment_4()