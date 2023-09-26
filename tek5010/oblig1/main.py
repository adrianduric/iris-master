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
    results_per_episode = []
    for _ in range(NUM_EPISODES):
        completed_tasks = 0
        for _ in range(NUM_EPOCHS):
            # Updating movement (velocity and position) of agent
            r1.update_velocity([task])
            r1.update_pos()
            
            # Checking if the agent is within the task radius, adding to
            # completed tasks and creating a new one if so
            if distance_euclid(task.pos, r1.pos) < task.task_radius:
                completed_tasks += 1
                task = Task(task_capacity=1, task_radius=50)
        results_per_episode.append(completed_tasks)
    
    # Plotting results
    x = np.linspace(1, NUM_EPISODES, NUM_EPISODES)
    y = results_per_episode
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
        for _ in range(agent_num):
            agent = Agent()
            agents.append(agent)

        # Starting simulation
        completed_tasks = 0
        for _ in range(NUM_EPOCHS):
            # Updating movement (velocity and position) of agent
            for agent in agents:
                agent.update_velocity([task])
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
        for _ in range(agent_num):
            agent = Agent()
            agents.append(agent)

        # Starting simulation
        completed_tasks = 0
        for _ in range(NUM_EPOCHS):
            # Updating movement (velocity and position) of agent
            for agent in agents:
                agent.update_velocity(tasks)
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
    for _ in NUM_TASKS:
        task = Task(task_capacity=3, task_radius=50)
        tasks.append(task)
        
        # Performing experiment with different numbers of agents
        for agent_num in NUM_AGENTS:
            # Initializing agents at random positions
            agents = []
            for _ in range(agent_num):
                agent = Agent()
                agents.append(agent)

            # Starting simulation
            completed_tasks = 0
            for _ in range(NUM_EPOCHS):
                # Updating movement (velocity and position) of agent
                for agent in agents:
                    agent.update_velocity(tasks)
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

def experiment_5():
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
                    # Callout is performed before updating velocities, so that each
                    # agent can evaluate whether conditions are met for it to follow
                    # the target_pos it may receive from callout.
                    agent.callout(agents)
                    agent.update_velocity(tasks)
                    agent.update_pos()
                
                for task_i in range(len(tasks)):
                    task = tasks[task_i]
                    task_completed = task.sufficient_agents_in_radius(agents)
                    if task_completed:
                        completed_tasks += 1
                        tasks[task_i] = Task(task_capacity=3, task_radius=50)
        results_per_task_num.append(completed_tasks)


def experiment(
        num_episodes: int,
        num_tasks: list,
        task_capacity: int,
        task_radius: float,
        num_agents: list,
        comm_distances: list,
        num_epochs: int
    ):

    results_per_episode = []
    for i in range(num_episodes):
  
        # Performing experiment with different numbers of tasks
        results_per_task_num = []
        tasks = []
        for task_num in num_tasks:
            # Initializing tasks at random positions
            task = Task(task_capacity=task_capacity, task_radius=task_radius)
            tasks.append(task)

            # Performing experiment with different numbers of agents
            results_per_agent_num = []
            for agent_num in num_agents:

                # Performing experiment with different communication distances
                results_per_comm_dist = []
                for comm_dist in comm_distances: 
                    # Initializing agents at random positions
                    agents = []
                    for agent_i in range(agent_num):
                        agent = Agent(comm_dist=comm_dist)
                        agents.append(agent)

                    # Performing simulation for a specified number of epochs
                    completed_tasks = 0
                    for i in range(num_epochs):
                        # Updating movement (velocity and position) of agent
                        for agent in agents:
                            # Callout is performed before updating velocities, so that each agent can evaluate
                            # (during velocity update) whether conditions are met for it to follow the target_pos
                            # it may receive from callout.
                            agent.callout(agents)
                            agent.update_velocity(tasks)
                            agent.update_pos()
                        
                        for task_i in range(len(tasks)):
                            task = tasks[task_i]
                            task_completed = task.sufficient_agents_in_radius(agents)
                            if task_completed:
                                completed_tasks += 1
                                tasks[task_i] = Task(task_capacity=3, task_radius=50)
                    results_per_comm_dist.append(completed_tasks)
                results_per_agent_num.append(np.average(np.array(results_per_comm_dist)))
            results_per_task_num.append(np.average(np.array(results_per_agent_num)))
        results_per_episode.append(np.average(np.array(results_per_task_num)))
    return results_per_episode, results_per_task_num, results_per_agent_num, results_per_comm_dist

if __name__ == "__main__":
    experiment_1()
    #experiment_2()
    #experiment_3()
    #experiment_4()
    #experiment_5()
    task_a_results, _, _, _ = experiment(num_episodes=20, num_tasks=[1], task_capacity=1, task_radius=50, num_agents=[1], comm_distances=[0], num_epochs=5000)
    
    x = np.linspace(1, 20, 20)
    y = task_a_results
    plt.plot(x, y, 'o')
    plt.title("Number of tasks solved per run")
    plt.xlabel("Run #")
    plt.ylabel("# of tasks solved")
    plt.show()
