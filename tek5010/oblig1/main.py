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
            r1.update_velocity()
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
    plt.title("TASK A)")
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
                agent.update_velocity()
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
    plt.title("TASK B)")
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
                agent.update_velocity()
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
    plt.title("TASK C)")
    plt.xlabel("# of agents")
    plt.ylabel("# of tasks solved")
    plt.show()

def experiment_4():

    # Storing results across all episodes, per task_num
    results = {}
    for task_num in NUM_TASKS:
        results[task_num] = []

    for i in range(NUM_EPISODES):
        # Performing experiment with different numbers of tasks
        tasks = []
        for task_num in NUM_TASKS:
            for _ in range(task_num):      
                task = Task(task_capacity=3, task_radius=50)
                tasks.append(task)
            
            # Initializing agents at random positions
            agents = []
            for _ in range(30): # For purpose of experiment, assuming R=30 agents
                agent = Agent()
                agents.append(agent)

            # Starting simulation
            completed_tasks = 0
            for _ in range(NUM_EPOCHS):
                # Updating movement (velocity and position) of agent
                for agent in agents:
                    agent.update_velocity()
                    agent.update_pos()
                
                for task_i in range(len(tasks)):
                    task = tasks[task_i]
                    task_completed = task.sufficient_agents_in_radius(agents)
                    if task_completed:
                        completed_tasks += 1
                        tasks[task_i] = Task(task_capacity=3, task_radius=50)
            results[task_num].append(completed_tasks)
        print(f"Episode {i + 1} complete.")

    # Plotting results
    median_results = {x: np.median(results[x]) for x in NUM_TASKS}
    average_results = {x: np.mean(results[x]) for x in NUM_TASKS}

    for x in NUM_TASKS:
        x_offsets = np.random.uniform(-0.5, 0.5, len(results[x]))  # Random offsets
        plt.scatter([x + offset for offset in x_offsets], results[x], color='blue')

    plt.plot(NUM_TASKS, [median_results[x] for x in NUM_TASKS], color='red', label='Median # tasks completed', marker='o')
    plt.plot(NUM_TASKS, [average_results[x] for x in NUM_TASKS], color='green', label='Average # tasks completed', marker='o')

    plt.title("TASK D)")
    plt.xlabel("# of tasks")
    plt.ylabel("# of tasks solved")
    plt.legend()
    plt.show()

def experiment_5():
    # Storing results across all episodes, per comm_dist
    results = {}
    for comm_dist in COMM_DISTANCES:
        results[comm_dist] = []

    for i in range(NUM_EPISODES):
        # Performing experiment with different communication distances
        for comm_dist in COMM_DISTANCES:   
            # Initializing tasks at random positions
            tasks = []
            for _ in range(2): # Assuming T=2 tasks
                task = Task(task_capacity=3, task_radius=50)
                tasks.append(task)

            # Initializing agents at random positions
            agents = []
            for _ in range(30): # Assuming R=30 agents
                agent = Agent(comm_dist=comm_dist)
                agents.append(agent)

            # Starting simulation
            completed_tasks = 0
            for _ in range(NUM_EPOCHS):
                for task_i in range(len(tasks)):
                    task = tasks[task_i]
                    task_completed = task.sufficient_agents_in_radius(agents)
                    if task_completed:
                        completed_tasks += 1
                        tasks[task_i] = Task(task_capacity=3, task_radius=50)

                # Updating movement (velocity and position) of agent
                for agent in agents:
                    # Callout is performed before updating velocities, so that each
                    # agent can evaluate whether conditions are met for it to follow
                    # the target_pos it may receive from callout.
                    if agent.inside_task_radius:
                        agent.callout(agents)
                    agent.update_velocity()
                    agent.update_pos()
            results[comm_dist].append(completed_tasks)
        print(f"Episode {i + 1} complete.")

    # Plotting results
    median_results = {x: np.median(results[x]) for x in COMM_DISTANCES}
    average_results = {x: np.mean(results[x]) for x in COMM_DISTANCES}

    for x in COMM_DISTANCES:
        x_offsets = np.random.uniform(-20, 20, len(results[x]))  # Random offsets
        plt.scatter([x + offset for offset in x_offsets], results[x], color='blue')

    plt.plot(COMM_DISTANCES, [median_results[x] for x in COMM_DISTANCES], color='red', label='Median # tasks completed', marker='o')
    plt.plot(COMM_DISTANCES, [average_results[x] for x in COMM_DISTANCES], color='green', label='Average # tasks completed', marker='o')

    plt.title("TASK E)")
    plt.xlabel("Communication distance")
    plt.ylabel("# of tasks solved")
    plt.legend()
    plt.show()

def experiment_6():
    # Storing results across all episodes, per comm_dist
    results = {}
    for comm_dist in COMM_DISTANCES:
        results[comm_dist] = []

    for i in range(NUM_EPISODES):
        # Performing experiment with different communication distances
        for comm_dist in COMM_DISTANCES:   
            # Initializing tasks at random positions
            tasks = []
            for _ in range(2): # Assuming T=2 tasks
                task = Task(task_capacity=3, task_radius=50)
                tasks.append(task)

            # Initializing agents at random positions
            agents = []
            for _ in range(30): # Assuming R=30 agents
                agent = Agent(comm_dist=comm_dist)
                agents.append(agent)

            # Starting simulation
            completed_tasks = 0
            for _ in range(NUM_EPOCHS):
                for task_i in range(len(tasks)):
                    task = tasks[task_i]
                    task_completed = task.sufficient_agents_in_radius(agents, invoke_calloff=True)
                    if task_completed:
                        completed_tasks += 1
                        tasks[task_i] = Task(task_capacity=3, task_radius=50)

                # Updating movement (velocity and position) of agent
                for agent in agents:
                    # Callout is performed before updating velocities, so that each
                    # agent can evaluate whether conditions are met for it to follow
                    # the target_pos it may receive from callout.
                    if agent.inside_task_radius:
                        agent.callout(agents)
                    agent.update_velocity()
                    agent.update_pos()
            results[comm_dist].append(completed_tasks)
        print(f"Episode {i + 1} complete.")

    # Plotting results
    median_results = {x: np.median(results[x]) for x in COMM_DISTANCES}
    average_results = {x: np.mean(results[x]) for x in COMM_DISTANCES}

    for x in COMM_DISTANCES:
        x_offsets = np.random.uniform(-20, 20, len(results[x]))  # Random offsets
        plt.scatter([x + offset for offset in x_offsets], results[x], color='blue')

    plt.plot(COMM_DISTANCES, [median_results[x] for x in COMM_DISTANCES], color='red', label='Median # tasks completed', marker='o')
    plt.plot(COMM_DISTANCES, [average_results[x] for x in COMM_DISTANCES], color='green', label='Average # tasks completed', marker='o')

    plt.title("TASK F)")
    plt.xlabel("Communication distance")
    plt.ylabel("# of tasks solved")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    print("\n\n################### TASK A) ###################\n\n")
    experiment_1()
    print("\n\n################### TASK B) ###################\n\n")
    experiment_2()
    print("\n\n################### TASK C) ###################\n\n")
    experiment_3()
    print("\n\n################### TASK D) ###################\n\n")
    experiment_4()
    print("\n\n################### TASK E) ###################\n\n")
    experiment_5()
    print("\n\n################### TASK F) ###################\n\n")
    experiment_6()