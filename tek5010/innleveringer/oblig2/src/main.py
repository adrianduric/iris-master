from utils import *
from constants import *
from agent import *
from task import *
import numpy as np
import matplotlib.pyplot as plt


def experiment():
    # Performing experiment with different communication distances
    for comm_dist in COMM_DISTANCES:   
        # Initializing tasks at random positions
        tasks = []
        for _ in range(NUM_TASKS):
            task = Task(task_capacity=TASK_CAPACITY, task_radius=TASK_RADIUS)
            tasks.append(task)

        # Initializing agents at random positions
        agents = []
        for _ in range(NUM_AGENTS):
            agent = Agent(comm_dist=comm_dist)
            agents.append(agent)

        # Starting simulation
        results = np.zeros((NUM_EPOCHS))
        completed_tasks = 0
        for i in range(NUM_EPOCHS):
            for task_i in range(len(tasks)):
                task = tasks[task_i]
                task_completed = task.sufficient_agents_in_radius(agents, invoke_auction=True)
                if task_completed:
                    completed_tasks += 1
                    tasks[task_i] = Task(task_capacity=TASK_CAPACITY, task_radius=TASK_RADIUS)

            for agent in agents:
                agent.update_velocity()
                agent.update_pos()
            results[i] = completed_tasks
        x = np.linspace(1, NUM_EPOCHS, NUM_EPOCHS)
        y = results
        plt.plot(x, y, label=f'Rd = {comm_dist}')
        print(f"Simulations for Rd = {comm_dist} complete.")

    # Plotting results
    plt.title("TASK A)")
    plt.xlabel("Time (# of epochs)")
    plt.ylabel("# of tasks solved")
    plt.legend()
    plt.savefig(fname='figures/task_a')
    plt.close()


if __name__ == "__main__":
    experiment()