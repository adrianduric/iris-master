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
            abs_velocity: float = AGENT_ABSOLUTE_VELOCITY,
            comm_dist: float = 0.0
            ):
        x = np.random.random()*1000 if x is None else x
        y = np.random.random()*1000 if y is None else y
        self.pos = np.array([x, y])
        self.velocity = np.array([vx, vy])
        self.abs_velocity = abs_velocity
        self.comm_dist = comm_dist
        self.target_pos = None
        self.inside_task_radius = False

    def is_in_task_radius(self, tasks: list):
        """
        Checks if agent is within radius of any of the given tasks. If so, sets the
        inside_task_radius flag to True and returns True. Else, returns False and sets
        the flag to False.
        """
        self.inside_task_radius = False
        for task in tasks:
            if distance_euclid(self.pos, task.pos) < task.task_radius:
                self.inside_task_radius = True
                return True
        return False

    def has_reached_target(self):
        """
        Checks if agent is within distance=abs_velocity of target_pos. If so, returns flag True.
        Else, returns False.
        """

        abs_distance = distance_euclid(self.pos, self.target_pos)
        if abs_distance >= self.abs_velocity:
            return True
        return False

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

    def update_velocity(self, tasks):
        """
        If target_pos is specified (not None) and the agent is not within abs_velocity range of it,
        and the agent is not already within any task radius, sets agent velocity towards that position.

        Otherwise, makes the agent's movement random by changing the velocity in each direction
        to some random number. Components vx and vy are set so that the absolute velocity sums up to
        abs_velocity.
        """

        # Removes target_pos (should it be set) if agent is inside any task radius
        if self.is_in_task_radius(tasks):
            self.target_pos = None

        # Goes towards target_pos if specified and absolute distance between pos and target_pos
        # is less than abs_velocity
        if self.target_pos is not None and not self.has_reached_target():
            self.velocity = self.target_pos - self.pos
            norm = np.linalg.norm(self.velocity)
            self.velocity = (self.velocity / norm) * self.abs_velocity
            return
        
        # If not, removes target_pos and initializes random movement
        self.target_pos = None

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
        self.velocity = self.velocity * self.abs_velocity

    def callout(self, agents: list):
        """
        When the agent is within the task radius of any task, it emits a signal to other agents
        within comm_dist to make them go towards that location, by setting their target_pos to the
        position of the agent emitting the signal.

        The called upon agents will then go towards the coordinate from which the signal was emitted until:
        a) they reach the signal location +/- abs_velocity (to prevent overshooting and agents
        getting stuck).
        b) they find themselves within a task radius themselves.
        The above conditions are checked for each agent when their velocities are updated.
        """

        # Checking whether the agent is within any task radius
        if self.inside_task_radius:
            # If so, send a signal to any agent within comm_dist
            for agent in agents:        
                # Checking if the agent is within the comm_dist
                if distance_euclid(self.pos, agent.pos) < self.comm_dist:
                    agent.target_pos = self.pos
                        

