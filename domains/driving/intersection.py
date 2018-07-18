"""
Defines the 'intersection' environment, along with the associated tasks.

This environment is a simple four-way intersection, with the agent's car
coming from one direction, and random NPC cars coming from the other directions.

Tasks include turning left and right, and going straight.  Each road has two lanes,
and reward is given for reaching the desired lane without hitting anything.
"""

import numpy as np
import math
from .environment import Environment
from .cars import DriverCar, NPCCar
from .tasks import Task


def intersection():
    """
    Constructs an instance of the intersection environment.

    :return: an instance of the intersection environment
    """

    # Initialize environment
    env = Environment(20, 20, 5, 40)

    env.add_wall(8, 0, 8, 6)
    env.add_wall(12, 0, 12, 6)
    env.add_wall(8, 14, 8, 20)
    env.add_wall(12, 14, 12, 20)
    env.add_wall(0, 8, 6, 8)
    env.add_wall(0, 12, 6, 12)
    env.add_wall(14, 8, 20, 8)
    env.add_wall(14, 12, 20, 12)
    env.add_wall(12, 6, 14, 8)
    env.add_wall(14, 12, 12, 14)
    env.add_wall(8, 14, 6, 12)
    env.add_wall(6, 8, 8, 6)

    # Define tasks

    def npcs():
        cars = []

        # Left to right
        count = np.random.randint(1, 3)
        distance = 4.0 + np.random.random() * 2.0
        position = 1.0 + np.random.random() * 2.0

        for _ in range(count):
            cars.append(NPCCar(position, 9.0, -0.5 * math.pi, 0.75))
            position -= distance

        # Right to left
        count = np.random.randint(1, 3)
        distance = 4.0 + np.random.random() * 2.0
        position = 19.0 - np.random.random() * 2.0

        for _ in range(count):
            cars.append(NPCCar(position, 11.0, 0.5 * math.pi, 0.75))
            position += distance

        return cars

    class Straight(Task):

        def __init__(self):
            Task.__init__(self, 11, 16, 11, 20, 1)

        def reset(self):
            car = DriverCar(11, 1, 0, 0.75)

            return car, npcs()

    class Left(Task):
        def __init__(self):
            Task.__init__(self, 4, 11, 0, 11, 1)

        def reset(self):
            car = DriverCar(11, 1, 0, 0.75)

            return car, npcs()

    class Right(Task):
        def __init__(self):
            Task.__init__(self, 9, 16, 20, 16, 1)

        def reset(self):
            car = DriverCar(11, 1, 0, 0.75)

            return car, npcs()

    env.add_task(Straight(), "straight")
    env.add_task(Left(), "left")
    env.add_task(Right(), "right")

    env.set_task("straight")
    env.reset()

    return env
