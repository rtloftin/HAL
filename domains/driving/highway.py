"""
Defines the 'highway' environment, along with the associated tasks.

This environment is a three lane highway with a single exit on the
right-hand side.  The car starts in the middle lane driving forward,
while most other cars are randomly generated.  Tasks include changing
to the right and left lanes, taking the exit, and passing a car in
front of the agent's car, before returning to the center lane.
"""

import numpy as np
from .environment import Environment
from .cars import DriverCar, NPCCar
from .tasks import Task


def highway():
    """
    Constructs an instance of the highway environment.

    :return: an instance of the highway environment
    """

    # Initialize environment
    env = Environment(20.0, 20.0, 5.0, 20)

    env.add_wall(6.0, 0.0, 6.0, 20.0)
    env.add_wall(12.0, 0.0, 12.0, 12.0)
    env.add_wall(12.0, 15.0, 12.0, 20.0)
    env.add_wall(12.0, 12.0, 20.0, 20.0)
    env.add_wall(12.0, 15.0, 17.0, 20.0)

    # Define tasks

    def npcs():
        cars = []

        # Left lane
        count = np.random.randint(1, 4)
        distance = 2.5 + np.random.random() * 2.0
        position = 1.0 + np.random.random() * 2.0

        for _ in range(count):
            cars.append(NPCCar(7.0, position, 0.0, 0.8))
            position += distance

        # Right lane
        count = np.random.randint(1, 4)
        distance = 2.5 + np.random.random() * 2.0
        position = 2.0 + np.random.random() * 2.0

        for _ in range(count):
            cars.append(NPCCar(11.0, position, 0.0, 0.6))
            position -= distance

        return cars

    class Exit(Task):

        def __init__(self):
            Task.__init__(self, 14.0, 15.5, 22, 23.5, 1.0)

        def reset(self):
            car = DriverCar(9.0, 1.0, 0.0, 0.75)

            return car, npcs()

    class Left(Task):
        def __init__(self):
            Task.__init__(self, 7.0, 15.0, 7.0, 20, 1.0)

        def reset(self):
            car = DriverCar(9.0, 1.0, 0.0, 0.75)

            return car, npcs()

    class Right(Task):
        def __init__(self):
            Task.__init__(self, 11.0, 15.0, 11.0, 20.0, 1.0)

        def reset(self):
            car = DriverCar(9.0, 1.0, 0.0, 0.75)

            return car, npcs()

    env.add_task(Exit(), "exit")
    env.add_task(Left(), "left")
    env.add_task(Right(), "right")

    env.set_task("exit")
    env.reset()

    return env
