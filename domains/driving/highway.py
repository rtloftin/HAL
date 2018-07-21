"""
Defines the 'highway' environment, along with the associated tasks.

This environment is a three lane highway with a single exit on the
right-hand side.  The car starts in the middle lane driving forward,
while most other cars are randomly generated.  Tasks include changing
to the right and left lanes, taking the exit, and passing a car in
front of the agent's car, before returning to the center lane.
"""

import numpy as np
import math
from .environment import Environment
from .cars import DriverCar, NPCCar
from .tasks import Task


def highway(npc=True):
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

        if npc:

            # Left lane
            count = np.random.randint(1, 3)
            distance = 2.5 + np.random.random() * 2.0
            position = 1.0 + np.random.random() * 2.0

            for _ in range(count):
                cars.append(NPCCar(7.0, position, 0.0, 0.8))
                position += distance

            # Right lane
            count = np.random.randint(1, 3)
            distance = 2.5 + np.random.random() * 2.0
            position = 2.0 + np.random.random() * 2.0

            for _ in range(count):
                cars.append(NPCCar(11.0, position, 0.0, 0.6))
                position -= distance

        return cars

    def acceleration(env, time, position):

        # Calculate acceleration
        if 0.0 >= time:
            return 0.1

        return 2 * ((position - env.y) / (time * time) - env.speed / time)

    class Exit(Task):

        def __init__(self):
            Task.__init__(self, 14.0, 15.5, 22, 23.5, 1.0)

        def reset(self):
            car = DriverCar(9.0, 1.0, 0.0, 0.75)

            return car, npcs()

        def expert(self, env):

            # Steering
            steering = 0.0

            if env.y >= 10.0:
                if -math.pi / 4.0 < env.direction:
                    steering = 0.2

            # Acceleration
            time = 0.0

            for car in env.npc:
                if car.x >= 10.0 and car.y <= 12.0:
                    time = max(time, (12.0 - car.y) / car.speed)

            return acceleration(env, time, 10.0), steering

    class Left(Task):
        def __init__(self):
            Task.__init__(self, 7.0, 15.0, 7.0, 20, 1.0)

        def reset(self):
            car = DriverCar(9.0, 1.0, 0.0, 0.75)

            return car, npcs()

        def expert(self, env):
            steering = 0.0

            if env.y >= 8.0:
                if 8.0 <= env.x and -0.5 <= env.direction:
                    steering = -0.1
                elif 8.0 >= env.x and 0.0 < env.direction:
                    steering = 0.1

            # Acceleration
            time = 0.0

            for car in env.npc:
                if car.x <= 8.0 and car.y <= 12.0:
                    time = max(time, (12.0 - car.y) / car.speed)

            return acceleration(env, time, 8.0), steering

    class Right(Task):
        def __init__(self):
            Task.__init__(self, 11.0, 15.0, 11.0, 20.0, 1.0)

        def reset(self):
            car = DriverCar(9.0, 1.0, 0.0, 0.75)

            return car, npcs()

        def expert(self, env):
            steering = 0.0

            if env.y >= 8.0:
                if 10.0 >= env.x and -0.5 <= env.direction:
                    steering = 0.1
                elif 10.0 <= env.x and 0.0 < env.direction:
                    steering = -0.1

            # Acceleration
            time = 0.0

            for car in env.npc:
                if car.x >= 10.0 and car.y <= 12.0:
                    time = max(time, (12.0 - car.y) / car.speed)

            return acceleration(env, time, 8.0), steering

    env.add_task(Exit(), "exit")
    env.add_task(Left(), "left")
    env.add_task(Right(), "right")

    env.set_task("exit")
    env.reset()

    return env
