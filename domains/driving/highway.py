"""
Defines the 'highway' environment, along with the associated tasks.

This environment is a three lane highway with a single exit on the
right-hand side.  The car starts in the middle lane driving forward,
while most other cars are randomly generated.  Tasks include changing
to the right and left lanes, taking the exit, and passing a car in
front of the agent's car, before returning to the center lane.
"""

import driving.Environment
import driving.DriverCar
import driving.NPCCar
import task.Task


def highway():
    """
    Constructs an instance of the highway environment.

    :return: an instance of the highway environment
    """

    # Initialize environment
    env = Environment(20, 20, 5, 20)

    env.add_wall(6, 0, 6, 20)
    env.add_wall(12, 0, 12, 12)
    env.add_wall(12, 15, 12, 20)
    env.add_wall(12, 12, 20, 20)
    env.add_wall(12, 15, 17, 20)

    # Define tasks

    class Exit(Task):

        def __init__(self):
            Task.__init__(14, 15.5, 22, 23.5, 1)

        def reset(self):
            car = DriverCar(9, 1, 0, 0)
            npc = []

            return car, npc

    class Left(Task):
        def __init__(self):
            Task.__init__(7, 12, 7, 20, 1)

        def reset(self):
            car = DriverCar(9, 1, 0, 0)
            npc = []

            return car, npc

    class Right(Task):
        def __init__(self):
            Task.__init__(11, 12, 11, 20, 1)

        def reset(self):
            car = DriverCar(9, 1, 0, 0)
            npc = []

            return car, npc

    env.add_task(Exit(), "exit")
    env.add_task(Left(), "left")
    env.add_task(Right(), "right")

    env.set_task("exit")
    env.reset()

    return env
