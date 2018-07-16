"""
Defines the 'highway' environment, along with the associated tasks.

This environment is a three lane highway with a single exit on the
right-hand side.  The car starts in the middle lane driving forward,
while most other cars are randomly generated.  Tasks include changing
to the right and left lanes, taking the exit, and passing a car in
front of the agent's car, before returning to the center lane.
"""

from .environment import Environment
from .environment import DriverCar, NPCCar
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

    class Exit(Task):

        def __init__(self):
            Task.__init__(self, 14.0, 15.5, 22, 23.5, 1.0)

        def reset(self):
            car = DriverCar(9.0, 1.0, 0.0, 0.0)
            npc = []

            return car, npc

    class Left(Task):
        def __init__(self):
            Task.__init__(self, 7.0, 12.0, 7.0, 20, 1.0)

        def reset(self):
            car = DriverCar(9.0, 1.0, 0.0, 0.0)
            npc = []

            return car, npc

    class Right(Task):
        def __init__(self):
            Task.__init__(self, 11.0, 12.0, 11.0, 20.0, 1.0)

        def reset(self):
            car = DriverCar(9.0, 1.0, 0.0, 0.0)
            npc = []

            return car, npc

    env.add_task(Exit(), "exit")
    env.add_task(Left(), "left")
    env.add_task(Right(), "right")

    env.set_task("exit")
    env.reset()

    return env
