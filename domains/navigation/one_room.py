"""
Implements the 'one room' environment from the BAM tutorial.
"""

from .environment import Environment, Task
from .sensor import RoundSensor


def one_room():
    """
    Constructs an instance of the 'Three Rooms' environment.

    :return: an instance of the 'Three Rooms' environment
    """

    # Initialize sensor model
    sensor = RoundSensor(4)

    # Initialize environment
    env = Environment(26, 26, sensor)
    env.add_obstacle(6, 6, 14, 1)
    env.add_obstacle(6, 19, 14, 1)
    env.add_obstacle(6, 7, 1, 4)
    env.add_obstacle(6, 15, 1, 4)
    env.add_obstacle(19, 7, 1, 4)
    env.add_obstacle(19, 15, 1, 4)

    # Define tasks

    class Right(Task):
        def __init__(self):
            Task.__init__(self, 24, 12, 2, 2)

        def reset(self):
            return 0, 12

    class Left(Task):
        def __init__(self):
            Task.__init__(self, 0, 12, 2, 2)

        def reset(self):
            return 25, 12

    env.add_task(Right(), 'right')
    env.add_task(Left(), 'left')

    env.set_task('right')
    env.reset()

    return env
