"""
Implements the 'Three Rooms' environment from the document.
"""

from .environment import Environment, Task
from .sensor import RoundSensor


def three_rooms():
    """
    Constructs an instance of the 'Three Rooms' environment.

    :return: an instance of the 'Three Rooms' environment
    """

    # Initialize sensor model
    sensor = RoundSensor(4)

    # Initialize environment
    env = Environment(39, 39, sensor)
    env.add_obstacle(6, 6, 27, 1)
    env.add_obstacle(6, 19, 27, 1)
    env.add_obstacle(6, 32, 27, 1)
    env.add_obstacle(6, 7, 1, 4)
    env.add_obstacle(6, 15, 1, 9)
    env.add_obstacle(6, 28, 1, 4)
    env.add_obstacle(32, 7, 1, 4)
    env.add_obstacle(32, 15, 1, 9)
    env.add_obstacle(32, 28, 1, 4)
    env.add_obstacle(19, 7, 1, 12)

    # Define tasks

    class Right(Task):

        def __init__(self):
            Task.__init__(self, 37, 18, 2, 3)

        def reset(self):
            return 0, 19

    class Left(Task):
        def __init__(self):
            Task.__init__(self, 0, 18, 2, 3)

        def reset(self):
            return 38, 19

    env.add_task(Right(), 'right')
    env.add_task(Left(), 'left')

    env.set_task('right')
    env.reset()

    return env
