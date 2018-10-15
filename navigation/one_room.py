"""
Implements the 'one room' environment from the BAM tutorial.
"""

from .environment import Environment, Map, Task, Region
from .sensor import RoundSensor


def one_room():
    """
    Constructs an instance of the 'One Room' environment.

    :return: an instance of the 'One Room' environment, a round sensor model attached to the environment
    """

    # Initialize map
    map = Map(26, 26)
    map.obstacle(6, 6, 14, 1)
    map.obstacle(6, 19, 14, 1)
    map.obstacle(6, 7, 1, 4)
    map.obstacle(6, 15, 1, 4)
    map.obstacle(19, 7, 1, 4)
    map.obstacle(19, 15, 1, 4)

    # Initialize tasks
    tasks = {
        "top": Task(Region(12, 24, 2, 2), [Region(10, 0, 4, 4)]),
        "bottom": Task(Region(12, 0, 2, 2), [Region(10, 20, 4, 4)])
    }

    # Initialize environment
    env = Environment(map, tasks)
    env.reset(task="top")

    # Initialize sensor model
    sensor = RoundSensor(env, 3)

    return env, sensor
