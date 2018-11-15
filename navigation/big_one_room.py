"""
Implements the 'big one room' environment from the BAM tutorial.
"""

from .environment import Environment, Map, Task, Region
from .sensor import RoundSensor


def big_one_room():
    """
    Constructs an instance of the 'Big One Room' environment.

    :return: an instance of the 'Big One Room' environment, a round sensor model attached to the environment
    """

    # Initialize map
    map = Map(80, 80)
    map.obstacle(0, 64, 4, 1)
    map.obstacle(76, 64, 4, 1)
    map.obstacle(12, 64, 56, 1)
    map.obstacle(16, 12, 16, 1)
    map.obstacle(48, 12, 16, 1)
    map.obstacle(16, 12, 1, 52)
    map.obstacle(62, 12, 1, 52)

    # Initialize tasks
    # tasks = {
    #     "right": Task(Region(48, 72, 1, 1), [Region(16, 0, 8, 8), Region(36, 0, 8, 8), Region(56, 0, 4, 4)]),
    #     "left": Task(Region(30, 72, 1, 1), [Region(16, 0, 8, 8), Region(36, 0, 8, 8), Region(56, 0, 4, 4)])
    # }

    tasks = {
        "right": Task(Region(48, 72, 1, 1), [Region(0, 0, 80, 8)]),
        "left": Task(Region(30, 72, 1, 1), [Region(0, 0, 80, 8)])
    }

    # Initialize environment
    env = Environment(map, tasks)
    env.reset(task="right")

    # Initialize sensor model
    sensor = RoundSensor(env, 4)

    return env, sensor
