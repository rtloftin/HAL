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
    map = Map(40, 40)
    map.obstacle(0, 32, 2, 1)
    map.obstacle(38, 32, 2, 1)
    map.obstacle(6, 32, 28, 1)
    map.obstacle(8, 6, 8, 1)
    map.obstacle(24, 6, 8, 1)
    map.obstacle(8, 6, 1, 26)
    map.obstacle(31, 6, 1, 26)

    # Initialize tasks
    # tasks = {
    #    "right": Task(Region(24, 36, 1, 1), [Region(8, 0, 4, 4), Region(18, 0, 4, 4), Region(28, 0, 4, 4)]),
    #    "left": Task(Region(15, 36, 1, 1), [Region(8, 0, 4, 4), Region(18, 0, 4, 4), Region(28, 0, 4, 4)])
    # }

    tasks = {
       "right": Task(Region(24, 36, 1, 1), [Region(0, 0, 40, 5)]),
       "left": Task(Region(15, 36, 1, 1), [Region(0, 0, 40, 5)])
    }

    # Initialize environment
    env = Environment(map, tasks)
    env.reset(task="right")

    # Initialize sensor model
    sensor = RoundSensor(env, 4)

    return env, sensor
