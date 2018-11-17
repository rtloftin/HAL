"""
Implements the 'Big Three Rooms' environment from the document.
"""

from .environment import Environment, Map, Task, Region
from .sensor import RoundSensor, SquareSensor


def big_three_rooms():
    """
    Constructs an instance of the 'Big Three Rooms' environment.

    :return: an instance of the 'Big Three Rooms' environment, a round sensor model attached to the environment
    """

    # NEW

    # Initialize map
    map = Map(80, 80)
    map.obstacle(12, 13, 54, 1)
    map.obstacle(12, 38, 54, 1)
    map.obstacle(12, 64, 54, 1)
    map.obstacle(12, 14, 1, 8)
    map.obstacle(12, 30, 1, 18)
    map.obstacle(12, 56, 1, 8)
    map.obstacle(65, 14, 1, 8)
    map.obstacle(65, 30, 1, 18)
    map.obstacle(65, 56, 1, 8)
    map.obstacle(38, 14, 1, 24)

    # Initialize tasks
    # tasks = {
    #     "right": Task(Region(76, 38, 1, 1), [Region(0, 36, 8, 8), Region(0, 16, 8, 8), Region(0, 56, 8, 8)]),
    #     "left": Task(Region(2, 38, 1, 1), [Region(72, 36, 8, 8), Region(72, 16, 8, 8), Region(72, 56, 8, 8)])
    # }

    tasks = {
       "right": Task(Region(77, 38, 1, 1), [Region(0, 0, 8, 80)]),
       "left": Task(Region(2, 38, 1, 1), [Region(72, 0, 8, 80)])
    }

    # Initialize environment
    env = Environment(map, tasks)
    env.reset(task="right")

    # Initialize sensor model
    sensor = RoundSensor(env, 4)

    return env, sensor
