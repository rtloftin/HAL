"""
Implements the 'Three Rooms' environment from the document.
"""

from .environment import Environment, Map, Task, Region
from .sensor import RoundSensor, SquareSensor


def three_rooms():
    """
    Constructs an instance of the 'Three Rooms' environment.

    :return: an instance of the 'Three Rooms' environment, a round sensor model attached to the environment
    """

    # NEW

    # Initialize map
    map = Map(40, 40)
    map.obstacle(6, 6, 27, 1)
    map.obstacle(6, 19, 27, 1)
    map.obstacle(6, 32, 27, 1)
    map.obstacle(6, 7, 1, 4)
    map.obstacle(6, 15, 1, 9)
    map.obstacle(6, 28, 1, 4)
    map.obstacle(32, 7, 1, 4)
    map.obstacle(32, 15, 1, 9)
    map.obstacle(32, 28, 1, 4)
    map.obstacle(19, 7, 1, 12)

    # Initialize tasks
    # tasks = {
    #    "right": Task(Region(38, 19, 1, 1), [Region(0, 18, 4, 4), Region(0, 8, 4, 4), Region(0, 28, 4, 4)]),
    #    "left": Task(Region(0, 19, 1, 1), [Region(36, 18, 4, 4), Region(36, 8, 4, 4), Region(36, 28, 4, 4)])
    # }

    tasks = {
       "right": Task(Region(38, 19, 1, 1), [Region(0, 18, 4, 4)]),
       "left": Task(Region(0, 19, 1, 1), [Region(36, 18, 4, 4)])
    }

    # Initialize environment
    env = Environment(map, tasks)
    env.reset(task="right")

    # Initialize sensor model
    sensor = RoundSensor(env, 4)

    return env, sensor
