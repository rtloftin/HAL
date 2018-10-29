"""
Implements the 'Three Rooms' environment from the document.
"""

from .environment import Environment, Map, Task, Region
from .sensor import RoundSensor


def three_rooms():
    """
    Constructs an instance of the 'Three Rooms' environment.

    :return: an instance of the 'Three Rooms' environment, a round sensor model attached to the environment
    """

    # NEW

    # Initialize map
    map = Map(40, 40)
    map.obstacle(6, 6, 14, 1)
    map.obstacle(6, 19, 14, 1)
    map.obstacle(6, 7, 1, 4)
    map.obstacle(6, 15, 1, 4)
    map.obstacle(19, 7, 1, 4)
    map.obstacle(19, 15, 1, 4)

    # Initialize tasks
    tasks = {
        "right": Task(Region(37, 18, 2, 3), [Region(0, 17, 2, 4)]),
        "left": Task(Region(0, 18, 2, 3), [Region(37, 17, 2, 4)])
    }

    # Initialize environment
    env = Environment(map, tasks)
    env.reset(task="right")

    # Initialize sensor model
    sensor = RoundSensor(env, 40)

    return env, sensor
