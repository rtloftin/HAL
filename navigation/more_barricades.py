"""
Implements the 'Barricades' environment.
"""

from .environment import Environment, Map, Task, Region
from .sensor import RoundSensor


def more_barricades():
    """
    Constructs an instance of the 'Barricades' environment.

    :return: an instance of the 'Barricades' environment, a round sensor model attached to the environment
    """

    # Initialize map
    map = Map(110, 40)

    map.obstacle(10, 32, 11, 1)
    map.obstacle(10, 8, 1, 24)
    map.obstacle(20, 8, 1, 24)

    map.obstacle(30, 32, 11, 1)
    map.obstacle(30, 8, 1, 24)
    map.obstacle(40, 8, 1, 24)

    map.obstacle(50, 32, 11, 1)
    map.obstacle(50, 8, 1, 24)
    map.obstacle(60, 8, 1, 24)

    map.obstacle(70, 32, 11, 1)
    map.obstacle(70, 8, 1, 24)
    map.obstacle(80, 8, 1, 24)

    map.obstacle(90, 32, 11, 1)
    map.obstacle(90, 8, 1, 24)
    map.obstacle(100, 8, 1, 24)


    tasks = {
        "right": Task(Region(17, 35, 1, 1), [Region(0, 0, 110, 8)]),
        "left": Task(Region(53, 35, 1, 1), [Region(0, 0, 110, 8)])
    }

    # Initialize environment
    env = Environment(map, tasks)
    env.reset(task="right")

    # Initialize sensor model
    sensor = RoundSensor(env, 4)

    return env, sensor
