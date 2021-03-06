"""
Defines the core classes representing a driving environment.

Specific environments will extend the environment class

WE MAY WANT TO REORGANIZE THIS API AT SOME POINT
"""

import math
from .sensor import Sensor
from .collision import Collision
from .cars import DriverCar
from .tasks import Task


class Wall:
    """
    A wall represented as a line segment
    """

    def __init__(self, x0, y0, x1, y1):
        """
        Initializes the wall

        :param x0: the first x coordinate of the segment
        :param y0: the first y coordinate of the segment
        :param x1: the second x coordinate of the segment
        :param y1: the second y coordinate of the segment
        """

        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1


class Environment:
    """
    Represents a simulation of the driving domain.

    This will include a sensor model as well.
    """

    def __init__(self, width, height, radius, resolution):
        """
        Initializes the environment.

        :param width:
        :param height:
        :param radius:
        :param resolution:
        """

        self._width = width
        self._height = height

        self._radius = radius
        self._resolution = resolution

        self._car = None
        self._cars = []

        self._walls = []

        self._tasks = {}
        self._task = None

        self._sensor = None
        self._collisions = None

        self._reward = 0.0
        self._complete = False

    def add_wall(self, x0, y0, x1, y1):
        """
        Adds a new wall to the environment.

        :param x0: the first x coordinate of the wall
        :param y0: the first y coordinate of the wall
        :param x1: the second x coordinate of the wall
        :param y1: the second y coordinate of the wall
        """

        self._walls.append(Wall(x0, y0, x1, y1))

    def add_task(self, task, name):
        """
        Adds a new task to the environment.

        :param task: the task object
        :param name: the name of the task
        """

        self._tasks[name] = task

    def set_task(self, name):
        """
        Sets the current task.

        :param name: the name of the task
        """

        self._task = self._tasks[name]

    def get_tasks(self):
        """
        Gets a list of task names defined for this environment.

        :return: a list of task names
        """

        return self._tasks.keys()

    def reset(self):
        """
        Resets the environment to a random initial state.

        Only works if there is a current task set, since the current
        task defines how the environment will be initialized.
        """

        self._complete = False
        self._reward = 0.0

        self._car, self._cars = self._task.reset()

        self._sensor = Sensor(self._car, self._cars, self._walls, self._radius, self._resolution)
        self._collisions = Collision(self._car, self._cars, self._walls, self._width, self._height, 0.5)

    def update(self, acceleration, steering, delta):
        """
        Updates the state of the environment according to the
        dynamics equations.

        :param acceleration: acceleration of the vehicle (negative for braking)
        :param steering: steering rate (positive for left negative for right)
        :param delta: the size of the time step for this update
        """

        # Update driver car
        self._car.update(acceleration, steering, delta)

        # Update NPC cars
        for car in self._cars:
            car.update(delta)

        # Update collision model
        self._collisions.update()

        # Update sensor model
        self._sensor.update()

        # Update task model
        self._reward, self._complete = self._task.evaluate(self.x, self.y, self._collisions.is_collision)

    def expert(self):
        """
        Gets the output of the expert policy for the current
        task in the current state.

        :return: the acceleration and steering for the current state
        """

        return self._task.expert(self)

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def x(self):
        return self._car.x

    @property
    def y(self):
        return self._car.y

    @property
    def direction(self):
        return self._car.theta

    @property
    def steering(self):
        return self._car.phi

    @property
    def speed(self):
        return self._car.speed

    @property
    def sensor(self):
        return self._sensor.vector

    @property
    def reward(self):
        return self._reward

    @property
    def complete(self):
        return self._complete or self._collisions.is_collision

    @property
    def npc(self):
        return self._cars

    @property
    def walls(self):
        return self._walls


def empty():
    """
    Builds an empty environment to test the driving dynamics.

    :return: an empty driving environment
    """

    # Initialize environment
    env = Environment(20.0, 20.0, 5.0, 20)

    # Define task

    class Nothing(Task):

        def __init__(self):
            Task.__init__(self, -1.0, -1.0, -2.0, -2.0, 1.0)

        def reset(self):
            car = DriverCar(10.0, 1.0, 0.0, 0.0)
            npc = []

            return car, npc

    env.add_task(Nothing(), "none")
    env.set_task("none")
    env.reset()

    return env
