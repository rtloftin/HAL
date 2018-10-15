import numpy as np

from enum import IntEnum
from collections import Iterable


class Action(IntEnum):
    """
    An enum describing the set of possible actions.
    """

    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


class Region:
    """
    Represents a rectangular region in the grid.
    """

    def __init__(self, x, y, width, height):
        """
        Defines the region

        :param x: the x coordinate of the top left corner of the region
        :param y: the y coordinate of the top left corner of the region
        :param width: the width of the region
        :param height: the height of the region
        """
        self._x = x
        self._y = y
        self._width = width
        self._height = height

    def contains(self, x, y):
        """
        Tests whether a given cell is within the region.

        :param x: the x coordinate of the cell
        :param y: the y coordinate of the cell
        :return: True if the location is within the region, False otherwise
        """

        dx = x - self._x
        dy = y - self._y

        return (0 <= dx < self._width) and (0 <= dy < self._height)

    def sample(self):
        """
        Randomly samples a cell from within this region.

        :return: the x and y coordinates of the cell
        """

        x = self._x + np.random.randint(self._width)
        y = self._y + np.random.randint(self._height)

        return x, y

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height


class Task:
    """
    Represents a single task, which is composed of a goal location, and a set of possible initial locations.
    """

    def __init__(self, goal, initial):
        """
        Initializes the task with a goal region and a set of starting locations.

        :param goal: the goal region
        :param initial: a region, or list of regions, from which the agent may begin
        """

        self._goal = goal
        self._initial = initial if isinstance(initial, Iterable) else [initial]

    def complete(self, x, y):
        """
        Determines if the agent has reached the goal.

        :param x: the agent's x coordinate
        :param y: the agent's y coordinate
        :return: True if the agent is within the goal region, false otherwise
        """

        return self._goal.contains(x, y)

    def sample(self):
        """
        Randomly samples an initial state for this task

        :return: the x and y coordinates of the cell
        """

        return np.random.choice(self._initial).sample()

    @property
    def goal(self):
        return self._goal

    @property
    def initial(self):
        return self._initial


class Map:
    """
    Represents an editable occupancy map, used to build
    the different maps without making the environment class
    itself mutable.
    """

    def __init__(self, width, height):
        """
        Initializes an empty map with the given dimensions.

        :param width: the width of the map
        :param height: the height of the map
        """

        self._width = width
        self._height = height

        self._occupied = np.zeros(shape=(width, height), dtype=np.bool_)

    def obstacle(self, x, y, width, height):
        """
        Adds a rectangular obstacle to the map.

        :param x: the x coordinate of the top-left corner of the obstacle
        :param y: the y coordinate of the top-left corner of the obstacle
        :param width: the width of the obstacle
        :param height: the height of the obstacle
        """

        for x_pos in range(x, x + width):
            for y_pos in range(y, y + height):
                self._occupied[x_pos, y_pos] = True

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def occupied(self):
        return self._occupied


class Environment:
    """
    Represents a simulation of a 2D navigation task. The environment is
    represented by the occupancy map, while we have a collection of named
    goal regions and starting locations.  A reward function based on the
    current goal location is also defined.
    """

    def __init__(self, map, tasks):
        """
        Initializes the environment with an occupancy map and a set of tasks

        :param map: the occupancy map defining the environment
        :param tasks: a dictionary of tasks
        """

        # Capture configuration
        self._width = map.width
        self._height = map.height
        self._occupied = map.occupied
        self._tasks = tasks

        # Initialize agent position
        self._x = 0
        self._y = 0

        # Initialize the current task
        self._task = None

    def reset(self, task=None):
        """
        Resets the agent's position. May also change the current task.

        :param task: the name of the current task, if None then the task is unchanged
        """

        # Change the task if necessary
        if task is not None:
            self._task = self._tasks[task]

        # Get a random initial position based on the task
        while True:
            self._x, self._y = self._task.sample()

            if 0 <= self._x < self._width and 0 <= self._y < self._height and not self._occupied[self._x, self._y]:
                break

    def update(self, action):
        """
        Updates the environment based on the action provided.

        :param action: the action taken, a value of the Action enum
        """

        # Update position
        x = self._x
        y = self._y

        if Action.UP == action:
            y += 1
        elif Action.DOWN == action:
            y -= 1
        elif Action.LEFT == action:
            x -= 1
        elif Action.RIGHT == action:
            x += 1

        if (0 <= x < self._width) and (0 <= y < self._height) and not self._occupied[x, y]:
            self._x = x
            self._y = y

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def occupied(self):
        return self._occupied

    @property
    def tasks(self):
        return self._tasks.items()

    @property
    def task(self):
        return self._task

    @property
    def complete(self):
        return self._task.complete(self._x, self._y)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y
