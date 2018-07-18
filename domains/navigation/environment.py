import numpy as np
from enum import IntEnum
from .sensor import SensorState


class Action(IntEnum):
    """
    An enum describing the set of possible actions.
    """

    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


class Task:
    """
    Represents a single task as a rectangular region.
    """

    def __init__(self, x, y, width, height):
        """
        Initializes the task.

        :param x: the x coordinate of the top left corner of the goal region
        :param y: the y coordinate of the top left corner of the goal region
        :param width: the width of the goal region
        :param height: the height of the goal region
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def goal(self, x, y):
        """
        Tests whether a given location is within the goal region.

        :param x: the x coordinate of the location
        :param y: the y coordinate of the location
        :return: True if the location is within the goal, False otherwise
        """

        dx = x - self.x
        dy = y - self.y

        return (0 <= dx < self.width) and (0 <= dy < self.height)


class Environment:
    """
    Represents a simulation of a 2D navigation task.

    The environment is represented by the occupancy map,
    provided, and tasks are defined by rectangular goal
    regions.  At any given time the agent has a location,
    but it also has a map of the environment that indicates
    whether a particular location has been seen, and whether
    it is occupied.  This map is updated as the agent
    explores the environment.
    """

    def __init__(self, width, height, sensor):
        """
        Initializes the environment.

        :param width: the width of the occupancy map
        :param height: the height of the occupancy map
        :param sensor: the sensor model
        """

        # Initialize agent position
        self.x = 0
        self.y = 0

        # Capture width and height
        self.width = width
        self.height = height

        # Initialize occupancy map
        self.occupancy = np.zeros(shape=(width, height), dtype=np.bool_)

        # Initialize sensor map
        self.map = np.empty(shape=(width, height), dtype=np.int_)
        self.map.fill(SensorState.UNKNOWN)

        # Initialize task dictionary
        self._tasks = {}
        self._task = None

        # Capture sensor model
        self._sensor = sensor

    def add_obstacle(self, x, y, width, height):
        """
        Adds a rectangular obstacle to the map.

        :param x: the x coordinate of the top-left corner of the obstacle
        :param y: the y coordinate of the top-left corner of the obstacle
        :param width: the width of the obstacle
        :param height: the height of the obstacle
        """

        for x_pos in range(x, x + width):
            for y_pos in range(y, y + height):
                self.occupancy[x_pos, y_pos] = True

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
        Resets the current state of the environment.

        Note that this also resets the sensor map.  It is
        the agent's job to accumulate sensor maps over
        multiple episodes.  Uses the current task to get
        the starting position for the agent, it is up to the
        task to ensure the position is valid.
        """

        # Get a random initial position, up to the task to make sure it is valid
        self.x, self.y = self._task.reset()

        # Clear the sensor map
        self.map = np.empty(shape=(self.width, self.height), dtype=np.int_)
        self.map.fill(SensorState.UNKNOWN)

    def update(self, action):
        """
        Updates the environment based on the action provided.

        :param action: the action taken, a value of the Action enum
        """

        # Update position
        x = self.x
        y = self.y

        if Action.UP == action:
            y = self.y + 1
        elif Action.DOWN == action:
            y = self.y - 1
        elif Action.LEFT == action:
            x = self.x - 1
        elif Action.RIGHT == action:
            x = self.x + 1

        if (0 <= x < self.width) and (0 <= y < self.height) and not self.occupancy[x, y]:
            self.x = x
            self.y = y

        # Update occupancy map
        self._sensor.update(self.x, self.y, self.occupancy, self.map)

    def expert(self):
        """
        Gets the action taken by the expert policy for the current task.

        :return: the expert action
        """

        return Action.NONE

    @property
    def goal_x(self):
        return self._task.x

    @property
    def goal_y(self):
        return self._task.y

    @property
    def goal_width(self):
        return self._task.width

    @property
    def goal_height(self):
        return self._task.height
