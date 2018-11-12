"""
Defines a class used to store sets of demonstrations of multiple tasks.
"""


class Step:
    """
    A single state-action pair.
    """

    def __init__(self, x, y, action):
        """
        Constructs the state-action pair.

        :param x: the agent's x coordinate
        :param y: the agent's y coordinate
        :param action: the action taken
        """

        self._x = x
        self._y = y
        self._action = action

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def action(self):
        return self._action


class Demonstrations:
    """
    A collection of demonstrations of multiple tasks, along
    with utility methods for manipulating them.
    """

    def __init__(self):
        """
        Initializes the data set.
        """

        self._tasks = dict()
        self._trajectories = dict()

        self._current_task = None
        self._current_trajectory = None

    def new(self, task):
        """
        Starts a new demonstration, all subsequent actions
        will be part of this demonstrated trajectory.

        :param task: the name of the task being demonstrated
        """

        if task not in self._tasks:
            self._tasks[task] = []
            self._trajectories[task] = []

        self._current_task = self._tasks[task]

        self._current_trajectory = []
        self._trajectories[task].append(self._current_trajectory)

    def step(self, x, y, action):
        """
        Demonstrates a single state-action pair.

        :param x: the agent's x coordinate
        :param y: the agent's y coordinate
        :param action: the action taken
        """

        if self._current_task is not None:
            step = Step(x, y, action)

            self._current_task.append(step)
            self._current_trajectory.append(step)

    def trajectories(self, task):
        """
        Gets a list of all the state-action trajectories for a task.

        :param task: the name of the task
        :return: a list of trajectories
        """

        return self._trajectories[task]

    def steps(self, task):
        """
        Gets a list of all the state-action pairs demonstrated for a task.

        :param task: the name of the task
        :return: a list of state-action pairs
        """

        return self._tasks[task]

    @property
    def tasks(self):
        return self._tasks.keys()
