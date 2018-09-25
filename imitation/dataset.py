"""
Defines a class used to store sets of demonstrations of multiple tasks.
"""


class Step:
    """
    A single state-action pair.
    """

    def __init__(self, state, action):
        """
        Constructs the state-action pair.

        :param state: the current state
        :param action: the action taken
        """

        self._state = state
        self._action = action

    @property
    def state(self):
        return self._state

    @property
    def action(self):
        return self._action


class Dataset:
    """
    A collection of demonstrations of multiple tasks, along
    with utility methods for manipulating them.
    """

    def __init__(self):
        """
        Initializes the data set.
        """

        self._tasks = dict()
        self._current = None

    def new(self, task):
        """
        Starts a new demonstration, all subsequent actions
        will be part of this demonstrated trajectory.

        :param task: the name of the task being demonstrated
        """

        if task not in self._tasks:
            self._tasks[task] = []

        self._current = []
        self._tasks[task].append(self._current)

    def step(self, state, action):
        """
        Demonstrates a single state-action pair.

        :param state: the current state
        :param action: the action taken
        """

        if self._current is not None:
            self._current.append(Step(state, action))

    def tasks(self):
        """
        Gets a list of the names of all of the tasks contained in this dataset.

        :return: a list of tasks names
        """

        return self._tasks.keys()

    def trajectories(self, task):
        """
        Gets a list of all the demonstrated trajectories for a task.

        :param task: the name of the task
        :return: a list of trajectories, that is, a list of lists
        """

        return self._tasks[task]

    def steps(self, task):
        """
        Gets a list of all the state action pairs demonstrated for a task.

        :param task: the name of the task
        :return: a list of state-action pairs
        """

        steps = []

        for trajectory in self._tasks[task]:
            steps.extend(trajectory)

        return steps
