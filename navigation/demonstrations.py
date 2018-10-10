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

    def step(self, x, y, action):
        """
        Demonstrates a single state-action pair.

        :param x: the agent's x coordinate
        :param y: the agent's y coordinate
        :param action: the action taken
        """

        if self._current is not None:
            self._current.append(Step(x, y, action))

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

    @property
    def tasks(self):
        return self._tasks.keys()


# Move this somewhere else -- probably to an experiment class
def generate(env, expert, episodes=1000, steps=500):
    """
    Generates a set of expert demonstrations of each task.

    :param env: the environment in which to generate the demonstrations
    :
    :param episodes: the number of demonstrated episodes for each task
    :param steps: the maximum number of steps per-episode
    :return: a Demonstrations object containing the demonstrations
    """

    data = Demonstrations()

    for task in environment.tasks:
        for demo in range(episodes):
            print("Task: " + task + ", demonstration " + str(demo))

            env.reset(task=task)
            data.new(task)
            step = 0

            while not env.complete and (step < steps):
                state = env.state()
                action = env.expert()
                data.step(state, action)
                env.update(action)
                step += 1

    return data
