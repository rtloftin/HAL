"""
Defines a class used to store expert trajectories which
some of the teachers will use to learn their feedback models.

WE MAY WANT TO MERGE THIS WITH THE SIMILAR CLASS USED FOR LEARNING FROM DEMONSTRATION
"""


class Step:
    """
    A single state-action pair, with immediate reward.
    """

    def __init__(self, state, action, reward):
        """
        Constructs the state-action pair.

        :param state: the current state
        :param action: the action taken
        :param reward: the immediate reward received
        """

        self._state = state
        self._action = action
        self._reward = reward

    @property
    def state(self):
        return self._state

    @property
    def action(self):
        return self._action


class Demonstrations:
    """
    A collection of demonstrations of multiple tasks.  Includes associated reward values.
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

    def step(self, state, action, reward):
        """
        Demonstrates a single state-action pair.

        :param state: the current state
        :param action: the action taken
        :param reward: the immediate reward received
        """

        if self._current is not None:
            self._current.append(Step(state, action, reward))

    def trajectories(self, task):
        """
        Gets a list of all the demonstrated trajectories for a task.

        :param task: the name of the task
        :return: a list of trajectories, that is, a list of lists
        """

        return self._tasks[task]

    @property
    def tasks(self):
        return self._tasks.keys()


def generate(env, episodes=1000, steps=500):
    """
    Generates a set of expert demonstrations of each task.

    :param env: the environment in which to generate the demonstrations
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
                env.update(action)
                data.step(state, action, env.reward)
                step += 1

    return data
