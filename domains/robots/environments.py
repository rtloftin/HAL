"""
Defines a collection of environments from the roboschool package.

We need to rethink the environment-agent interface we are using
"""

from .RoboschoolAnt_v0 import Policy as AntPolicy
from .RoboschoolHopper_v0 import Policy as HopperPolicy
from .RoboschoolReacher_v0 import Policy as ReacherPolicy
from .RoboschoolInvertedPendulumSwingup_v0 import Policy as PendulumPolicy
from ..space import Continuous

import gym


class Environment:
    """
    A wrapper for the roboschool environments.
    """

    def __init__(self, env, expert):
        """
        Initializes the wrapper object.

        :param env: the underlying Gym environment
        :param expert: the expert policy for this environment
        """

        self._env = env
        self._expert = expert

        self._state_space = Continuous(env.observation_space.shape,
                                       low=env.observation_space.low, high=env.observation_space.high)
        self._action_space = Continuous(env.action_space.shape, low=env.action_space.low, high=env.action_space.high)

        self._state = self._env.reset()
        self._reward = 0
        self._complete = False

    def reset(self):
        """
        Starts a new episode.
        """

        self._state = self._env.reset()
        self._reward = 0
        self._complete = False

    def update(self, action):
        """
        Takes a single step in the environment.

        :param action: the action to take at this step
        """
        if not self._complete:
            self._state, self._reward, self._complete, _ = self._env.step(action)

    def expert(self):
        """
        Gets an action from the expert policy for the current state.

        :return: the action taken by the expert's policy
        """

        return self._expert(self._state)

    @property
    def state_space(self):
        return self._state_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def state(self):
        return self._state

    @property
    def reward(self):
        return self._reward

    @property
    def complete(self):
        return self._complete


def ant():
    """
    Gets an instance of the Ant environment.

    :return: an Ant environment
    """

    return Environment(gym.make("RoboschoolAnt-v1"), AntPolicy())


def hopper():
    """
    Gets an instance of the Hopper environment.

    :return: an Hopper environment
    """

    return Environment(gym.make("RoboschoolHopper-v1"), HoperPolicy())


def reacher():
    """
    Gets an instance of the Reacher environment.

    :return: an Reacher environment
    """

    return Environment(gym.make("RoboschoolReacher-v1"), ReacherPolicy())


def pendulum():
    """
    Gets an instance of the Pendulum environment.

    :return: an Pendulum environment
    """

    return Environment(gym.make("RoboschoolInvertedPendulumSwingup-v1"), PendulumPolicy())
