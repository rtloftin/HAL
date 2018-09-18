"""
Defines a collection of environments from the roboschool package.

We need to rethink the environment-agent interface we are using
"""

from .RoboschoolAnt_v0 import Policy as AntPolicy
from .RoboschoolHopper_v0 import Policy as HopperPolicy
from .RoboschoolReacher_v0 import Policy as ReacherPolicy
from .RoboschoolInvertedPendulumSwingup_v0 import Policy as PendulumPolicy
import gym


class Ant:

    def __init__(self):
        self._env = gym.make("RoboschoolAnt-v1")
        self.state_size = self._env.observation_space.shape[0]
        self.action_size = self._env.action_space.shape[0]
        self.action_low = self._env.action_space.low
        self.action_high = self._env.action_space.high
        self.discrete_action = False

        self._expert = AntPolicy(self.state_size, self.action_size)

        self.state = None
        self.reward = 0
        self.complete = False

    def reset(self):
        self.state = self._env.reset()
        self.reward = 0
        self.complete = False

    def update(self, action):
        if not self.complete:
            self.state, self.reward, self.complete, _ = self._env.step(action)

    def expert(self):
        return self._expert(self.state)


class Hopper:

    def __init__(self):
        self._env = gym.make("RoboschoolHopper-v1")
        self.state_size = self._env.observation_space.shape[0]
        self.action_size = self._env.action_space.shape[0]
        self.action_low = self._env.action_space.low
        self.action_high = self._env.action_space.high
        self.discrete_action = False

        self._expert = HopperPolicy(self.state_size, self.action_size)

        self.state = None
        self.reward = 0
        self.complete = False

    def reset(self):
        self.state = self._env.reset()
        self.reward = 0
        self.complete = False

    def update(self, action):
        if not self.complete:
            self.state, self.reward, self.complete, _ = self._env.step(action)

    def expert(self):
        return self._expert(self.state)


class Reacher:

    def __init__(self):
        self._env = gym.make("RoboschoolReacher-v1")
        self.state_size = self._env.observation_space.shape[0]
        self.action_size = self._env.action_space.shape[0]
        self.action_low = self._env.action_space.low
        self.action_high = self._env.action_space.high
        self.discrete_action = False

        self._expert = ReacherPolicy(self.state_size, self.action_size)

        self.state = None
        self.reward = 0
        self.complete = False

    def reset(self):
        self.state = self._env.reset()
        self.reward = 0
        self.complete = False

    def update(self, action):
        if not self.complete:
            self.state, self.reward, self.complete, _ = self._env.step(action)

    def expert(self):
        return self._expert(self.state)


class Pendulum:
    def __init__(self):
        self._env = gym.make("RoboschoolInvertedPendulumSwingup-v1")
        self.state_size = self._env.observation_space.shape[0]
        self.action_size = self._env.action_space.shape[0]
        self.action_low = self._env.action_space.low
        self.action_high = self._env.action_space.high
        self.discrete_action = False

        self._expert = PendulumPolicy(self.state_size, self.action_size)

        self.state = None
        self.reward = 0
        self.complete = False

    def reset(self):
        self.state = self._env.reset()
        self.reward = 0
        self.complete = False

    def update(self, action):
        if not self.complete:
            self.state, self.reward, self.complete, _ = self._env.step(action)

    def expert(self):
        return self._expert(self.state)
