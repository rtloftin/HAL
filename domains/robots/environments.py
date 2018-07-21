from .RoboschoolAnt_v0 import Policy as AntPolicy
from .RoboschoolHopper_v0 import Policy as HopperPolicy
from .RoboschoolReacher_v0 import Policy as ReacherPolicy
from .RoboschoolInvertedPendulumSwingup_v0 import Policy as PendulumPolicy
import gym


class Ant:

    def __init__(self):
        self._env = gym.make("RoboschoolAnt-v1")
        self._expert = AntPolicy(self._env.observation_space, self._env.action_space)

        self.state = None
        self.reward = 0
        self.complete = False

    def reset(self):
        self.state = self._env.reset()
        self.reward = 0
        self.complete = False

    def update(self, action):
        self.state, self.reward, self.complete, _ = self._env.update(action)

    def expert(self):
        return self._expert(self.state)


class Hopper:
    def __init__(self):
        self._env = gym.make("RoboschoolHopper-v1")
        self._expert = HopperPolicy(self._env.observation_space, self._env.action_space)

        self.state = None
        self.reward = 0
        self.complete = False

    def reset(self):
        self.state = self._env.reset()
        self.reward = 0
        self.complete = False

    def update(self, action):
        self.state, self.reward, self.complete, _ = self._env.update(action)

    def expert(self):
        return self._expert(self.state)


class Reacher:
    def __init__(self):
        self._env = gym.make("RoboschoolReacher-v1")
        self._expert = ReacherPolicy(self._env.observation_space, self._env.action_space)

        self.state = None
        self.reward = 0
        self.complete = False

    def reset(self):
        self.state = self._env.reset()
        self.reward = 0
        self.complete = False

    def update(self, action):
        self.state, self.reward, self.complete, _ = self._env.update(action)

    def expert(self):
        return self._expert(self.state)


class Pendulum:
    def __init__(self):
        self._env = gym.make("RoboschoolPendulum-v1")
        self._expert = PendulumPolicy(self._env.observation_space, self._env.action_space)

        self.state = None
        self.reward = 0
        self.complete = False

    def reset(self):
        self.state = self._env.reset()
        self.reward = 0
        self.complete = False

    def update(self, action):
        self.state, self.reward, self.complete, _ = self._env.update(action)

    def expert(self):
        return self._expert(self.state)
