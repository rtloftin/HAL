from .environment import Action
from .sensor import Occupancy

import tensorflow as tf
import numpy as np


class DummyGrid:
    """
    A planning model which represents the dynamics
    of unmapped states with a coarse discretization
    of the map, with unknown connectivity.
    """

    def __init__(self, width, height, **kwargs):
        """
        Initializes the abstract grid model without any task models defined.

        :param env: the navigation environment
        :param kwargs: configuration parameters
        """

        # Capture parameters
        self._gamma = kwargs['gamma']
        self._beta = kwargs['beta']
        self._planning_depth = kwargs['planning_depth']
        self._reward_penalty = kwargs['reward_penalty']

        # Compute model dimensions
        states = width * height
        actions = len(Action)

        self._states = states

        # Define transitions from base states
        next = np.empty([states, actions], dtype=np.int32)
        identity = np.empty([states, actions], dtype=np.int32)

        def transition(cell, nx, ny, action):
            identity[cell, action] = cell

            if 0 <= nx < width and 0 <= ny < height:
                next[cell, action] = (nx * height) + ny
            else:
                next[cell, action] = cell

        for x in range(width):
            for y in range(height):
                cell = (x * height) + y

                transition(cell, x, y, Action.STAY)
                transition(cell, x, y + 1, Action.UP)
                transition(cell, x, y - 1, Action.DOWN)
                transition(cell, x - 1, y, Action.LEFT)
                transition(cell, x + 1, y, Action.RIGHT)

        next = tf.constant(base_next, dtype=tf.int32)
        identity = tf.constant(base_identity, dtype=tf.int32)

        # Define transition update from sensor data
        self._sensor_input = tf.placeholder(tf.int32, shape=[width, height])

        sensor = tf.reshape(self._sensor_input, [base_states])
        transitions = tf.where(tf.gather(tf.equal(sensor, Occupancy.OCCUPIED), next), identity, next)

        self._transitions = tf.Variable(tf.zeros([states, actions], dtype=tf.int32), trainable=False, use_resource=True)
        self._sensor_update = tf.assign(self._transitions, transitions)

        # Define dummy model penalty
        self._model_penalty = tf.constant(0., dtype=tf.float32)

    def task(self):
        """
        Constructs a new Q-function for a new task, associated
        with a new intent vector variable.

        TODO: Figure out how to make this thing work

        :return: a tensor representing the Q-values for this task, a task intent Tensor, a scalar reward penalty
        """

        # Define the reward function
        rewards = tf.Variable(tf.zeros([self._states], dtype=tf.float32))

        # Define the reward penalty
        penalty = self._reward_penalty * tf.reduce_mean(tf.square(rewards))

        # Define value function
        v = tf.zeros([self._states], dtype=tf.float32)

        for _ in range(self._planning_depth):

            # Compute the Q-function
            q = self._gamma * tf.gather(v, self._base_transitions)

            # Compute the base value function
            policy = tf.exp(self._beta * q)
            v = rewards + tf.reduce_sum(policy * q, axis=1) / tf.reduce_sum(policy, axis=1)

        # Compute output Q-values
        q_values = self._gamma * tf.gather(v, self._transitions)

        return q_values, rewards, penalty

    @property
    def sensor_input(self):
        return self._sensor_input

    @property
    def sensor_update(self):
        return self._sensor_update

    @property
    def penalty(self):
        return self._model_penalty


def dummy_grid(width, height,
               planning_depth=200,
               gamma=0.99,
               beta=1.0,
               reward_penalty=100.):

    return lambda: DummyGrid(width, height,
                             planning_depth=planning_depth,
                             gamma=gamma,
                             beta=beta,
                             reward_penalty=reward_penalty)
