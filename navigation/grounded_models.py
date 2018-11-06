"""
A collection of abstract planning models, which allow
for the direct integration of sensor visibility data.
"""

from .environment import Action
from .sensor import Occupancy

import tensorflow as tf
import numpy as np


class AbstractGrid:
    """
    A planning model which represents the dynamics
    of unmapped states with a coarse discretization
    of the map, with unknown connectivity.
    """

    def __init__(self, **kwargs):
        """
        Initializes the abstract grid model without any task models defined.

        :param kwargs: configuration parameters
        """

        # Unpack parameters
        width = kwargs['width']
        height = kwargs['height']
        h_step = kwargs['h_step']
        v_step = kwargs['v_step']
        link_mean = kwargs['link_mean']
        link_penalty = kwargs['link_penalty']

        self._base_gamma = kwargs['gamma']
        self._abstract_gamma = self._base_gamma ** ((h_step + v_step) // 2)

        self._beta = kwargs['beta']
        self._planning_depth = kwargs['planning_depth']
        self._reward_penalty = kwargs['reward_penalty']

        # Compute model dimensions
        base_states = width * height
        base_actions = len(Action)

        abstract_width = width // h_step
        abstract_height = height // v_step
        abstract_states = abstract_width * abstract_height
        abstract_actions = 4
        sub_states = h_step * v_step

        self._base_states = base_states
        self._abstract_states = abstract_states

        # Define transitions from base states
        base_next = np.empty([base_states, base_actions], dtype=np.int32)
        base_identity = np.empty([base_states, base_actions], dtype=np.int32)
        base_abstract = np.empty([base_states], dtype=np.int32)

        def base(cell, nx, ny, action):
            base_identity[cell, action] = cell

            if 0 <= nx < width and 0 <= ny < height:
                base_next[cell, action] = (nx * height) + ny
            else:
                base_next[cell, action] = cell

        for x in range(width):
            for y in range(height):
                cell = (x * height) + y
                base_abstract[cell] = ((x // h_step) * abstract_height) + (y // v_step)

                base(cell, x, y, Action.STAY)
                base(cell, x, y + 1, Action.UP)
                base(cell, x, y - 1, Action.DOWN)
                base(cell, x - 1, y, Action.LEFT)
                base(cell, x + 1, y, Action.RIGHT)

        base_next = tf.constant(base_next, dtype=tf.int32)
        base_identity = tf.constant(base_identity, dtype=tf.int32)

        self._base_abstract = tf.constant(base_abstract, dtype=tf.int32)

        # Define transitions from abstract states
        abstract_next = np.empty([abstract_states, abstract_actions], dtype=np.int32)
        abstract_base = np.empty([abstract_states, sub_states], dtype=np.int32)

        def abstract(cell, nx, ny):
            if 0 <= nx < abstract_width and 0 <= ny < abstract_height:
                return (nx * abstract_height) + ny
            return cell

        for x in range(abstract_width):
            for y in range(abstract_height):
                cell = (x * abstract_height) + y

                # Abstract actions
                abstract_next[cell, 0] = abstract(cell, x, y + 1)
                abstract_next[cell, 1] = abstract(cell, x, y - 1)
                abstract_next[cell, 2] = abstract(cell, x - 1, y)
                abstract_next[cell, 3] = abstract(cell, x + 1, y)

                # Base actions
                start_x = x * h_step
                start_y = y * v_step

                for x_offset in range(0, h_step):
                    for y_offset in range(0, v_step):
                        abstract_base[cell, (x_offset * v_step) + y_offset] = \
                            ((start_x + x_offset) * height) + start_y + y

        self._abstract_transitions = tf.constant(abstract_next, dtype=tf.int32)
        self._abstract_base = tf.constant(abstract_base, dtype=tf.int32)

        # Define abstract dynamics model
        model = tf.Variable(tf.fill([abstract_states, abstract_actions], link_mean), dtype=tf.float32)

        self._model = tf.nn.sigmoid(model)
        self._model_penalty = link_penalty * tf.reduce_mean(tf.square(link_mean - model))

        # Define transition update from sensor data
        self._sensor_input = tf.placeholder(tf.int32, shape=[width, height])
        sensor = tf.reshape(self._sensor_input, [base_states])

        visible = tf.not_equal(sensor, Occupancy.UNKNOWN)
        mask = tf.where(tf.equal(sensor, Occupancy.CLEAR), tf.ones([base_states], dtype=tf.float32),
                        tf.zeros([base_states], dtype=tf.float32))
        base = tf.where(tf.gather(tf.equal(sensor, Occupancy.OCCUPIED), base_next), base_identity, base_next)

        self._visible = tf.Variable(tf.zeros([base_states], dtype=tf.bool), trainable=False, use_resource=True)
        self._mask = tf.Variable(tf.zeros([base_states], dtype=tf.float32), trainable=False, use_resource=True)
        self._base_transitions = tf.Variable(tf.zeros([base_states, base_actions], dtype=tf.int32),
                                             trainable=False, use_resource=True)

        self._sensor_update = tf.group(tf.assign(self._visible, visible), tf.assign(self._mask, mask),
                                       tf.assign(self._base_transitions, base))

    def task(self):
        """
        Constructs a new Q-function for a new task, associated
        with a new intent vector variable.

        TODO: Figure out how to make this thing work

        :return: a tensor representing the Q-values for this task, a task intent Tensor, a scalar reward penalty
        """

        # Define the reward function
        rewards = tf.Variable(tf.zeros([self._base_states], dtype=tf.float32))

        # Define the reward penalty
        penalty = self._reward_penalty * tf.reduce_mean(tf.square(rewards))

        # Define value function
        vb = tf.zeros([self._base_states], dtype=tf.float32)
        va = tf.zeros([self._abstract_states], dtype=tf.float32)

        for _ in range(self._planning_depth):

            # Combine value functions
            v = tf.gather(va, self._base_abstract)
            v = tf.where(self._visible, vb, v)

            # Compute the base Q-function
            qb = self._base_gamma * tf.gather(v, self._base_transitions)

            # Compute the abstract Q-function
            qaa = self._abstract_gamma * self._model * tf.gather(va, self._abstract_transitions)
            qab = self._abstract_gamma * tf.gather(self._mask * vb, self._abstract_base)

            # Compute the base value function
            policy = tf.exp(self._beta * qb)
            vb = rewards + tf.reduce_sum(policy * qb) / tf.reduce_sum(policy, axis=1)

            # Compute the abstract value function
            abstract_policy = tf.exp(self._beta * qaa)
            base_policy = tf.exp(self._beta * qab)
            normal = tf.reduce_sum(abstract_policy, axis=1) + tf.reduce_sum(base_policy, axis=1)
            va = (tf.reduce_sum(abstract_policy * qaa, axis=1) + tf.reduce_sum(base_policy * qab, axis=1)) / normal

        # Compute output Q-values
        v = tf.gather(va, self._base_abstract)
        v = tf.where(self._visible, vb, v)
        q_values = self._base_gamma * tf.gather(v, self._base_transitions)

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


def abstract_grid(width, height,
                  h_step=5,
                  v_step=5,
                  planning_depth=200,
                  gamma=0.99,
                  beta=1.0,
                  link_mean=1.,
                  link_penalty=10.0,
                  reward_penalty=100.):
    """
    Returns a builder which constructs abstract grid
    objects attached to a given graph and sensor model.

    :param width: the width of the sensor map
    :param height: the height of the sensor map
    :param h_step: the width of the abstract states
    :param v_step: the height of the abstract states
    :param planning_depth: the number of value iterations to perform
    :param gamma: the one-step discount factor
    :param beta: the action selection temperature for planning
    :param link_mean: the mean of the log-probabilities of a successful connection
    :param link_penalty: the inverse variance of the log-probabilities of a successful connection
    :param reward_penalty: the inverse variance of the reward values
    :return: a builder method for abstract grid models
    """

    return lambda: AbstractGrid(width=width,
                                height=height,
                                h_step=h_step,
                                v_step=v_step,
                                planning_depth=planning_depth,
                                gamma=gamma,
                                beta=beta,
                                link_mean=link_mean,
                                link_penalty=link_penalty,
                                reward_penalty=reward_penalty)
