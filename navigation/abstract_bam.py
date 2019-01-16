"""
An implementation of BAM for the robot navigation domain, which uses
an abstract, high-level representation of the dynamics in unmapped
areas of the environment.
"""

from .environment import Action
from .sensor import Occupancy

import tensorflow as tf
import numpy as np
import time


class Agent:
    """
    A BAM agent which uses an abstract planning model.
    """

    def __init__(self, env, **kwargs):
        """
        Constructs a new abstract BAM agent.

        :param env: the navigation environment this agent must operate in
        :param kwargs: the configuration parameters for the agent.
        """

        # Capture environment
        self._env = env

        # Capture environment height
        self._height = env.height

        # Unpack configuration
        model_fn = kwargs['model_fn']
        beta = kwargs['beta']
        learning_rate = kwargs['learning_rate']
        rms_prop = kwargs['rms_prop']

        self._pretrain_batches = kwargs['pretrain_batches']
        self._batch_size = kwargs['batch_size']
        self._online_batches = kwargs['online_batches']

        # Initialize placeholders
        self._session = None
        self._sensor = None
        self._data = None

        # Construct computation graph
        self._graph = tf.Graph()

        with self._graph.as_default():

            # Construct dynamics model
            self._model = model_fn()

            # Define state and action inputs
            self._state_input = tf.placeholder(tf.int32, shape=[self._batch_size])
            self._action_input = tf.placeholder(tf.int32, shape=[self._batch_size])
            self._policy_input = tf.placeholder(tf.int32, shape=[1])

            # Build individual task models
            self._reward_functions = dict()
            self._reward_updates = dict()
            self._policies = dict()
            self._policy = None

            for task, _ in env.tasks:

                # Define the value and reward functions
                values, rewards, penalty = self._model.task()
                self._reward_functions[task] = rewards

                # Define state values
                batch_values = tf.gather(values, self._state_input)
                mean = tf.expand_dims(tf.reduce_mean(batch_values, axis=1), axis=1)
                variance = 0.001 + tf.expand_dims(tf.reduce_mean(tf.square(batch_values - mean), axis=1), axis=1)
                normalized = beta * ((batch_values - mean) / tf.sqrt(variance))

                # Define the action prediction loss
                partition = tf.log(tf.reduce_sum(tf.exp(normalized), axis=1))
                likelihood = tf.reduce_sum(tf.one_hot(self._action_input, len(Action)) * normalized, axis=1)

                loss = tf.reduce_mean(partition - likelihood) + penalty + self._model.penalty

                if rms_prop:
                    self._reward_updates[task] = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)
                else:
                    self._reward_updates[task] = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

                # Define the action output
                self._policies[task] = tf.argmax(values, axis=1)

    def session(self, sensor, data):
        """
        Re-initializes the learning agent with new training data
        and a new sensor map, and constructs a new Tensorflow
        session, which is returned to be used in a with block.

        :param sensor: the sensor model, with associated training data
        :param data: the demonstration data the agent needs to learn from
        :return: a context manager to clean up any resources used by the agent
        """

        # Initialize Tensorflow session
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_options)
        self._session = tf.Session(graph=self._graph, config=config)

        # Capture sensor and training data
        self._sensor = sensor
        self._data = data

        # Initialize the agent
        try:
            with self._graph.as_default():
                self._session.run(tf.global_variables_initializer())

            self._session.run(self._model.sensor_update, feed_dict={
                self._model.sensor_input: sensor.map
            })

            for task in data.tasks:
                update = self._reward_updates[task]
                samples = data.steps(task)

                for b in range(self._pretrain_batches):
                    batch = np.random.choice(samples, self._batch_size)
                    states = []
                    actions = []

                    for step in batch:
                        states.append(((step.x * self._height) + step.y))
                        actions.append(step.action)

                    self._session.run(update, feed_dict={
                        self._state_input: states,
                        self._action_input: actions
                    })

        except Exception as e:
            self._session.close()
            raise e

        return self._session

    def update(self):
        """
        Updates the agent's cost estimates to reflect new sensor data.
        """

        self._session.run(self._model.sensor_update, feed_dict={
            self._model.sensor_input: self._sensor.map
        })

        for task in self._data.tasks:
            update = self._reward_updates[task]
            samples = self._data.steps(task)

            for _ in range(self._online_batches):
                batch = np.random.choice(samples, self._batch_size)
                states = []
                actions = []

                for step in batch:
                    states.append((step.x * self._height) + step.y)
                    actions.append(step.action)

                self._session.run(update, feed_dict={
                    self._state_input: states,
                    self._action_input: actions
                })

    def task(self, name):
        """
        Sets the task the agent is currently performing.

        :param name: the name of the task
        """

        self._session.run(self._model.sensor_update, feed_dict={
            self._model.sensor_input: self._sensor.map
        })

        self._policy = self._session.run(self._policies[name])

    def act(self, x, y):
        """
        Samples an action from the agent's policy for the current state

        :param x: the agent's x coordinate
        :param y: the agent's y coordinate
        :return: the sampled action
        """

        return self._policy[(x * self._height) + y]

    def rewards(self, task):
        """
        Gets the reward function for a given task.

        :param task: the name of the task
        :return: a 2D np.array containing the rewards, the minimum value, the maximum value
        """

        rewards = self._session.run(self._reward_functions[task])
        reward = np.empty((self._env.width, self._env.height), dtype=np.float32)

        for x in range(self._env.width):
            for y in range(self._env.height):
                reward[x, y] = rewards[(x * self._height) + y]

        return reward


def builder(model_fn,
            beta=1.0,
            learning_rate=0.001,
            batch_size=128,
            pretrain_batches=100,
            online_batches=50,
            rms_prop=False):
    """
    Returns a factory method for constructing abstract BAM agents with the given configuration.

    :param model_fn: the function used to construct the abstract model
    :param beta: the action selection temperature
    :param learning_rate: the learning rate for training the cost functions
    :param batch_size: the batch size for training the cost functions
    :param pretrain_batches: the number of batch updates to perform to build the initial cost estimates
    :param online_batches: the number of batch updates to perform after each model update
    :param rms_prop: whether to use RMSProp updates instead of the default Adam updates
    :return: a new builder for BAM agents
    """

    def build(env):
        return Agent(env,
                     model_fn=model_fn,
                     beta=beta,
                     learning_rate=learning_rate,
                     batch_size=batch_size,
                     pretrain_batches=pretrain_batches,
                     online_batches=online_batches,
                     rms_prop=rms_prop)

    return build
