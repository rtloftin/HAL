"""
An implementation of HAL for the robot navigation domain.

TODO: Figure out how this should actually work
"""

from .environment import Action
from .sensor import Occupancy

import tensorflow as tf
import numpy as np
import time


class Agent:
    """
    A BAM agent.
    """

    def __init__(self, graph, session, sensor, data, **kwargs):
        """
        Constructs the agent and initializes the cost function estimates.

        :param graph: the TensorFlow graph for the agent to use
        :param session: the TensorFlow session for the agent to use
        :param sensor: the sensor model used to observe the environment
        :param data: the demonstrated state-action trajectories
        :param kwargs: the configuration parameters for the agent.
        """

        # Capture instance objects
        self._sensor = sensor
        self._data = data
        self._session = session

        # Unpack configuration
        model_fn = kwargs['model_fn']
        beta = kwargs['beta']
        bellman_weight = kwargs['bellman_weight']
        learning_rate = kwargs['learning_rate']
        pretrain_batches = kwargs['pretrain_batches']
        rms_prop = kwargs['rms_prop']

        self._batch_size = kwargs['batch_size']
        self._online_batches = kwargs['online_batches']

        # Build planning and learning graph
        with graph.as_default():

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

            for task in data.tasks:

                # Define the value and reward functions
                values, rewards, reward_penalty = self._model.task()
                self._reward_functions[task] = rewards

                # Define state values
                batch_values = tf.gather(values, self._state_input)
                mean = tf.expand_dims(tf.reduce_mean(batch_values, axis=1), axis=1)
                variance = 0.001 + tf.expand_dims(tf.reduce_mean(tf.square(batch_values - mean), axis=1), axis=1)
                normalized = beta * ((batch_values - mean) / tf.sqrt(variance))

                # Define the action prediction loss
                partition = tf.log(tf.reduce_sum(tf.exp(normalized), axis=1))
                likelihood = tf.reduce_sum(tf.one_hot(self._action_input, len(Action)) * normalized, axis=1)
                actions = tf.reduce_mean(partition - likelihood)

                # Define bellman error - uses the underlying sensor model
                policy = tf.exp(beta * values)
                v = rewards + (tf.reduce_sum(policy * values, axis=1) / tf.reduce_sum(policy, axis=1))

                residual = values - (self._model.base_gamma * tf.gather(v, self._model.base_transitions))
                bellman = tf.reduce_mean(tf.square(tf.where(self._model.visible, residual, tf.zeros_like(residual))))

                # Define loss and update
                loss = reward_penalty + self._model.penalty + actions + (bellman_weight * bellman)

                if rms_prop:
                    self._reward_updates[task] = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)
                else:
                    self._reward_updates[task] = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

                # Define the action output
                self._policies[task] = tf.argmax(values, axis=1)

            # Initialize the model
            session.run(tf.global_variables_initializer())

        # Pre-train the model
        session.run(self._model.sensor_update, feed_dict={
            self._model.sensor_input: sensor.map
        })

        for task in data.tasks:
            update = self._reward_updates[task]
            samples = data.steps(task)

            for b in range(pretrain_batches):
                batch = np.random.choice(samples, self._batch_size)
                states = []
                actions = []

                for step in batch:
                    states.append(((step.x * sensor.height) + step.y))
                    actions.append(step.action)

                session.run(update, feed_dict={
                    self._state_input: states,
                    self._action_input: actions
                })

        print("HAL CLEAR PROBABILITY: " + str(session.run(self._model.average)))

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
                    states.append((step.x * self._sensor.height) + step.y)
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

        return self._policy[(x * self._sensor.height) + y]

    def rewards(self, task):
        """
        Gets the reward function for a given task.

        :param task: the name of the task
        :return: a 2D np.array containing the rewards, the minimum value, the maximum value
        """

        rewards = self._session.run(self._reward_functions[task])
        reward = np.empty((self._sensor.width, self._sensor.height), dtype=np.float32)

        for x in range(self._sensor.width):
            for y in range(self._sensor.height):
                reward[x, y] = rewards[(x * self._sensor.height) + y]

        return reward


def builder(model_fn,
            beta=1.0,
            bellman_weight=1.0,
            learning_rate=0.001,
            batch_size=128,
            pretrain_batches=100,
            online_batches=50,
            rms_prop=False):
    """
    Returns a builder which itself returns a context manager which
    constructs an HAL agent with the given configuration

    :param model_fn: the function used to construct the abstract model
    :param beta: the action selection temperature
    :param bellman_weight: the relative weight of the bellman error relative to the action prediction loss.
    :param learning_rate: the learning rate for training the cost functions
    :param batch_size: the batch size for training the cost functions
    :param pretrain_batches: the number of batch updates to perform to build the initial cost estimates
    :param online_batches: the number of batch updates to perform after each model update
    :param rms_prop: whether to use RMSProp updates instead of the default Adam updates
    :return: a new builder for BAM agents
    """

    def manager(sensor, data):

        class Manager:
            def __enter__(self):
                self._graph = tf.Graph()

                gpu_options = tf.GPUOptions(allow_growth=True)
                config = tf.ConfigProto(gpu_options=gpu_options)
                self._session = tf.Session(graph=self._graph, config=config)

                try:
                    agent = Agent(self._graph, self._session, sensor, data,
                                  model_fn=model_fn,
                                  beta=beta,
                                  bellman_weight=bellman_weight,
                                  learning_rate=learning_rate,
                                  batch_size=batch_size,
                                  pretrain_batches=pretrain_batches,
                                  online_batches=online_batches,
                                  rms_prop=rms_prop)
                except Exception as e:
                    self._session.close()
                    raise e

                return agent

            def __exit__(self, exc_type, exc_val, exc_tb):
                self._session.close()
                self._graph = None

                return False

        return Manager()

    return manager
