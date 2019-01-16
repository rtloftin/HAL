"""
An implementation of multitask, model-based ML-IRL for the robot navigation domain.
"""

from .environment import Action
from .sensor import Occupancy

import tensorflow as tf
import numpy as np
import time


class Agent:
    """
    A multi-task, model-based ML-IRL agent.
    """

    def __init__(self, env, **kwargs):
        """
        Constructs a new model-based ML-IRL agent.

        :param env: the navigation environment this agent must operate in
        :param kwargs: the configuration parameters for the agent.
        """

        # Capture environment
        self._env = env

        # Capture environment height
        self._height = env.height

        # Unpack configuration
        gamma = kwargs['gamma']
        beta = kwargs['beta']
        planning_depth = kwargs['planning_depth']
        obstacle_prior = kwargs['obstacle_prior']
        penalty = kwargs['penalty']
        learning_rate = kwargs['learning_rate']
        rms_prop = kwargs['rms_prop']
        use_baseline = kwargs['use_baseline']

        self._pretrain_batches = kwargs['pretrain_batches']
        self._batch_size = kwargs['batch_size']
        self._online_batches = kwargs['online_batches']

        # Get the number of states and actions
        num_states = env.width * env.height
        num_actions = len(Action)

        # Define the transition model
        transitions = np.empty([num_states, num_actions], dtype=np.int32)

        def nxt(cell, nx, ny):
            if 0 <= nx < env.width and 0 <= ny < env.height:
                return (nx * env.height) + ny
            return cell

        for x in range(env.width):
            for y in range(env.height):
                cell = (x * env.height) + y
                transitions[cell, Action.STAY] = cell
                transitions[cell, Action.UP] = nxt(cell, x, y + 1)
                transitions[cell, Action.DOWN] = nxt(cell, x, y - 1)
                transitions[cell, Action.LEFT] = nxt(cell, x - 1, y)
                transitions[cell, Action.RIGHT] = nxt(cell, x + 1, y)

        # Build planning and learning graph
        self._reward_functions = dict()
        self._reward_updates = dict()
        self._policies = dict()
        self._policy = None

        self._values = dict()
        self._value = None

        # Initialize placeholders
        self._session = None
        self._sensor = None
        self._data = None

        # Construct computation graph
        self._graph = tf.Graph()

        with self._graph.as_default():

            # Define state and action inputs
            self._state_input = tf.placeholder(tf.int32, shape=[self._batch_size])
            self._action_input = tf.placeholder(tf.int32, shape=[self._batch_size])

            # Define transition constant
            transitions = tf.constant(transitions, dtype=tf.int32)

            # Define transition probability model
            self._sensor_input = tf.placeholder(tf.int32, shape=[env.width, env.height])
            occupancy = tf.reshape(self._sensor_input, [num_states])

            success = tf.where(tf.equal(occupancy, Occupancy.CLEAR),
                               tf.ones([num_states], dtype=tf.float32), tf.zeros([num_states], dtype=tf.float32))
            success = tf.where(tf.equal(occupancy, Occupancy.UNKNOWN),
                               tf.fill([num_states], 1. - obstacle_prior), success)

            probabilities = tf.Variable(tf.zeros([num_states, num_actions], dtype=tf.float32),
                                        trainable=False, use_resource=True)

            self._transition_update = tf.assign(probabilities, tf.gather(success, transitions))

            # Build individual task models
            for task, _ in env.tasks:

                # Define the reward function
                reward = tf.Variable(tf.zeros([num_states], dtype=tf.float32))
                self._reward_functions[task] = reward

                # Define value functions
                values = tf.zeros([num_states, num_actions], dtype=tf.float32)

                for _ in range(planning_depth):
                    if use_baseline:
                        baseline = tf.expand_dims(tf.reduce_mean(values, axis=1), axis=1)
                        policy = tf.exp(beta * gamma * (values - baseline))
                    else:
                        policy = tf.exp(beta * gamma * values)

                    normal = tf.reduce_sum(policy, axis=1)
                    v = reward + (gamma * tf.reduce_sum(policy * values, axis=1) / normal)

                    values = probabilities * tf.gather(v, transitions)
                    values = values + (tf.expand_dims(v, axis=1) * (1. - probabilities))

                # Define the action prediction loss
                batch_values = tf.gather(values, self._state_input)
                mean = tf.expand_dims(tf.reduce_mean(batch_values, axis=1), axis=1)
                variance = 0.001 + tf.expand_dims(tf.reduce_mean(tf.square(batch_values - mean), axis=1), axis=1)
                normalized = beta * gamma * ((batch_values - mean) / tf.sqrt(variance))

                partition = tf.log(tf.reduce_sum(tf.exp(normalized), axis=1))
                likelihood = tf.reduce_sum(tf.one_hot(self._action_input, num_actions) * normalized, axis=1)

                loss = tf.reduce_mean(partition - likelihood) + (penalty * tf.reduce_mean(tf.square(reward)))

                if rms_prop:
                    self._reward_updates[task] = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)
                else:
                    self._reward_updates[task] = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

                # Define the action output
                self._policies[task] = tf.argmax(values, axis=1)

                # Define value function output
                self._values[task] = values

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

            self._session.run(self._transition_update, feed_dict={
                self._sensor_input: sensor.map
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

        self._session.run(self._transition_update, feed_dict={
            self._sensor_input: self._sensor.map
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

        self._session.run(self._transition_update, feed_dict={
            self._sensor_input: self._sensor.map
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
        reward = np.empty((self._env.width, self._env.height), dtype=np.float32)

        for x in range(self._env.width):
            for y in range(self._env.height):
                reward[x, y] = rewards[(x * self._env.height) + y]

        return reward


def builder(beta=1.0,
            gamma=0.99,
            planning_depth=150,
            obstacle_prior=0.2,
            penalty=100.,
            learning_rate=0.001,
            batch_size=128,
            pretrain_batches=100,
            online_batches=50,
            rms_prop=False,
            use_baseline=False):
    """
    Returns a factory method for constructing model-based ML-IRL agents with the given configuration.

    :param beta: the temperature parameter for the soft value iteration
    :param gamma: the discount factor
    :param planning_depth: the number of value iterations to perform
    :param obstacle_prior: the probability of an obstacle being in an unobserved cell
    :param penalty: the regularization term for the cost functions
    :param learning_rate: the learning rate for training the cost functions
    :param batch_size: the batch size for training the cost functions
    :param pretrain_batches: the number of batch updates to perform to build the initial cost estimates
    :param online_batches: the number of batch updates to perform after each model update
    :param rms_prop: whether to use RMSProp updates instead of the default Adam updates
    :param use_baseline: whether to use a mean baseline when computing intermediate policies
    :return: a new builder for ML-IRL agents
    """

    def build(env):
        return Agent(env,
                     beta=beta,
                     gamma=gamma,
                     planning_depth=planning_depth,
                     obstacle_prior=obstacle_prior,
                     penalty=penalty,
                     learning_rate=learning_rate,
                     batch_size=batch_size,
                     pretrain_batches=pretrain_batches,
                     online_batches=online_batches,
                     rms_prop=rms_prop,
                     use_baseline=use_baseline)

    return build
