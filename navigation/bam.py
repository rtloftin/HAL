"""
An implementation of multitask BAM for the robot navigation domain.
"""

from .environment import Action
from .sensor import Occupancy

import tensorflow as tf
import numpy as np


class Agent:
    """
    A BAM agent for the robot navigation domain.
    """

    def __init__(self, sensor, data, graph, session, **kwargs):
        """
        Constructs the agent and initializes the cost function estimates.

        :param sensor: the sensor model used to observe the environment
        :param data: the demonstrated state-action trajectories
        :param graph: the TensorFlow graph for the agent to use
        :param session: the TensorFlow session for the agent to use
        :param kwargs: the configuration parameters for the agent.
        """

        # Capture instance objects
        self._sensor = sensor
        self._data = data
        self._session = session

        # Unpack configuration
        gamma = kwargs['gamma']
        beta = kwargs['beta']
        penalty = kwargs['penalty']
        iterations = kwargs['iterations']
        learning_rate = kwargs['learning_rate']

        self._baseline = kwargs['baseline']
        self._batch_size = kwargs['batch_size']
        self._num_batches = kwargs['num_batches']

        # Initialize the transition dynamics model
        self._transitions = np.empty([sensor.width * sensor.height, len(Actions), 2], dtype=np.int32)
        self._probabilities = np.empty_like(self._transitions, dtype=np.float32)

        self._build()

        # Build planning and learning graph
        self._state_inputs = dict()
        self._action_inputs = dict()
        self._action_outputs = dict()
        self._reward_updates = dict()

        with graph.as_default():

            # Define model variables
            transitions = tf.Variable(self._transitions, dtype=tf.int32, trainable=False, use_resource=True)
            probabilities = tf.Variable(self._probabilities, dtype=tf.float32, trainable=False, use_resource=True)

            self._transition_input = tf.placeholder(dtype=tf.int32, shape=self._transitions.shape)
            self._probability_input = tf.placeholder(dtype=tf.int32, shape=self._probabilities.shape)

            self._transfer = tf.group(tf.assign(transitions, self._transition_input),
                                      tf.assign(probabilities, self._probability_input))

            # Define value iteration
            def update(q, t):
                v = tf.log(tf.reduce_sum(tf.exp(beta * q), axis=1))
                q = tf.gather(v, transitions)
                return reward + (gamma * tf.reduce_sum(probabilities * q, axis=2))

            def limit(q, t):
                return t < iterations

            # Iterate over all tasks
            for task in data.tasks:

                # Define the reward function
                reward = tf.Variable(tf.random_normal([sensor.width * sensor.height], stddev=0.1, dtype=tf.float32))

                # Define value function
                values = tf.while_loop(limit, update, [tf.zeros([sensor.width * sensor.height, len(Action)]), 0])
                values = beta * tf.gather(values, state_input)

                # Define state and action inputs
                state_input = tf.placeholder(dtype=tf.int32, shape=[None])
                action_input = tf.placeholder(dtype=tf.int32, shape=[None])

                # Define the action prediction loss
                partition = tf.log(tf.reduce_sum(tf.exp(values), axis=1))
                likelihood = tf.gather(values, action_input, axis=1)

                loss = tf.reduce_mean(partition - likelihood) + (penalty * tf.reduce_mean(tf.square(reward)))
                update = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

                # Define the action output
                action_output = tf.multinomial(values, 1)

                # Add to dictionaries
                self._state_inputs[task] = state_input
                self._action_inputs[task] = action_input
                self._action_outputs[task] = action_output
                self._reward_updates[task] = update

            # Initialize the model
            session.run(tf.global_variables_initializer())

        # Pre-train the model
        for task in data.tasks:
            update = self._reward_updates[task]
            state_input = self._state_inputs[task]
            action_input = self._action_inputs[task]

            samples = data.steps(task)

            for _ in self._num_batches:
                batch = np.random.choice(samples, self._batch_size)
                states = []
                actions = []

                for step in batch:
                    states.append((step.x * sensor.height) + step.y)
                    actions.append(step.action)

                session.run(update, feed_dict={
                    state_input: states,
                    action_input: actions
                })

    def _cell(self, x, y, new_x, new_y, action):
        current = (x * self._sensor.height) + y
        next = (new_x * self._sensor.height) + new_y

        if 0 <= new_x < self._sensor.width and 0 <= new_y < self._sensor.height:
            self._transitions[current, action, 0] = next
            self._transitions[current, action, 1] = current

            if Occupancy.CLEAR == self._sensor.map[new_x, new_y]:
                success = 1.
            elif Occupancy.OCCUPIED == self._sensor.map[new_x, new_y]:
                success = 0.
            else:
                success = self._baseline

            self._probabilities[current, action, 0] = success
            self._probabilities[current, action, 1] = 1. - success
        else:
            self._transitions[current, action, 0] = current
            self._transitions[current, action, 1] = current
            self._probabilities[current, action, 0] = 1.
            self._probabilities[current, action, 1] = 0.

    def _build(self):
        for x in range(self._sensor.width):
            for y in range(self._sensor.height):
                self._cell(x, y, x, y, Action.STAY)
                self._cell(x, y, x, y + 1, Action.UP)
                self._cell(x, y, x, y - 1, Action.DOWN)
                self._cell(x, y, x - 1, y, Action.LEFT)
                self._cell(x, y, x + 1, y, Action.RIGHT)

    def update(self):
        """
        Updates the agent's cost estimates to reflect new sensor data.
        """

        self._build()
        self._session.run(self._transfer, feed_dict={
            self._transition_input: self._transitions,
            self._probability_input: self._probabilities
        })

        for task in self._data.tasks:
            update = self._reward_updates[task]
            state_input = self._state_inputs[task]
            action_input = self._action_inputs[task]

            samples = self._data.steps(task)

            for _ in self._num_batches:
                batch = np.random.choice(samples, self._batch_size)
                states = []
                actions = []

                for step in batch:
                    states.append((step.x * self._sensor.height) + step.y)
                    actions.append(step.action)

                self._session.run(update, feed_dict={
                    state_input: states,
                    action_input: actions
                })

    def act(self, x, y, task):
        """
        Samples an action from the agent's policy for the current state

        :param x: the agent's x coordinate
        :param y: the agent's y coordinate
        :param task: the name of the task being performed
        :return: the sampled action
        """

        state_input = self._state_inputs[task]
        action_output = self._action_outputs[task]

        return self._session.run(action_output, feed_dict={
            state_input: [(x * self._sensor.height) + y]
        })[0, 0]


def builder(beta=1.0,
            gamma=0.95,
            penalty=0.01,
            iterations=200,
            baseline=0.2,
            learning_rate=0.001,
            batch_size=128,
            num_batches=1000):
    """
    Returns a builder which itself returns a context manager which
    constructs an ML-IRL agent with the given

    :param beta: the temperature parameter for the soft value iteration
    :param gamma: the discount factor
    :param penalty: the regularization term for the cost functions
    :param iterations: the number of value iterations to perform
    :param baseline: the probability of an obstacle being in an unobserved cell
    :param learning_rate: the learning rate for training the cost functions
    :param batch_size: the batch size for training the cost functions
    :param num_batches: the number of batches for training the cost functions
    :return: a new builder for ML-IRL agents
    """

    def manager(sensor, data):

        class Manager:
            def __enter__(self):
                self._graph = tf.Graph()
                self._session = tf.Session(graph=self._graph)

                try:
                    agent = Agent(sensor, data, self._graph, self._session,
                                  beta=beta,
                                  gamma=gamma,
                                  penalty=penalty,
                                  iterations=iterations,
                                  baseline=baseline,
                                  learning_rate=learning_rate,
                                  batch_size=batch_size,
                                  num_batches=num_batches)
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