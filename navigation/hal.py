"""
An implementation of the HAL algorithm for the robot navigation domain.
"""

from .environment import Action
from .sensor import Occupancy

import tensorflow as tf
import numpy as np


class Agent:
    """
    A HAL agent.
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
        planning_depth = kwargs['planning_depth']
        obstacle_mean = kwargs['obstacle_mean']
        obstacle_variance = kwargs['obstacle_variance']
        penalty = kwargs['penalty']
        learning_rate = kwargs['learning_rate']
        pretrain_batches = kwargs['pretrain_batches']

        self._batch_size = kwargs['batch_size']
        self._online_batches = kwargs['online_batches']

        # Get the number of states and actions
        num_states = sensor.width * sensor.height
        num_actions = len(Action)

        # Define the transition structure
        transitions = np.empty([num_states, num_actions, 2], dtype=np.int32)

        def cell(x, y, nx, ny, action):
            current = (x * sensor.height) + y

            if 0 <= nx < sensor.width and 0 <= ny < sensor.height:
                transitions[current, action, 0] = (nx * sensor.height) + ny
            else:
                transitions[current, action, 0] = current

            transitions[current, action, 1] = current

        for x in range(sensor.width):
            for y in range(sensor.height):
                cell(x, y, x, y, Action.STAY)
                cell(x, y, x, y + 1, Action.UP)
                cell(x, y, x, y - 1, Action.DOWN)
                cell(x, y, x - 1, y, Action.LEFT)
                cell(x, y, x + 1, y, Action.RIGHT)

        # Build planning and learning graph
        with graph.as_default():

            # Define transition constant
            transitions = tf.constant(transitions, dtype=tf.int32)

            # Define occupancy mask
            self._occupancy = tf.placeholder(tf.int32, shape=[sensor.width, sensor.height])
            occupancy = tf.reshape(self._occupancy, [num_states])

            visible = tf.Variable(tf.fill([num_states], False), dtype=tf.bool, trainable=False, use_resource=True)
            occupied = tf.Variable(tf.fill([num_states], False), dtype=tf.bool, trainable=False, use_resource=True)

            self._occupancy_update = tf.group(tf.assign(visible, tf.not_equal(occupancy, Occupancy.UNKNOWN)),
                                              tf.assign(occupied, tf.equal(occupancy, Occupancy.OCCUPIED)))

            # Define dynamics model
            model = tf.Variable(tf.fill([num_states], obstacle_mean), dtype=tf.float32)
            model_penalty = tf.reduce_mean(obstacle_variance * tf.square(obstacle_mean - model))

            # Define transition probabilities
            probabilities = tf.nn.sigmoid(model)
            probabilities = tf.where(visible, tf.where(occupied, tf.ones([num_states], dtype=tf.float32),
                                                       tf.zeros([num_states], dtype=tf.float32)), probabilities)

            succeed = tf.gather(1. - probabilities, transitions[:, :, 0])
            fail = tf.gather(probabilities, transitions[:, :, 0])

            probabilities = tf.stack([succeed, fail], axis=-1)

            # Define state and action inputs
            self._state_input = tf.placeholder(tf.int32, shape=[self._batch_size])
            self._action_input = tf.placeholder(tf.int32, shape=[self._batch_size])
            self._policy_input = tf.placeholder(tf.int32, shape=[1])

            # Define value iteration update
            def update(q, r):
                policy = tf.exp(beta * gamma * q)
                normal = tf.reduce_sum(policy, axis=1)
                v = r + (gamma * tf.reduce_sum(policy * q, axis=1) / normal)

                q = tf.gather(v, transitions)
                q = tf.reduce_sum(probabilities * q, axis=2)

                return q

            # Build individual task models
            self._reward_functions = dict()
            self._reward_updates = dict()
            self._policies = dict()
            self._policy = None

            for task in data.tasks:

                # Define the reward function
                reward = tf.Variable(tf.zeros([num_states], dtype=tf.float32))
                self._reward_functions[task] = reward

                # Define value function
                values = tf.zeros([num_states, num_actions], dtype=tf.float32)

                for _ in range(planning_depth):
                    values = update(values, reward)

                policy_value = beta * tf.gather(values, self._policy_input)
                values = beta * tf.gather(values, self._state_input)

                # Define the action prediction loss
                partition = tf.log(tf.reduce_sum(tf.exp(values), axis=1))
                likelihood = tf.reduce_sum(tf.one_hot(self._action_input, num_actions) * values, axis=1)

                loss = tf.reduce_mean(partition - likelihood)
                loss = loss + (penalty * tf.reduce_mean(tf.square(reward))) + model_penalty
                self._reward_updates[task] = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

                # Define the action output
                # self._policies[task] = tf.multinomial(policy_value, 1)
                self._policies[task] = tf.argmax(policy_value, axis=1)

            # Initialize the model
            session.run(tf.global_variables_initializer())

        # Pre-train the model
        session.run(self._occupancy_update, feed_dict={
            self._occupancy: sensor.map
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

    def update(self):
        """
        Updates the agent's cost estimates to reflect new sensor data.
        """

        self._session.run(self._occupancy_update, feed_dict={
            self._occupancy: self._sensor.map
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

        self._policy = self._policies[name]

    def act(self, x, y):
        """
        Samples an action from the agent's policy for the current state

        :param x: the agent's x coordinate
        :param y: the agent's y coordinate
        :return: the sampled action
        """

        # return self._session.run(self._policy, feed_dict={
        #     self._state_input: [(x * self._sensor.height) + y]
        # })[0, 0]

        return self._session.run(self._policy, feed_dict={
            self._policy_input: [(x * self._sensor.height) + y]
        })[0]

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


def builder(beta=1.0,
            gamma=0.99,
            base_depth=10,
            abstract_depth=10,
            planning_depth=150,
            obstacle_mean=-0.5,
            obstacle_variance=0.2,
            penalty=0.1,
            learning_rate=0.01,
            batch_size=64,
            pretrain_batches=100,
            online_batches=100):
    """
    Returns a builder which itself returns a context manager which
    constructs an HAL agent with the given configuration.

    :param beta: the temperature parameter for the soft value iteration
    :param gamma: the discount factor
    :param planning_depth: the number of value iterations to perform
    :param obstacle_mean: the mean of the log probability of an obstacle
    :param obstacle_variance: the variance of the log probability
    :param penalty: the regularization term for the cost functions
    :param learning_rate: the learning rate for training the cost functions
    :param batch_size: the batch size for training the cost functions
    :param pretrain_batches: the number of batch updates to perform to build the initial cost estimates
    :param online_batches: the number of batch updates to perform after each model update
    :return: a new builder for BAM agents
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
                                  planning_depth=planning_depth,
                                  obstacle_mean=obstacle_mean,
                                  obstacle_variance=obstacle_variance,
                                  penalty=penalty,
                                  learning_rate=learning_rate,
                                  batch_size=batch_size,
                                  pretrain_batches=pretrain_batches,
                                  online_batches=online_batches)
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
