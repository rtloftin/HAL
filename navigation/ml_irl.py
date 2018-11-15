"""
An implementation of ML-IRL that has access to the true transition dynamics.
"""

from .environment import Action
from .sensor import Occupancy

import tensorflow as tf
import numpy as np
import time


class Agent:
    """
    A multi-task, ML-IRL agent.
    """

    def __init__(self, env, data, graph, session, **kwargs):
        """
        Constructs the agent and initializes the cost function estimates.

        :param env: the navigation environment the agent operates in
        :param data: the demonstrated state-action trajectories
        :param graph: the TensorFlow graph for the agent to use
        :param session: the TensorFlow session for the agent to use
        :param kwargs: the configuration parameters for the agent.
        """

        # Capture session
        self._session = session

        # Capture environment width and height
        self._width = env.width
        self._height = env.height

        # Unpack configuration
        beta = kwargs['beta']
        gamma = kwargs['gamma']
        planning_depth = kwargs['planning_depth']
        penalty = kwargs['penalty']
        learning_rate = kwargs['learning_rate']
        batch_size = kwargs['batch_size']
        num_batches = kwargs['num_batches']
        rms_prop = kwargs['rms_prop']

        # Get the number of states and actions
        num_states = env.width * env.height
        num_actions = len(Action)

        # Define the transition model
        transitions = np.empty([env.width * env.height, len(Action)], dtype=np.int32)

        def next(cell, nx, ny):
            if 0 <= nx < env.width and 0 <= ny < env.height and not env.occupied[nx, ny]:
                return (nx * env.height) + ny
            return cell

        for x in range(env.width):
            for y in range(env.height):
                cell = (x * env.height) + y
                transitions[cell, Action.STAY] = cell
                transitions[cell, Action.UP] = next(cell, x, y + 1)
                transitions[cell, Action.DOWN] = next(cell, x, y - 1)
                transitions[cell, Action.LEFT] = next(cell, x - 1, y)
                transitions[cell, Action.RIGHT] = next(cell, x + 1, y)

        # Build learning graph for each task
        self._reward_functions = dict()
        self._reward_updates = dict()
        self._policies = dict()
        self._policy = None

        with graph.as_default():

            # Define transition constant
            transitions = tf.constant(transitions, dtype=tf.int32)

            # Define state and action inputs
            self._state_input = tf.placeholder(tf.int32, shape=[batch_size])
            self._action_input = tf.placeholder(tf.int32, shape=[batch_size])

            for task in data.tasks:

                # Define the reward function
                reward = tf.Variable(tf.zeros([num_states], dtype=tf.float32))
                self._reward_functions[task] = reward

                # Build task value functions
                values = tf.zeros([num_states, num_actions], dtype=tf.float32)

                for _ in range(planning_depth):
                    policy = tf.exp(beta * gamma * values)
                    normal = tf.reduce_sum(policy, axis=1)
                    values = reward + (gamma * tf.reduce_sum(policy * values, axis=1) / normal)
                    values = tf.gather(values, transitions)

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

                # Define the policy output
                self._policies[task] = tf.argmax(values, axis=1)

            # Initialize the model
            session.run(tf.global_variables_initializer())

        # Train reward functions
        for task in data.tasks:
            update = self._reward_updates[task]
            samples = data.steps(task)

            for b in range(num_batches):
                batch = np.random.choice(samples, batch_size)
                states = []
                actions = []

                for step in batch:
                    states.append(((step.x * env.height) + step.y))
                    actions.append(step.action)

                session.run(update, feed_dict={
                    self._state_input: states,
                    self._action_input: actions
                })

    def update(self):
        """
        A dummy update method for compatibility with the other agent classes.
        """
        pass

    def task(self, name):
        """
        Sets the task the agent is currently performing.

        :param name: the name of the task
        """

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
        reward = np.empty((self._width, self._height), dtype=np.float32)

        for x in range(self._width):
            for y in range(self._height):
                reward[x, y] = rewards[(x * self._height) + y]

        return reward


def builder(env,
            beta=1.0,
            gamma=0.99,
            planning_depth=150,
            penalty=100.,
            learning_rate=0.001,
            batch_size=128,
            num_batches=500,
            rms_prop=False):
    """
    Returns a builder which itself returns a context manager which
    constructs a model-based ML-IRL agent with the given configuration.

    :param env: the navigation environment this agent must operate in
    :param beta: the temperature parameter for the soft value iteration
    :param gamma: the discount factor
    :param planning_depth: the number of value iterations to perform
    :param penalty: the regularization term for the cost functions
    :param learning_rate: the learning rate for training the cost functions
    :param batch_size: the batch size for training the cost functions
    :param num_batches: the number of batch updates to perform to find the cost estimates
    :param rms_prop: whether to use RMSProp updates instead of the default Adam updates
    :return: a new builder for ML-IRL agents
    """

    def manager(sensor, data):

        class Manager:
            def __enter__(self):
                self._graph = tf.Graph()

                gpu_options = tf.GPUOptions(allow_growth=True)
                config = tf.ConfigProto(gpu_options=gpu_options)
                self._session = tf.Session(graph=self._graph, config=config)

                try:
                    agent = Agent(env, data, self._graph, self._session,
                                  beta=beta,
                                  gamma=gamma,
                                  planning_depth=planning_depth,
                                  penalty=penalty,
                                  learning_rate=learning_rate,
                                  batch_size=batch_size,
                                  num_batches=num_batches,
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
