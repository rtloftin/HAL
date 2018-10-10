"""
An implementation of multitask ML-IRL for the robot navigation domain.
"""

from .environment import Action

import tensorflow as tf
import numpy as np


class Agent:
    """
    A multi-task behavioral cloning agent.  This does not
    implement an interface for learning from interaction
    with the environment, but simply runs supervised on
    a set of demonstrated actions.
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

        # Initialize the transition dynamics model
        self._transitions = np.empty([sensor.width * sensor.height, len(Actions), 2], dtype=np.int32)
        self._probabilities = np.empty_like(self._transitions, dtype=np.float32)

    def update(self):
        """
        Updates the agent's cost estimates to reflect new sensor data.
        """

    def act(self, x, y):
        """
        Samples an action from the agent's policy for

        :param x: the agent's x coordinate
        :param y: the agent's y coordinate
        :return: the sampled action
        """

        return None


def builder(beta=1.0,
            gamma=0.95,
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
