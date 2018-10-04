"""
Defines a teacher that provides feedback using
the SABL model, where an action is correct if it
matches the action provided by the expert for a task.

While this module handles continuous actions, it is not practical
for high-dimensional actions spaces, as the probability of positive
feedback is too low.
"""

import numpy as np


class Teacher:
    """
    A multi-task teacher which gives feedback according to the SABL
    model.  Actions are considered correct if they match the expert's
    sampled action for the current state.
    """

    def __init__(self, env, **kwargs):
        """
        Initializes the teacher, but does not assign it to a specific environment.

        :param kwargs: the configuration parameters
        """

        self._env = env
        self._epsilon = kwargs['epsilon']
        self._mu_plus = kwargs['mu_plus']
        self._mu_minus = kwargs['mu_minus']
        self._action_tolerance = kwargs['action_tolerance']
        self._environment = None

    def feedback(self, action):
        """
        Gets feedback for the given action.  Feedback depends on
        the current state of the environment, and on the current
        task for the environment.

        :param action: the agent's action
        :return: the real-valued feedback signal
        """

        expert_action = self._env.expert()

        if self._env.action_space.discrete:
            is_correct = (expert_action == action)
        else:
            expert_action = np.asarray(expert_action, dtype=np.float32)
            action = np.asarray(action, dtype=np.float32)

            error = max(abs(expert_action - action))
            is_correct = (error <= self._action_tolerance)

        if np.random.rand() <= self._epsilon:
            is_correct = not is_correct

        if is_correct:
            return 0.0 if np.random.rand() <= self._mu_plus else 1.0
        else:
            return 0.0 if np.random.rand() <= self._mu_minus else -1.0


def builder(epsilon=0.0,
            mu_plus=0.9,
            mu_minus=0.9,
            action_tolerance=0.1):
    """
    Returns a builder which constructs a context manager used to allocate and initialize a new SABL teacher.

    :param epsilon: the teacher's error rate in providing feedback
    :param mu_plus: the probability of giving no feedback for a correct action
    :param mu_minus: the probability of giving no feedback for an incorrect actiona
    :param action_tolerance: the maximum error for a correct action (only applies for continuous actions)
    :return: a SABL builder with the desired configuration
    """

    def manager(env, data):

        class Manager:

            def __enter__(self):
                return Teacher(env,
                               epsilon=epsilon,
                               mu_plus=mu_plus,
                               mu_minus=mu_minus,
                               action_tolerance=action_tolerance)

            def __exit__(self, exc_type, exc_val, exc_tb):
                return False

        return Manager()

    return manager
