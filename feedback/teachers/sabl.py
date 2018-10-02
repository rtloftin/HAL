"""
Defines a teacher that provides feedback using
the SABL model, where an action is correct if it
matches the action provided by the expert for a task
"""

import numpy as np


class Agent:
    """
    A multi-task teacher which gives feedback according to the SABL
    model.  Actions are considered correct if they match the expert's
    sampled action for the current state.
    """

    def __init__(self, **kwargs):
        """
        Initializes the teacher, but does not assign it to a specific environment.

        :param kwargs: the configuration parameters
        """

        self._epsilon = kwargs['epsilon']
        self._mu_plus = kwargs['mu_plus']
        self._mu_minus = kwargs['mu_minus']
        self._action_tolerance = kwargs['action_tolerance']
        self._environment = None

    def environment(self, env):
        """
        Gives the teacher access to the current environment, and
        the corresponding expert policies.

        :param env: the environment defining the tasks to be taught
        """

        self._environment = env

    def feedback(self, action):
        """
        Gets feedback for the given action.  Feedback depends on
        the current state of the environment, and on the current
        task for the environment.

        :param action: the agent's action
        :return: the real-valued feedback signal
        """

        expert_action = self._environment.expert()

        if self._environment.action_space.discrete:
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


def manager(epsilon=0.0,
            mu_plus=0.9,
            mu_minus=0.9,
            action_tolerance=0.1):
    """
    Returns a context manager which is used to instantiate and clean up a
    SABL teacher. This isn't really necessary for this type of teacher, but
    it maintains a common interface with the other types of teachers.

    :param epsilon: the teacher's error rate in providing feedback
    :param mu_plus: the probability of giving no feedback for a correct action
    :param mu_minus: the probability of giving no feedback for an incorrect actiona
    :param action_tolerance: the maximum error for a correct action (only applies for continuous actions)
    :return: a context manager which creates a new SABL teacher
    """

    class Manager:

        def __enter__(self):
            return Agent(epsilon=epsilon,
                         mu_plus=mu_plus,
                         mu_minus=mu_minus,
                         action_tolerance=action_tolerance)

        def __exit__(self, exc_type, exc_val, exc_tb):
            """
            Closes the session associated with the current agent.

            :param exc_type: ignored
            :param exc_val: ignored
            :param exc_tb: ignored
            :return: always False, never suppress exceptions
            """

            return False

    return Manager()
