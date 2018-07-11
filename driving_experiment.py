"""
Conducts a set of experiments in the driving domain.

Start with learning from demonstration, then work on
learning from feedback.
"""


def cloning(env, tasks, experts):
    """
    Evaluates the behavioral cloning algorithm in isolation.

    :param env: the environment simulation
    :param tasks: a list of named tasks
    :param experts: a dictionary of expert policies
    """


def mce_irl(env, tasks, experts):
    """
    Evaluates the MCE-IRL algorithm in isolation.

    :param env: the environment simulation
    :param tasks: a list of named tasks
    :param experts: a dictionary of expert policies
    """