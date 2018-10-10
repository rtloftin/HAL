"""
Defines an agent that acts optimally in a given navigation environment.
"""

import numpy as np
import tensorflow as tf

from .environment import Action


class Expert:
    """
    An agent that generates optimal actions for navigation tasks in a specific environment.
    """

    def __init__(self, env):
        """
        Initializes the agent by computing the optimal
        policies for each task in the environment.

        :param env: the environment the expert needs to act in
        """

        # Capture the height of the environment, for computing state indices
        self._height = env.height

        # Define the transition model
        transitions = np.empty([env.width * env.height, len(Action)])

        def index(x, y):
            return (x * env.height) + y

        def valid(x, y):
            if x < 0 or x >= env.height:
                return False
            if y < 0 or y >= env.width:
                return False
            if env.occupied[x, y]:
                return False
            return True

        for x in range(env.width):
            for y in range(env.height):
                cell = index(x, y)
                transitions[cell, Action.STAY] = cell
                transitions[cell, Action.UP] = index(x, y + 1) if valid(x, y + 1) else cell
                transitions[cell, Action.DOWN] = index(x, y - 1) if valid(x, y - 1) else cell
                transitions[cell, Action.LEFT] = index(x - 1, y) if valid(x - 1, y) else cell
                transitions[cell, Action.RIGHT] = index(x + 1, y) if valid(x + 1, y) else cell

        # Define value iteration graph
        graph = tf.Graph()

        with graph.as_default():
            reward_input = tf.placeholder(dtype=tf.float32, shape=[env.width * env.height])

            def update(q, t):
                v = tf.reduce_max(q, axis=1)
                n = tf.gather(v, transitions)

                return reward_input + n

            def limit(q, t):
                return t < 4 * (env.width + env.height)

            value_output = tf.while_loop(limit, update, [tf.zeros_like(transitions), 0])

        # Compute the optimal policies
        self._policies = dict()
        self._policy = None

        with tf.Session(graph=graph) as sess:
            reward = np.empty(env.width * env.height, dtype=np.float32)

            for name, task in env.tasks:

                # Initialize reward function
                for x in range(env.width):
                    for y in range(env.height):
                        reward[index(x, y)] = 1.0 if task.complete(x, y) else 0.0

                # Compute value function
                values = sess.run(value_output, feed_dict={reward_input})

                # Construct policy
                max = np.max(values, axis=1)
                policy = []

                for s in range(env.width * env.height):
                    actions = []

                    for a in range(len(Action)):
                        if values[s, a] == max[s]:
                            actions.append(a)

                    policy.append(actions)

                self._policies[name] = policy

    def task(self, name):
        """
        Sets the task the expert is currently performing

        :param name: the name of the task
        """

        self._policy = self._policies[name]

    def act(self, x, y):
        """
        Samples an expert action for the current state and task.

        :return: the sampled action
        """

        return np.random.choice(self._policy[(self._height * x) + y])
