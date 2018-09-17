"""
An implementation of behavioral cloning for learning from demonstration.
"""

import tensorflow as tf
import numpy as np


class TaskAgent:
    """
    A behavioral cloning agent for a single task.
    """

    def __init__(self, graph, session, kwargs):
        """
        Initializes the agent.

        :param graph: the Tensorflow graph to be used
        :param session: the Tensorflow session to be used
        :param kwargs: the configuration session for the agent
        """

        # Capture the session
        self._session = session

        # Capture the learning parameters
        self._batch_size = kwargs['batch_size']
        self._num_batches = kwargs['num_batches']
        self._discrete_action = kwargs['discrete_action']

        # Initialize the data set
        self._data = []

        # Construct the model
        with graph.as_default():
            self._state_input = tf.placeholder(dtype=tf.float32, shape=[None, kwargs['state_size']])
            output = kwargs['model_fn'](self._state_input)

            # Define loss and action output
            if self._discrete_action:
                self._action_input = tf.placeholder(dtype=tf.int32, shape=[None])

                one_hot = tf.one_hot(self._action_input, kwargs['action_size'])
                exp = tf.exp(output)

                loss = tf.reduce_sum(tf.log(tf.reduce_sum(exp, axis=1)) - (one_hot * output))
                self._action = tf.multinomial(output, 1)
            else:
                self._action_input = tf.placeholder(dtype=tf.float32, shape=[None, kwargs['action_size']])

                action_mean, action_deviation = tf.split(output, 2, axis=1)
                action_deviation = tf.exp(tf.multiply(action_deviation, 0.5))

                loss = tf.reduce_sum(tf.square((self._action_input - action_mean) / action_deviation))

                noise = tf.random_normal(tf.shape(action_mean))
                self._action = action_mean + (noise * action_deviation)

            # Define policy update
            self._update = tf.train.AdamOptimizer(learning_rate=kwargs['learning_rate']).minimize(loss)

            # Initialize model
            self._session.run(tf.global_variables_initializer())

    def demonstrate(self, state, action):
        """
        Adds a new state-action pair to the data set.

        :param state: the state
        :param action: the action taken in that state
        """

        self._data.append((state, action))

    def act(self, state):
        """
        Samples an action from the agent's policy for the given state.

        :param state: the current state
        :return: the sampled action
        """

        action = self._session.run(action, feed_dict={self._state_input: [state]})

        if self._discrete_action:
            return action[0, 0]
        else:
            return action[0]

    def update(self):
        """
        Updates the agent's policy based on the available data
        """

        for _ in range(self._num_batches):
            batch = np.random.choice(self._data, size=self._batch_size)
            states = []
            actions = []

            for sample in batch:
                states.append(sample[0])
                actions.append(sample[1])

            self._session.run(self._update, feed_dict={
                self._state_input: states,
                self._action_input: actions
            })


class Agent:
    """
    A multi-task behavioral cloning agent.  This does not
    implement an interface for learning from interaction
    with the environment, but simply runs supervised on
    a set of demonstrated actions.
    """

    def __init__(self, kwargs):
        """
        Initializes the agent.  Just initializes the dictionary of
        task models and the Tensorflow graph and session.

        :param kwargs: the configuration parameters for the agent.
        """

        # Capture configuration
        self._kwargs = kwargs

        # Create graph and session
        self._graph = tf.Graph()
        self._session = tf.Session(graph=self._graph)

        # Create task dictionary and define current task
        self._tasks = dict()
        self._current = None

    def set_task(self, name):
        """
        Sets the task that the agent is currently learning to perform.

        :param name: the name of the task
        """

        if name not in self._tasks:
            self._tasks[name] = TaskAgent(self._graph, self._session, self._kwargs)

        self._current = self._tasks[name]

    def reset(self):
        """
        Indicates to the agent that a new episode has started, that is
        the current state was sampled independently of the previous state.

        This is a dummy method because behavioral cloning doesn't care about transitions
        """

    def demonstrate(self, state, action):
        """
        Demonstrates the teacher's next action.

        :param state: the current state
        :param action:  the teacher's action
        """

        self._current.demonstrate(state, action)

    def act(self, state):
        """
        Samples an action from the agent's policy for the current task.

        :param state: the current state
        :return: the sampled action
        """

        return self._current.act(state)


def factory(model_fn, state_size, action_size,
            discrete_action=False,
            learning_rate=0.01,
            batch_size=10,
            num_batches=1000):
    """
    Gets a method which constructs new multi-task behavioral cloning agents.

    :param model_fn: the function used to build the model graph
    :param state_size: the number of state features
    :param action_size: the number of actions or action features
    :param discrete_action: whether or not the actions are discrete
    :param learning_rate: the learning rate used for training the policies
    :param batch_size: the batch size used for training the policies
    :param num_batches: the number of batches used for training the policies
    :return: a new behavioral cloning agent
    """
    kwargs = {
        'model_fn': model_fn,
        'state_size': state_size,
        'action_size': action_size,
        'discrete_action': discrete_action,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'num_batches': num_batches
    }

    return lambda: Agent(kwargs)
