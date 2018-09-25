"""
An implementation of the basic TAMER algorithm without temporal credit
assignment.  Allows for real-valued feedback.
"""

import tensorflow as tf
import numpy as np


class TaskAgent:
    """
    A TAMER agent for a single task.
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
        self._action_size = kwargs['action_size']
        self._epsilon = kwargs['epsilon']

        # Compute action ranges
        low = np.asarray(kwargs['action_low'], dtype=np.float32)
        high = np.asarray(kwargs['action_high'], dtype=np.float32)

        self._action_scale = high - low
        self._action_offset = low

        # Initialize the data set
        self._data = []

        # Construct the model
        with graph.as_default():
            self._feedback_input = tf.placeholder(dtype=tf.float32, shape=[None])
            self._state_input = tf.placeholder(dtype=tf.float32, shape=[None, self._action_size])

            output = kwargs['model_fn'](self._state_input)

            # Define loss and action output
            if self._discrete_action:
                self._action_input = tf.placeholder(dtype=tf.int32, shape=[None])

                one_hot = tf.one_hot(self._action_input, self._action_size)
                value = tf.reduce_sum(one_hot * output, axis=1)

                loss = tf.reduce_sum(tf.square(value - feedback))
                self._action = tf.argmax(output, axis=1)
            else:
                self._action_input = tf.placeholder(dtype=tf.float32, shape=[None, self._action_size])

                action_center = output[:, 1:self._action_size + 1]
                action_weight = output[:, -self._action_size:]

                regret = tf.reduce_sum(tf.exp(action_weight) * tf.square(action_center - self._action_input), axis=1)
                value = output[:, 0] - regret

                loss = tf.reduce_sum(tf.square(self._feedback_input - value))
                self._action = action_center

            # Define policy update
            self._update = tf.train.AdamOptimizer(learning_rate=kwargs['learning_rate']).minimize(loss)

            # Initialize model
            self._session.run(tf.global_variables_initializer())

    def feedback(self, state, action, feedback):
        """
        Adds a new state-action pair, with associated feedback, to the data set.

        :param state: the state
        :param action: the action taken in that state
        :param feedback: the feedback for this action
        """

        self._data.append((state, action, feedback))

    def act(self, state, explore):
        """
        Samples an action from the agent's policy for the given state.

        :param state: the current state
        :param explore: whether or not to allow exploratory actions
        :return: the sampled action
        """

        if explore and np.random.rand() <= self._epsilon:
            if self._discrete_action:
                return np.random.randint(0, self._action_size)
            else:
                return (np.random.rand(self._action_size) * self._action_scale) + self._action_offset
        else:
            return self._session.run(action, feed_dict={self._state_input: [state]})[0]

    def update(self):
        """
        Updates the agent's policy based on the available data
        """

        for _ in range(self._num_batches):
            batch = np.random.choice(self._data, size=self._batch_size)
            states = []
            actions = []
            feedback = []

            for sample in batch:
                states.append(sample[0])
                actions.append(sample[1])
                feedback.append(sample[2])

            self._session.run(self._update, feed_dict={
                self._state_input: states,
                self._action_input: actions,
                self._feedback_input: feedback
            })


class Agent:
    """
    A multi-task TAMER agent.  Does not implement an
    interface for learning from interaction with the environment.
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

        This is a dummy method because TAMER doesn't care about transitions
        """

    def feedback(self, state, action, feedback):
        """
        Provides feedback for an action in a given state

        :param state: the current state
        :param action:  the agent's action
        :param feedback: the teacher's feedback signal
        """

        if 0.0 != feedback:
            self._current.feedback(state, action, feedback)

    def act(self, state, explore=False):
        """
        Samples an action from the agent's policy for the current task.

        :param state: the current state
        :param explore: whether to allow exploratory actions
        :return: the sampled action
        """

        return self._current.act(state, explore)


def factory(model_fn, state_size, action_size,
            action_high=None,
            action_low=None,
            discrete_action=False,
            epsilon=0.15,
            learning_rate=0.01,
            batch_size=10,
            num_batches=1000):
    """
    Gets a method which constructs a new multitask TAMER agent. This agent does not
    do temporal credit assignment, but does allow real-valued feedback.  For continuous action
    spaces, we use a variation on the Normalized Advantage Function architecture.

    :param model_fn: the function used to build the model graph
    :param state_size: the number of state features
    :param action_size: the number of actions or action features
    :param action_high: the maximum value for each action dimension (continuous action)
    :param action_low: the minimum value for each action dimension (continuous action)
    :param discrete_action: whether or not the actions are discrete
    :param epsilon: the rate at which the agent selects suboptimal actions when exploring
    :param learning_rate: the learning rate used for training the policies
    :param batch_size: the batch size used for training the policies
    :param num_batches: the number of batches used for training the policies
    :return: a new TAMER agent
    """

    kwargs = {
        'model_fn': model_fn,
        'state_size': state_size,
        'action_size': action_size,
        'action_high': action_high,
        'action_low': action_low,
        'discrete_action': discrete_action,
        'learning_rate': learning_rate,
        'epsilon': epsilon,
        'batch_size': batch_size,
        'num_batches': num_batches
    }

    return lambda: Agent(kwargs)
