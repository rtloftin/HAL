"""
An implementation of behavioral cloning for learning from demonstration.
"""

import tensorflow as tf
import numpy as np


class Sample:
    """
    Represents a demonstrated state-action pair.  Used to overcome
    Numpy's issue with randomly sampling from lists of tuples
    """

    def __init__(self, state, action):
        """
        Initializes the sample.

        :param state: the current state
        :param action: the action taken
        """

        self.state = state
        self.action = action


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

        # Capture the state and action spaces
        state_space = kwargs['state_space']
        action_space = kwargs['action_space']

        # Capture the learning parameters
        self._discrete_action = action_space.discrete
        self._batch_size = kwargs['batch_size']
        self._num_batches = kwargs['num_batches']

        # Initialize the data set
        self._data = []

        # Construct the model
        with graph.as_default():
            with tf.variable_scope(None, default_name='task'):
                self._state_input = tf.placeholder(dtype=tf.float32, shape=[None] + list(state_space.shape))
                output = kwargs['model_fn'](self._state_input)

                # Define loss and action output
                if self._discrete_action:
                    self._action_input = tf.placeholder(dtype=tf.int32, shape=[None])

                    one_hot = tf.one_hot(self._action_input, action_space.size)
                    exp = tf.exp(output)

                    loss = tf.reduce_mean(tf.log(tf.reduce_sum(exp, axis=1)) - (one_hot * output))
                    self._action = tf.multinomial(output, 1)
                else:
                    self._action_input = tf.placeholder(dtype=tf.float32, shape=[None] + list(action_space.shape))

                    # This loss is wrong

                    action_mean = output[:, 0]
                    action_deviation = output[:, 1]

                    loss = tf.square((self._action_input - action_mean) / tf.exp(action_deviation))
                    loss = tf.reduce_mean(tf.multiply(loss, 0.5) + action_deviation)

                    noise = tf.random_normal(tf.shape(action_mean))
                    self._action = action_mean + (noise * tf.exp(action_deviation))

                # Define policy update
                self._update = tf.train.AdamOptimizer(learning_rate=kwargs['learning_rate']).minimize(loss)

                # Initialize the model
                self._session.run(tf.variables_initializer(tf.global_variables(scope=tf.get_variable_scope().name)))

    def demonstrate(self, state, action):
        """
        Adds a new state-action pair to the data set.

        :param state: the state
        :param action: the action taken in that state
        """

        self._data.append(Sample(state, action))

    def act(self, state):
        """
        Samples an action from the agent's policy for the given state.

        :param state: the current state
        :return: the sampled action
        """

        action = self._session.run(self._action, feed_dict={self._state_input: [state]})
        action = action[0, 0] if self._discrete_action else action[0]

        return action

    def incorporate(self):
        """
        Updates the agent's policy based on the available data
        """

        for b in range(self._num_batches):
            print("batch " + str(b))

            batch = np.random.choice(self._data, size=self._batch_size)
            states = []
            actions = []

            for sample in batch:
                states.append(sample.state)
                actions.append(sample.action)

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

    def __init__(self, **kwargs):
        """
        Initializes the agent.  Just initializes the dictionary of
        task models and the Tensorflow graph and session.

        :param kwargs: the configuration parameters for the agent.
        """

        # Capture configuration
        self._kwargs = kwargs

        # Create graph and session
        self._graph = tf.Graph()
        self._session = None

        # Create task dictionary and define current task
        self._tasks = dict()
        self._current = None

    def __enter__(self):
        """
        Initializes the TensorFlow session used by this agent.

        :return: the agent itself
        """

        self._session = tf.Session(graph=self._graph)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Closes the TensorFlow session.

        :param exc_type: ignored
        :param exc_val: ignored
        :param exc_tb: ignored
        :return: always False, never suppress exceptions
        """

        self._session.close()

    def set_task(self, name):
        """
        Sets the task that the agent is currently learning to perform.

        :param name: the name of the task
        """

        if name not in self._tasks:
            self._tasks[name] = TaskAgent(self._graph, self._session, self._kwargs)

        self._current = self._tasks[name]

    def incorporate(self):
        """
        Updates the agent's policies to incorporate all of the
        demonstrations that have been provided so far.
        """

        for task in self._tasks.values():
            task.incorporate()

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

    def act(self, state, evaluation=False):
        """
        Samples an action from the agent's policy for the current
        task, and records the state-action pair.

        :param state: the current state
        :param evaluation: behavioral cloning ignores this argument, never records its own actions
        :return: the sampled action
        """

        return self._current.act(state)


def build(model_fn, state_space, action_space,
          learning_rate=0.01,
          batch_size=10,
          num_batches=1000):
    """
    Constructs a new multi-task behavioral cloning agents.

    :param model_fn: the function used to build the model graph
    :param state_space: the state space
    :param action_space: the action space
    :param learning_rate: the learning rate used for training the policies
    :param batch_size: the batch size used for training the policies
    :param num_batches: the number of batches used for training the policies
    :return: a new behavioral cloning agent
    """

    return Agent(model_fn=model_fn,
                 state_space=state_space,
                 action_space=action_space,
                 learning_rate=learning_rate,
                 batch_size=batch_size,
                 num_batches=num_batches)
