"""
An implementation of the Advantage-SABL learning algorithm.
"""

import tensorflow as tf
import numpy as np


class Sample:
    """
    Represents a state-action feedback sample.  Used to overcome
    issues with numpy converting lists of tuples to 2D arrays.
    """

    def __init__(self, state, action, explicit=True, correct=True):
        """
        Initializes the sample.

        :param state: the state
        :param action: the action taken
        :param explicit: whether the feedback was explicit
        :param correct: whether the feedback was positive or negative
        """

        self.state = state
        self.action = action
        self.explicit = explicit
        self.correct = correct


class TaskAgent:
    """
    An A-SABL agent for a single task.
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

        # Capture the action space
        action_space = kwargs['action_space']

        self._discrete_action = action_space.discrete

        if self._discrete_action:
            self._num_actions = action_space.size
        else:
            self._action_shape = action_space.shape

            low = np.asarray(action_space.low, dtype=np.float32)
            high = np.asarray(action_space.high, dtype=np.float32)

            self._action_scale = high - low
            self._action_offset = low

        # Capture the strategy
        mu_plus = kwargs['mu_plus']
        mu_minus = kwargs['mu_minus']

        self._implicit = (mu_plus != mu_minus)

        # Capture the learning parameters
        self._exploration = kwargs['exploration']
        self._batch_size = kwargs['batch_size']
        self._num_batches = kwargs['num_batches']

        # Construct the model
        with graph.as_default(), tf.variable_scope(None, default_name='task'):
            self._correct_input = tf.placeholder(dtype=tf.bool, shape=[None])
            self._explicit_input = tf.placeholder(dtype=tf.bool, shape=[None])
            self._state_input = tf.placeholder(dtype=tf.float32, shape=[None] + list(kwargs['state_space'].shape))

            output = kwargs['value_fn'](self._state_input)

            # Define loss and action output
            if self._discrete_action:

                # Action input
                self._action_input = tf.placeholder(dtype=tf.int32, shape=[None])

                # Action loss
                one_hot = tf.one_hot(self._action_input, self._action_space.size)
                value = tf.reduce_sum(one_hot * output, axis=1)

                advantage = value - tf.reduce_mean(value, axis=1)

                # Action output
                self._action = tf.argmax(output, axis=1)
            else:

                # Action input
                self._action_input = tf.placeholder(dtype=tf.float32, shape=[None] + list(self._action_shape))

                # Action loss
                advantage = kwargs['advantage_fn'](self._state_input)

                action_center = advantage[:, 0]
                action_weight = advantage[:, 1]

                action_axes = list(range(1, len(self._action_shape) + 1))

                regret = tf.exp(action_weight) * tf.square(action_center - self._action_input)
                value = output[:, 0] - tf.reduce_sum(regret, axis=action_axes)

                advantage = prediction - (kwargs['action_error'] * tf.reduce_sum(action_weight))

                # Action output
                self._action = action_center

            # A-SABL loss
            epsilon = kwargs['epsilon']
            probability = epsilon + (1.0 - (2.0 * epsilon / (1.0 + tf.exp(-kwargs['alpha'] * advantage))))

            positive = (1.0 - mu_plus) * probability
            negative = (1.0 - mu_minus) * (1.0 - probability)
            neutral = (mu_plus * probability) + (mu_minus * (1.0 - probability))

            loss = tf.reduce_mean(tf.log(tf.where(self._explicit_input,
                                                  tf.where(self._correct_input, positive, negative), neutral)))

            # Define policy update
            self._update = tf.train.AdamOptimizer(learning_rate=kwargs['learning_rate']).minimize(loss)

            # Initialize model
            session.run(tf.variables_initializer(tf.global_variables(scope=tf.get_variable_scope().name)))

        # Initialize data set
        self._samples = []

    def feedback(self, state, action, feedback):
        """
        Adds a new state-action pair, with associated feedback, to the data set.

        :param state: the state
        :param action: the action taken in that state
        :param feedback: the feedback for this action
        """

        if 0.0 < feedback:
            self._samples.append(Sample(state, action, correct=True))
        elif 0.0 > feedback:
            self._samples.append(Sample(state, action, correct=False))
        elif self._implicit:
            self._samples.append(Sample(state, action, explicit=False))

    def act(self, state, evaluation):
        """
        Samples an action from the agent's policy for the given state.

        :param state: the current state
        :param evaluation: whether this is an evaluation action
        :return: the sampled action
        """

        if not evaluation and np.random.rand() <= self._epsilon:
            if self._discrete_action:
                return np.random.randint(0, self._num_actions)
            else:
                return (np.random.random_sample(self._action_shape) * self._action_scale) + self._action_offset
        else:
            return self._session.run(self._action, feed_dict={self._state_input: [state]})[0]

    def update(self):
        """
        Updates the agent's policy based on the available data
        """

        for _ in range(self._num_batches):
            batch = np.random.choice(self._samples, size=self._batch_size)
            states = []
            actions = []
            explicit = []
            correct = []

            for sample in batch:
                states.append(sample.state)
                actions.append(sample.action)
                explicit.append(sample.explicit)
                correct.append(sample.correct)

            self._session.run(self._update, feed_dict={
                self._state_input: states,
                self._action_input: actions,
                self._explicit_input: explicit,
                self._correct_input: correct
            })


class Agent:
    """
    A multi-task TAMER agent.  Does not implement an
    interface for learning from interaction with the environment.
    """

    def __init__(self, graph, session, **kwargs):
        """
        Initializes the agent.  Just initializes the dictionary of
        task models and the Tensorflow graph and session.

        :param kwargs: the configuration parameters for the agent.
        """

        # Capture configuration
        self._kwargs = kwargs

        # Create graph and session
        self._graph = graph
        self._session = session

        # Create task dictionary and define current task
        self._tasks = dict()
        self._current = None

    def reset(self, task=None):
        """
        Indicates to the agent that a new episode has started, that is
        the current state was sampled independently of the previous state.
        May also change the current task

        :param task: the name of the task for the new episode
        """

        if self._current is not None:
            self._current.update()

        if task is not None:
            if task not in self._tasks:
                self._tasks[task] = TaskAgent(self._graph, self._session, self._kwargs)

            self._current = self._tasks[task]

    def feedback(self, state, action, feedback):
        """
        Provides feedback for an action in a given state

        :param state: the current state
        :param action:  the agent's action
        :param feedback: the teacher's feedback signal
        """

        if self._current is not None and 0.0 != feedback:
            self._current.feedback(state, action, feedback)

    def act(self, state, evaluation=False):
        """
        Samples an action from the agent's policy for the current task.

        :param state: the current state
        :param evaluation: whether this is an evaluation action (and so should not be exploratory)
        :return: the sampled action
        """

        if self._current is not None:
            return self._current.act(state, evaluation)
        else:
            return None


def manager(value_fn, state_space, action_space,
            advantage_fn=None,
            exploration=0.15,
            action_error=0.1,
            alpha=1.0,
            epsilon=0.1,
            mu_plus=0.9,
            mu_minus=0.9,
            learning_rate=0.01,
            batch_size=10,
            num_batches=100):
    """
    Returns a context manager which is used to instantiate and clean up TAMER
    agent with the provided configuration.

    :param value_fn: the function used to build the state-value function graph
    :param state_space: the state space
    :param action_space: the action space
    :param advantage_fn: the function used to build the value estimator graph (only for continuous actions)
    :param exploration: the rate at which the agent selects suboptimal actions when exploring
    :param action_error: the assumed noise in the baseline policy (only for continuous actions)
    :param alpha: the scale factor determining how the feedback probability depends on the advantage
    :param epsilon: the assumed error rate in providing feedback
    :param mu_plus: the probability of providing no feedback for a correct action
    :param mu_minus: the probability of providing no feedback for an incorrect action.
    :param learning_rate: the learning rate for the estimator update
    :param batch_size: the batch size used for each update
    :param num_batches: the number of batch updates per episode
    :return: a context manager which creates a new TAMER agent
    """

    class Manager:

        def __enter__(self):
            self._graph = tf.Graph()
            self._session = tf.Session(graph=self._graph)

            return Agent(self._graph, self._session,
                         value_fn=value_fn,
                         advantage_fn=advantage_fn,
                         state_space=state_space,
                         action_space=action_space,
                         exploration=exploration,
                         action_error=action_error,
                         alpha=alpha,
                         epsilon=epsilon,
                         mu_plus=mu_plus,
                         mu_minus=mu_minus,
                         learning_rate=learning_rate,
                         batch_size=batch_size,
                         num_batches=num_batches)

        def __exit__(self, exc_type, exc_val, exc_tb):
            """
            Closes the session associated with the current agent.

            :param exc_type: ignored
            :param exc_val: ignored
            :param exc_tb: ignored
            :return: always False, never suppress exceptions
            """

            self._session.close()
            self._session = None
            self._graph = None

            return False

    return Manager()
