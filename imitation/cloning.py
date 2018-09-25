"""
An implementation of behavioral cloning for learning from demonstration.
"""

import tensorflow as tf
import numpy as np


class TaskAgent:
    """
    A behavioral cloning agent for a single task.
    """

    def __init__(self, data, graph, session, kwargs):
        """
        Initializes the agent.

        :param data: the demonstrated state-action pairs for the task
        :param graph: the Tensorflow graph to be used
        :param session: the Tensorflow session to be used
        :param kwargs: the configuration session for the agent
        """

        # Capture the state and action spaces
        state_space = kwargs['state_space']
        action_space = kwargs['action_space']

        # Capture the learning parameters
        self._batch_size = kwargs['batch_size']
        self._num_batches = kwargs['num_batches']

        # Capture instance variables
        self._session = session
        self._discrete_action = action_space.discrete

        # Construct the model
        with graph.as_default(), tf.variable_scope(None, default_name='task'):
            self._state_input = tf.placeholder(dtype=tf.float32, shape=[None] + list(state_space.shape))
            output = kwargs['model_fn'](self._state_input)

            # Define loss and action output
            if self._discrete_action:
                action_input = tf.placeholder(dtype=tf.int32, shape=[None])

                one_hot = tf.one_hot(action_input, action_space.size)
                exp = tf.exp(output)

                loss = tf.reduce_mean(tf.log(tf.reduce_sum(exp, axis=1)) - (one_hot * output))
                self._action = tf.multinomial(output, 1)
            else:
                action_input = tf.placeholder(dtype=tf.float32, shape=[None] + list(action_space.shape))

                # This loss is wrong

                action_mean = output[:, 0]
                action_deviation = output[:, 1]

                loss = tf.square((action_input - action_mean) / tf.exp(action_deviation))
                loss = tf.reduce_mean(tf.multiply(loss, 0.5) + action_deviation)

                noise = tf.random_normal(tf.shape(action_mean))
                self._action = action_mean + (noise * tf.exp(action_deviation))

            # Define policy update
            update = tf.train.AdamOptimizer(learning_rate=kwargs['learning_rate']).minimize(loss)

            # Initialize the model
            self._session.run(tf.variables_initializer(tf.global_variables(scope=tf.get_variable_scope().name)))

            # train the model
            for b in range(kwargs['num_batches']):
                print("batch " + str(b))

                batch = np.random.choice(data, size=kwargs['batch_size'])
                states = []
                actions = []

                for sample in batch:
                    states.append(sample.state)
                    actions.append(sample.action)

                self._session.run(update, feed_dict={
                    self._state_input: states,
                    action_input: actions
                })

    def act(self, state):
        """
        Samples an action from the agent's policy for the given state.

        :param state: the current state
        :return: the sampled action
        """

        action = self._session.run(self._action, feed_dict={self._state_input: [state]})
        action = action[0, 0] if self._discrete_action else action[0]

        return action


class Agent:
    """
    A multi-task behavioral cloning agent.  This does not
    implement an interface for learning from interaction
    with the environment, but simply runs supervised on
    a set of demonstrated actions.
    """

    def __init__(self, graph, session, **kwargs):
        """
        Constructs the agent, but does not define the individual task
        agents, as we don't know what the tasks will be yet

        :param graph: the TensorFlow graph for the agent to use
        :param session: the TensorFlow session for the agent to use
        :param kwargs: the configuration parameters for the agent.
        """

        # Capture graph, session, and configuration
        self._kwargs = kwargs
        self._graph = graph
        self._session = session

        # Create task dictionary and define current task
        self._tasks = dict()
        self._current = None

    def demonstrate(self, data):
        """
        Initializes the agent with a set of demonstrations.

        :param data: the demonstration data set
        """

        for task in data.tasks():
            self._tasks[task] = TaskAgent(data.steps(task), self._graph, self._session, self._kwargs)

    def reset(self, task=None):
        """
        Indicates to the agent that a new episode has started, that is
        the current state was sampled independently of the previous
        state.  Also allows for the current task to be set.

        :param task: the name of the current task, if set, will change the task the agent is executing
        """

        if task is not None:
            self._current = self._tasks[task]

    def act(self, state, evaluation=False):
        """
        Samples an action from the agent's policy for the current
        task, and records the state-action pair.

        :param state: the current state
        :param evaluation: behavioral cloning ignores this argument, and never records its own actions
        :return: the sampled action
        """

        return self._current.act(state)


def manager(model_fn, state_space, action_space,
            learning_rate=0.01,
            batch_size=10,
            num_batches=1000):
    """
    Returns a context manager which is used to instantiate and clean up
    a cloning agent with the provided configuration.

    :param model_fn: the function used to build the model graph
    :param state_space: the state space
    :param action_space: the action space
    :param learning_rate: the learning rate used for training the policies
    :param batch_size: the batch size used for training the policies
    :param num_batches: the number of batches used for training the policies
    :return: a context manager which creates a new cloning agent
    """

    class Manager:

        def __enter__(self):
            self._graph = tf.Graph()
            self._session = tf.Session(graph=self._graph)

            return Agent(self._graph, self._session,
                         model_fn=model_fn,
                         state_space=state_space,
                         action_space=action_space,
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
