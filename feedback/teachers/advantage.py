"""
Defines a teacher that provides feedback equal to the advantage
of the agent's action relative to some baseline policy.  Uses a
rough approximation of the advantage for continuous actions. The
teacher learns a state-action value function for each task using
a set of simulated roll-outs.
"""

import tensorflow as tf
import numpy as np


class Sample:
    """
    Represents an arbitrary state-action feedback sample.  Used to overcome
    issues with numpy converting lists of tuples to 2D arrays.
    """

    def __init__(self, state, action, value):
        """
        Initializes the sample.

        :param state: the state
        :param action: the action taken
        :param value: the value of the sample
        """

        self.state = state
        self.action = action
        self.value = value


class TaskTeacher:
    """
    A pre-trained teacher for a single task.
    """

    def __init__(self, graph, session, env, task, data, kwargs):
        """
        Initializes the task-specific teacher.

        :param graph: the TensorFlow graph to use
        :param session: the TensorFlow session to use
        :param env: the environment being used
        :param task: the task being taught
        :param data: the demonstrated expert trajectories for this task
        :param kwargs: the configuration parameters
        """

        # Capture the session
        self._session = session

        # Build the state-action value graph
        with graph.as_default(), tf.variable_scope(task):
            self._state_input = tf.placeholder(dtype=tf.float32, shape=[None] + list(env.state_space.shape))
            value_input = tf.placeholder(dtype=tf.float32, shape=[None])

            value = kwargs['value_fn'](self._state_input)[:, 0]

            if env.action_space.discrete:

                # Action input
                self._action_input = tf.placeholder(dtype=tf.int32, shape=[None])

                # Prediction loss
                one_hot = tf.one_hot(self._action_input, self._action_space.size)
                prediction = tf.reduce_sum(one_hot * value, axis=1)

                loss = tf.reduce_mean(tf.square(prediction - value_input))

                # Advantage output
                self._advantage = prediction - tf.reduce_mean(value, axis=1)

            else:

                # Action input
                self._action_input = tf.placeholder(dtype=tf.float32, shape=[None] + list(self._action_shape))

                # Prediction loss
                advantage = kwargs['advantage_fn'](self._state_input)

                action_center = advantage[:, 0]
                action_weight = tf.exp(advantage[:, 1])

                action_axes = list(range(1, len(self._action_shape) + 1))

                regret = action_weight * tf.square(action_center - self._action_input)
                prediction = value - tf.reduce_sum(regret, axis=action_axes)

                loss = tf.reduce_mean(tf.square(self._feedback_input - prediction))

                # Advantage output
                self._advantage = prediction - (kwargs['action_error'] * tf.reduce_sum(action_weight))

            update = tf.train.AdamOptimizer(learning_rate=kwargs['learning_rate']).minimize(loss)

            # Initialize model
            session.run(tf.variables_initializer(tf.global_variables(scope=tf.get_variable_scope().name)))

        # Train the model
        discount = kwargs['discount']
        samples = []

        for trajectory in data:
            value = 0.0

            for step in reversed(trajectory):
                value = step.reward + (discount * value)
                samples.append(Sample(step.state, step.action, value))

        for _ in kwargs['num_batches']:
            batch = np.random.choice(samples, kwargs['batch_size'], replace=True)
            states = []
            actions = []
            values = []

            for sample in batch:
                states.append(sample.state)
                actions.append(sample.action)
                values.append(sample.value)

            session.run(update, feed_dict={
                self._state_input: states,
                self._action_input: actions,
                value_input: values
            })

    def feedback(self, state, action):
        """
        Gets feedback for the given action.  Feedback depends on
        the current state of the environment, and on the current
        task for the environment.

        :param state the current state
        :param action: the agent's action
        :return: the real-valued feedback signal
        """

        return self._session.run(self._advantage, feed_dict={
            self._state_input: [state],
            self._action_input: [action]
        })[0]


class Teacher:
    """
    A multi-task teacher which gives feedback equal to the
    estimated advantage of the agent's action.  Uses a learned
    state action value function to generate feedback for each task.
    """

    def __init__(self, graph, session, env, data, **kwargs):
        """
        Initializes the teacher's advantage models.

        :param graph: the TensorFlow graph to use
        :param session: the TensorFlow session to use
        :param env: the environment the teacher is training in
        :param data: a set of expert demonstrations to use
        :param kwargs: the configuration parameters
        """

        self._env = env
        self._tasks = dict()

        for task in env.tasks:
            self._tasks[task] = TaskTeacher(graph, session, env, task, data, kwargs)

    def feedback(self, action):
        """
        Gets feedback for the given action.  Feedback depends on
        the current state of the environment, and on the current
        task for the environment.

        :param action: the agent's action
        :return: the real-valued feedback signal
        """

        return self._tasks[self._env.task].feedback(self._env.state, action)


def builder(value_fn,
            advantage_fn=None,
            action_error=0.1,
            discount=0.99,
            learning_rate=0.001,
            batch_size=128,
            num_batches=1000):
    """
    Returns a builder which itself returns a context manager which
    constructs an advantage teacher with the given configuration.

    :param value_fn: the function used to build the value estimator graph
    :param advantage_fn: the function used to build the value estimator graph (only for continuous actions)
    :param action_error: the assumed noise in the baseline policy (only for continuous actions)
    :param discount: the discount factor
    :param learning_rate: the learning rate for training the estimators
    :param batch_size: the batch size for training the estimators
    :param num_batches: the number of batches for training the estimators
    :return: a new builder for advantage teachers.
    """

    def manager(env, data):

        class Manager:
            def __enter__(self):
                self._graph = tf.Graph()
                self._session = tf.Session(graph=self._graph)

                try:
                    teacher = Teacher(self._graph, self._session, env, data,
                                      value_fn=value_fn,
                                      advantage_fn=advantage_fn,
                                      action_error=action_error,
                                      discount=discount,
                                      learning_rate=learning_rate,
                                      batch_size=batch_size,
                                      num_batches=num_batches)
                except Exception as e:
                    self._session.close()
                    raise e

                return teacher

            def __exit__(self, exc_type, exc_val, exc_tb):
                self._session.close()
                self._graph = None

                return False

        return Manager()

    return manager
