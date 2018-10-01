"""
Defines an imitation learning agent which uses the GAIL
algorithm, with policy updates based on PPO (as opposed to TRPO).
"""

import tensorflow as tf
import numpy as np


class Sample:
    """
    Represents an arbitrary state-action sample.  Mainly to overcome the
    issue with Numpy treating arrays of tuples as two-dimensional arrays.
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


class Trajectory:
    """
    Represents a state action trajectory
    """

    def __init__(self):
        self._states = []
        self._actions = []

    def __len__(self):
        return len(self._states)

    def step(self, state, action):
        """
        Adds a new step to the trajectory.

        :param state: the state
        :param action: the action
        """

        self.states.append(state)
        self.actions.append(action)

    @property
    def states(self):
        return self._states

    @property
    def actions(self):
        return self._actions


class TaskAgent:
    """
    A GAIL agent for a single task
    """

    def __init__(self, data, graph, session, kwargs):
        """
        Initializes the agent.

        :param data: the demonstrated state-action pairs for this task
        :param graph: the Tensorflow graph to be used
        :param session: the Tensorflow session to be used
        :param kwargs: the configuration session for the agent
        """

        # Get the state and action spaces
        state_space = kwargs['state_space']
        action_space = kwargs['action_space']

        # Capture the configuration parameters needed
        self._discrete_action = action_space.discrete
        self._discount = kwargs['discount']
        self._mixing = kwargs['mixing']
        self._batch_size = kwargs['batch_size']
        self._num_batches = kwargs['num_batches']
        self._num_episodes = kwargs['num_episodes']

        # Capture instance variables
        self._session = session
        self._data = data

        # Build the policy network and learning update graph
        with graph.as_default(), tf.variable_scope(None, default_name='task'):

            # Define common state input
            self._state_input = tf.placeholder(dtype=tf.float32, shape=[None] + list(state_space.shape))

            # Define the discriminator, and the target and hypothesis models for the actor and critic
            # Define critic
            with tf.variable_scope("critic"):
                self._critic = kwargs['critic_fn'](self._state_input)[:, 0]

            # Define target actor
            with tf.variable_scope("target_actor"):
                target_actor = kwargs['actor_fn'](self._state_input)
                target_actor_vars = tf.trainable_variables(scope=tf.get_variable_scope().name)

            # Define hypothesis actor
            with tf.variable_scope("hypothesis_actor"):
                hypothesis_actor = kwargs['actor_fn'](self._state_input)
                hypothesis_actor_vars = tf.trainable_variables(scope=tf.get_variable_scope().name)

            # Define actor parameter transfer op
            target_actor_vars = dict(map(lambda x: (x.name[len('target_actor'):], x), target_actor_vars))
            hypothesis_actor_vars = dict(map(lambda x: (x.name[len('hypothesis_actor'):], x), hypothesis_actor_vars))

            self._transfer_actor = []

            for key, var in hypothesis_actor_vars.items():
                self._transfer_actor.append(tf.assign(target_actor_vars[key], var))

            # Define action input and policy ratios, and discriminator
            if action_space.discrete:

                # Action input
                self._action_input = tf.placeholder(dtype=tf.int32, shape=[None])
                one_hot = tf.one_hot(self._action_input, action_space.size)

                # Discriminator
                with tf.variable_scope("discriminator"):
                    discriminator = kwargs['cost_fn'](self._state_input)

                discriminator = tf.reduce_sum(one_hot * discriminator, axis=1)

                # Probability ratios
                target = tf.exp(target_actor)
                target = tf.reduce_sum(one_hot * target, axis=1) / tf.reduce_sum(target, axis=1)

                hypothesis = tf.exp(hypothesis_actor)
                hypothesis = tf.reduce_sum(one_hot * hypothesis, axis=1) / tf.reduce_sum(hypothesis, axis=1)

                ratio = hypothesis / tf.stop_gradient(target)

                # Action output
                self._action = tf.multinomial(policy_output, 1)
            else:

                # Action input
                self._action_input = tf.placeholder(dtype=tf.float32, shape=[None] + list(action_space.shape))

                # Discriminator
                with tf.variable_scope("discriminator"):
                    discriminator = kwargs['cost_fn'](self._state_input)[:, 0]

                # Probability ratios
                target_mean = target_actor[:, 0]
                target_deviation = target_actor[:, 1]

                hypothesis_mean = hypothesis_actor[:, 0]
                hypothesis_deviation = hypothesis_actor[:, 1]

                target = tf.square((self._action_input - target_mean) / tf.exp(target_deviation))
                target = tf.reduce_sum((0.5 * target) + target_deviation, axis=1)

                hypothesis = tf.square((self._action_input - hypothesis_mean) / tf.exp(hypothesis_deviation))
                hypothesis = tf.reduce_sum((0.5 * hypothesis) + hypothesis_deviation, axis=1)

                ratio = tf.exp(tf.stop_gradient(target) - hypothesis)

                # Action output
                noise = tf.random_normal(tf.shape(target_mean))
                self._action = target_mean + (noise * tf.exp(target_deviation))

            # Discriminator update
            self._expert_input = tf.placeholder(dtype=tf.float32, shape=[None])
            self._cost = -tf.log(1.0 + tf.exp(-discriminator))

            loss = self._expert_input * tf.log(1.0 + tf.exp(discriminator)) - (1.0 - self._expert_input) * self._cost
            loss = tf.reduce_sum(loss)

            self._discriminator_update = tf.train.AdamOptimizer(learning_rate=kwargs['learning_rate']).minimize(loss)

            # Critic update
            self._value_input = tf.placeholder(dtype=tf.float32, shape=[None])

            loss = tf.reduce_mean(tf.square(self._value_input - self._critic))
            self._critic_update = tf.train.AdamOptimizer(learning_rate=kwargs['learning_rate']).minimize(loss)

            # Actor update
            self._advantage_input = tf.placeholder(dtype=tf.float32, shape=[None])

            clipped_ratio = tf.clip_by_value(ratio, 1.0 - kwargs['clip_epsilon'], 1.0 + kwargs['clip_epsilon'])
            loss = -tf.reduce_mean(tf.minimum(ratio * self._advantage_input, clipped_ratio * self._advantage_input))

            self._actor_update = tf.train.AdamOptimizer(learning_rate=kwargs['learning_rate']).minimize(loss)

            # Variable assertion
            self._is_finite = []
            self._is_inf = []
            self._is_nan = []

            for var in tf.trainable_variables(scope=tf.get_variable_scope().name):
                self._is_finite.append(tf.reduce_all(tf.debugging.is_finite(var)))
                self._is_inf.append(tf.reduce_any(tf.debugging.is_inf(var)))
                self._is_nan.append(tf.reduce_any(tf.debugging.is_nan(var)))

            self._is_finite = tf.reduce_all(tf.stack(self._is_finite))
            self._is_inf = tf.reduce_any(tf.stack(self._is_inf))
            self._is_nan = tf.reduce_any(tf.stack(self._is_nan))

            # Initialize the model
            self._session.run(tf.variables_initializer(tf.global_variables(scope=tf.get_variable_scope().name)))

        # Initialize internal state
        self._trajectories = []
        self._trajectory = None
        self._episode_count = 0

    def _update(self):
        """
        Updates the agent's policy based on recent experience.
        """

        # Update discriminator
        samples = []

        for trajectory in self._trajectories:
            for t in range(len(trajectory)):
                samples.append(Sample(trajectory.states[t], trajectory.actions[t], 0.0))

        for _ in range(self._num_batches):
            expert_batch = np.random.choice(self._data, self._batch_size, replace=True)
            agent_batch = np.random.choice(samples, self._batch_size, replace=True)
            states = []
            actions = []
            expert = []

            for sample in expert_batch:
                states.append(sample.state)
                actions.append(sample.action)
                expert.append(1.0)

            for sample in agent_batch:
                states.append(sample.state)
                actions.append(sample.action)
                expert.append(0.0)

            self._session.run(self._discriminator_update, feed_dict={
                self._state_input: states,
                self._action_input: actions,
                self._expert_input: expert
            })

        # Update critic
        samples = []

        for trajectory in self._trajectories:
            values, costs = self._session.run([self._critic, self._cost], feed_dict={
                self._state_input: trajectory.states,
                self._action_input: trajectory.actions
            })
            acc = 0.0

            for t in reversed(range(len(trajectory))):
                value = acc - costs[t]
                acc = self._discount * (((1.0 - self._mixing) * values[t]) + (self._mixing * value))
                samples.append(Sample(trajectory.states[t], trajectory.actions[t], value))

        for _ in range(self._num_batches):
            batch = np.random.choice(samples, self._batch_size, replace=True)
            states = []
            values = []

            for sample in batch:
                states.append(sample.state)
                values.append(sample.value)

            self._session.run(self._critic_update, feed_dict={
                self._state_input: states,
                self._value_input: values
            })

        # Update actor
        samples = []

        for trajectory in self._trajectories:
            values, costs = self._session.run([self._critic, self._cost], feed_dict={
                self._state_input: trajectory.states,
                self._action_input: trajectory.actions
            })
            acc = 0.0

            for t in reversed(range(len(trajectory))):
                value = acc - costs[t]
                acc = self._discount * (((1.0 - self._mixing) * values[t]) + (self._mixing * value))
                samples.append(Sample(trajectory.states[t], trajectory.actions[t], value - values[t]))

        for _ in range(self._num_batches):

            batch = np.random.choice(samples, self._batch_size, replace=True)
            states = []
            actions = []
            advantages = []

            for sample in batch:
                states.append(sample.state)
                actions.append(sample.action)
                advantages.append(sample.value)

            self._session.run(self._actor_update, feed_dict={
                self._state_input: states,
                self._action_input: actions,
                self._advantage_input: advantages
            })

        # Transfer parameters
        self._session.run(self._transfer_actor)

        # Validate parameters
        if not self._session.run(self._is_finite):
            if self._session.run(self._is_inf):
                print("At least one parameter became infinite")
            if self._session.run(self._is_nan):
                print("At least one parameter became NaN")

    def reset(self):
        """
        Tells the agent that a new episode has been started, the agent may
        choose to run an update at this time.
        """

        if self._trajectory is not None and len(self._trajectory) != 0:
            self._trajectories.append(self._trajectory)
            self._episode_count += 1

        if self._episode_count == self._num_episodes:
            self._update()
            self._episode_count = 0

        self._trajectory = Trajectory()

    def act(self, state, evaluation=False):
        """
        Records the current state, and selects the agent's action

        :param state: a representation of the current state
        :param evaluation: if true, this indicates that this action should not be recorded
        :return: a representation of the next action
        """

        action = self._session.run(self._action, feed_dict={self._state_input: [state]})
        action = action[0, 0] if self._discrete_action else action[0]

        if not evaluation:
            self._trajectory.step(state, action)

        return action


class Agent:
    """
    A multi-task GAIL agent.
    """

    def __init__(self, graph, session, **kwargs):
        """
        Initializes the agent.  Just initializes the dictionary of
        task models and the TensorFlow graph and session.

        :param kwargs: the configuration parameters for the agent.
        """

        # Capture graph, session and configuration
        self._graph = graph
        self._session = session
        self._kwargs = kwargs

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

        self._current.reset()

    def act(self, state, evaluation=False):
        """
        Samples an action from the agent's policy for the current task.

        :param state: the current stated
        :param evaluation: if true, the agent will ignore this state
        :return: the sampled action
        """

        return self._current.act(state, evaluation)


def manager(actor_fn, critic_fn, cost_fn, state_space, action_space,
            discount=0.99,
            mixing=0.9,
            learning_rate=0.0005,
            clip_epsilon=0.05,
            batch_size=50,
            num_batches=20,
            num_episodes=10):
    """
    Returns a context manager which is used to instantiate and clean up GAIL
    agent with the provided configuration, one that uses PPO policy updates.

    :param actor_fn: the function used to build the actor graphs
    :param critic_fn: the function used to build the critic graphs
    :param cost_fn: the function used to build the discriminator graphs
    :param state_space: the state space
    :param action_space: the action space
    :param discount: the discount factor of the MDP
    :param mixing: the mixing factor for the advantage estimators
    :param learning_rate: the learning rate used for training the policies
    :param clip_epsilon: the clipping radius for the policy ratio
    :param batch_size: the batch size used for training the policies
    :param num_batches: the number of gradient steps to do per update
    :param num_episodes: the number of episodes performed between updates
    :return: a context manager which creates a new GAIL agent
    """

    class Manager:

        def __enter__(self):
            self._graph = tf.Graph()
            self._session = tf.Session(graph=self._graph)

            return Agent(self._graph, self._session,
                         actor_fn=actor_fn,
                         critic_fn=critic_fn,
                         cost_fn=cost_fn,
                         state_space=state_space,
                         action_space=action_space,
                         discount=discount,
                         mixing=mixing,
                         learning_rate=learning_rate,
                         clip_epsilon=clip_epsilon,
                         batch_size=batch_size,
                         num_batches=num_batches,
                         num_episodes=num_episodes)

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
