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
        self._rewards = []

    def __len__(self):
        return len(self._states)

    def step(self, state, action):
        """
        Adds a new step to the trajectory.

        :param state: the state
        :param action: the action
        """

        self._states.append(state)
        self._actions.append(action)
        self._rewards.append(0.0)

    def reward(self, reward):
        """
        Adds to the reward value of the most recent step.

        :param reward: the immediate reward
        """

        if len(self._rewards) != 0:
            self._rewards[-1] += reward

    @property
    def states(self):
        return self._states

    @property
    def actions(self):
        return self._actions

    @property
    def rewards(self):
        return self._rewards


class TaskAgent:
    """
    A GAIL agent for a single task.
    """

    def __init__(self, graph, session, kwargs):
        """
        Initializes the agent.

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

        # Create graph and session
        self._graph = tf.Graph()
        self._session = None

        # Build the policy network and learning update graph
        with self._graph.as_default():

            # Define common state input
            self._state_input = tf.placeholder(dtype=tf.float32, shape=[None] + list(state_space.shape))

            # Define target and hypothesis models for the actor and critic
            with tf.variable_scope("target_critic"):
                target_critic = kwargs['critic_fn'](self._state_input)

            with tf.variable_scope("hypothesis_critic"):
                hypothesis_critic = kwargs['critic_fn'](self._state_input)

            with tf.variable_scope("target_actor"):
                target_actor = kwargs['actor_fn'](self._state_input)

            with tf.variable_scope("hypothesis_actor"):
                hypothesis_actor = kwargs['actor_fn'](self._state_input)

            # Define parameter transfer operations
            target_critic_vars = tf.trainable_variables(scope='target_critic')
            target_critic_vars = dict(map(lambda x: (x.name[len('target_critic'):], x), target_critic_vars))

            hypothesis_critic_vars = tf.trainable_variables(scope='hypothesis_critic')
            hypothesis_critic_vars = dict(map(lambda x: (x.name[len('hypothesis_critic'):], x), hypothesis_critic_vars))

            target_actor_vars = tf.trainable_variables(scope='target_actor')
            target_actor_vars = dict(map(lambda x: (x.name[len('target_actor'):], x), target_actor_vars))

            hypothesis_actor_vars = tf.trainable_variables(scope='hypothesis_actor')
            hypothesis_actor_vars = dict(map(lambda x: (x.name[len('hypothesis_actor'):], x), hypothesis_actor_vars))

            self._transfer_critic = []
            self._transfer_actor = []

            for key, var in hypothesis_critic_vars.items():
                self._transfer_critic.append(tf.assign(target_critic_vars[key], var))

            for key, var in hypothesis_actor_vars.items():
                self._transfer_actor.append(tf.assign(target_actor_vars[key], var))

            # Define action input and policy ratios
            if action_space.discrete:

                # Action input
                self._action_input = tf.placeholder(dtype=tf.int32, shape=[None])
                one_hot = tf.one_hot(self._action_input, action_space.size)

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

                # Probability ratios
                target_mean = target_actor[:, 0]
                target_deviation = target_actor[:, 1]

                hypothesis_mean = hypothesis_actor[:, 0]
                hypothesis_deviation = hypothesis_actor[:, 1]

                target = tf.square(self._action_input - target_mean) / tf.exp(target_deviation)
                target = tf.reduce_sum(target + target_deviation, axis=1)

                hypothesis = tf.square(self._action_input - hypothesis_mean) / tf.exp(hypothesis_deviation)
                hypothesis = tf.reduce_sum(hypothesis + hypothesis_deviation, axis=1)

                ratio = tf.exp(tf.multiply(tf.stop_gradient(target) - hypothesis, 0.5))

                # Action output
                noise = tf.random_normal(tf.shape(target_mean))
                self._action = target_mean + (noise * tf.exp(tf.multiply(target_deviation, 0.5)))

            # Critic update
            self._critic = target_critic
            self._value_input = tf.placeholder(dtype=tf.float32, shape=[None])

            loss = tf.reduce_mean(tf.square(self._value_input - hypothesis_critic))
            self._critic_update = tf.train.AdamOptimizer(learning_rate=kwargs['learning_rate']).minimize(loss)

            # Actor update
            self._advantage_input = tf.placeholder(dtype=tf.float32, shape=[None])

            clipped_ratio = tf.clip_by_value(ratio, 1.0 - kwargs['clip_epsilon'], 1.0 + kwargs['clip_epsilon'])
            loss = -tf.reduce_mean(tf.minimum(ratio * self._advantage_input, clipped_ratio * self._advantage_input))

            self._actor_update = tf.train.AdamOptimizer(learning_rate=kwargs['learning_rate']).minimize(loss)

            # Initializer
            self._initialize = tf.variables_initializer(tf.global_variables())

        # Initialize internal state
        self._trajectories = []
        self._trajectory = None
        self._episode_count = 0

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
    A multi-task GAIL agent.
    """

    def __init__(self, kwargs):
        """
        Initializes the agent.  Just initializes the dictionary of
        task models and the TensorFlow graph and session.

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

    def incorporate(self):
        """
        This is just a placeholder, GAIL updates its policies online
        """

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
        Samples an action from the agent's policy for the current task.

        :param state: the current stated
        :param evaluation: whether the agent should ignore this state and the action taken
        :return: the sampled action
        """

        return self._current.act(state, evaluation)


def build(actor_fn, critic_fn, cost_fn, state_space, action_space,
          discount=0.99,
          mixing=0.9,
          learning_rate=0.0005,
          clip_epsilon=0.05,
          batch_size=50,
          num_batches=20,
          num_episodes=10):
    """
    Builds a new GAIL imitation learning agent, one which uses
    PPO policy updates, and handles multiple tasks

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
    :return: a new GAIL imitation learning agent
    """

    return Agent(actor_fn=actor_fn,
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
