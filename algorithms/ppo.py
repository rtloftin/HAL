"""
Defines a reinforcement learning agent which uses
proximal policy optimization.
"""

import tensorflow as tf
import numpy as np
import gym
import roboschool


class Sample:
    """
    Represents a state-action sample.
    """

    def __init__(self, state, action, value=0.0):
        """
        Initializes the sample.

        :param state: the current state
        :param action: the action taken
        """

        self.state = state
        self.action = action
        self.value = value


class Trajectory:
    """
    Represents a state action trajectory
    """

    def __init__(self):
        self._steps = []
        self._current = None

    def __len__(self):
        return len(self._steps)

    def step(self, state, action):
        """
        Adds a new step to the trajectory.

        :param state: the state
        :param action: the action
        """

        self._current = Sample(state, action)
        self._steps.append(self._current)

    def reward(self, reward):
        """
        Adds to the reward value of the most recent step.

        :param reward: the immediate reward
        """

        if self._current is not None:
            self._current.value += reward

    def accumulate(self, discount):
        """
        Calculates the advantage value for each time step.

        :param discount: the discount factor
        :return: a list of tuples of (state, action, advantage)
        """

        samples = []
        advantage = 0

        for step in reversed(self._steps):
            advantage = step.value + (discount * advantage)
            samples.append(Sample(step.state, step.action, value=advantage))

        return samples


class Agent:
    """
    An online RL agent which updates its policy
    using a version of the PPO algorithm.
    """

    def __init__(self, **kwargs):
        """
        Initializes a new PPO agent.

        :param kwargs: the configuration options for the agent
        """

        # Get the state and action spaces
        state_space = kwargs['state_space']
        action_space = kwargs['action_space']

        # Capture the configuration parameters needed
        self._discrete_action = action_space.discrete
        self._discount = kwargs['discount']
        self._batch_size = kwargs['batch_size']
        self._num_batches = kwargs['num_batches']
        self._num_episodes = kwargs['num_episodes']

        # Create graph and session
        self._graph = tf.Graph()
        self._session = tf.Session(graph=self._graph)

        # Build the policy network and learning update graph
        with self._graph.as_default():
            self._state_input = tf.placeholder(dtype=tf.float32, shape=[None] + list(state_space.shape))

            with tf.variable_scope("policy"):
                policy_output = kwargs['model_fn'](self._state_input)

            with tf.variable_scope("hypothesis"):
                hypothesis_output = kwargs['model_fn'](self._state_input)

            if action_space.discrete:
                self._action_input = tf.placeholder(dtype=tf.int32, shape=[None])

                one_hot = tf.one_hot(self._action_input, action_space.size)

                policy = tf.exp(policy_output)
                policy = tf.reduce_sum(one_hot * policy, 1) / tf.reduce_sum(policy, 1)

                hypothesis = tf.exp(hypothesis_output)
                hypothesis = tf.reduce_sum(one_hot * hypothesis, 1) / tf.reduce_sum(hypothesis, 1)

                ratio = hypothesis / tf.stop_gradient(policy)

                self._action_output = tf.multinomial(policy_output, 1)
            else:
                self._action_input = tf.placeholder(dtype=tf.float32, shape=[None] + list(action_space.shape))

                policy_mean = policy_output[:, 0]
                policy_deviation = policy_output[:, 1]

                hypothesis_mean = hypothesis_output[:, 0]
                hypothesis_deviation = hypothesis_output[:, 1]

                policy = tf.square(self._action_input - policy_mean) / tf.exp(policy_deviation)
                policy = tf.reduce_sum(policy + policy_deviation, axis=1)

                hypothesis = tf.square(self._action_input - hypothesis_mean) / tf.exp(hypothesis_deviation)
                hypothesis = tf.reduce_sum(hypothesis + hypothesis_deviation, axis=1)

                ratio = tf.exp(tf.multiply(tf.stop_gradient(policy) - hypothesis, 0.5))

                noise = tf.random_normal(tf.shape(policy_mean))
                self._action = policy_mean + (noise * tf.exp(tf.multiply(policy_deviation, 0.5)))

            self._advantage_input = tf.placeholder(dtype=tf.float32, shape=[None])

            clipped_ratio = tf.clip_by_value(ratio, 1.0 - kwargs['clip_epsilon'], 1.0 + kwargs['clip_epsilon'])
            loss = -tf.reduce_mean(tf.minimum(ratio * self._advantage_input, clipped_ratio * self._advantage_input))

            self._update_hypothesis = tf.train.AdamOptimizer(learning_rate=kwargs['learning_rate']).minimize(loss)

            policy_variables = tf.trainable_variables(scope='policy')
            policy_variables = dict(map(lambda x: (x.name[len('policy'):], x), policy_variables))

            hypothesis_variables = tf.trainable_variables(scope='hypothesis')
            hypothesis_variables = dict(map(lambda x: (x.name[len('hypothesis'):], x), hypothesis_variables))

            self._transfer_hypothesis = []

            for key, var in hypothesis_variables.items():
                self._transfer_hypothesis.append(tf.assign(policy_variables[key], var))

            self._session.run(tf.global_variables_initializer())
            self._session.run(self._transfer_hypothesis)

        # Internal state
        self._data = []
        self._trajectory = None
        self._episode_count = 0

        # Reset the agent so it treats the next step as an initial state
        self.reset()

    def _update(self):
        """
        Updates the agent's policy based on recent experience.
        """

        # Perform updates
        for _ in range(self._num_batches):

            # Construct batch
            batch = np.random.choice(self._data, self._batch_size, replace=False)
            states = []
            actions = []
            advantages = []

            for sample in batch:
                states.append(sample.state)
                actions.append(sample.action)
                advantages.append(sample.value)

            # Run update
            self._session.run(self._update_hypothesis, feed_dict={
                self._state_input: states,
                self._action_input: actions,
                self._advantage_input: advantages
            })

        # Transfer parameters
        self._session.run(self._transfer_hypothesis)

        # Clear experience data
        self._data = []

    def reset(self):
        """
        Tells the agent that a new episode has been started, the agent may
        choose to run an update at this time.
        """

        if self._trajectory is not None and len(self._trajectory) != 0:
            self._data.extend(self._trajectory.accumulate(self._discount))
            self._episode_count += 1

        self._trajectory = Trajectory()

        if self._episode_count == self._num_episodes:
            self._update()
            self._episode_count = 0

    def observe(self, state, action, reward):
        """
        Adds a step to the agent's experience

        :param state: the current state
        :param action: the action taken
        :param reward: the reward signal for the current time step
        """

        self._trajectory.append(state, action, reward)

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

    def reward(self, reward):
        """
        Gives an immediate reward to the agent for the most recent step.

        :param reward: the reward value
        """

        self._trajectory.reward(reward)


def build(model_fn, state_space, action_space,
          discount=0.99,
          learning_rate=0.0005,
          clip_epsilon=0.05,
          batch_size=50,
          num_batches=20,
          num_episodes=10):
    """
    Builds a new PPO reinforcement learning agent.  We may want to get
    rid of the keyword arguments dictionary altogether.

    :param model_fn: the function used to build the model graph
    :param state_space: the number of state features
    :param action_space: the number of actions or action features
    :param discount: the discount factor used to estimate advantages
    :param learning_rate: the learning rate used for training the policies
    :param clip_epsilon: the clipping radius for the policy ratio
    :param batch_size: the batch size used for training the policies
    :param num_batches: the number of gradient steps to do per update
    :param num_episodes: the number of episodes performed between updates
    :return: a new PPO reinforcement learning agent
    """

    return Agent(model_fn=model_fn,
                 state_space=state_space,
                 action_space=action_space,
                 discount=discount,
                 learning_rate=learning_rate,
                 clip_epsilon=clip_epsilon,
                 batch_size=batch_size,
                 num_batches=num_batches,
                 num_episodes=num_episodes)
