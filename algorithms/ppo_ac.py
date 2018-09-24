"""
Defines a reinforcement learning agent which uses a
simple version of proximal policy optimization, without an
actor-critic style baseline value function.
"""

import tensorflow as tf
import numpy as np
import gym
import roboschool


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
                target = tf.reduce_sum(one_hot * target, 1) / tf.reduce_sum(target, 1)

                hypothesis = tf.exp(hypothesis_actor)
                hypothesis = tf.reduce_sum(one_hot * hypothesis, 1) / tf.reduce_sum(hypothesis, 1)

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

    def __enter__(self):
        """
        Defines the TensorFlow session that this agent will
        use, and initializes the model parameters.

        :return: the agent itself
        """
        self._session = tf.Session(graph=self._graph)
        self._session.run(self._initialize)
        self._session.run(self._transfer_critic)
        self._session.run(self._transfer_actor)
        self.reset()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Releases the agent's TensorFlow session

        :param exc_type: ignored
        :param exc_val: ignored
        :param exc_tb: ignored
        :return: always false, do not suppress exceptions
        """

        self._session.close()

    def _update_critic_full(self):
        """
        Updates the critic to estimate the value of the current policy.  Only updates the target network
        once, that is, it only does one round of value iteration, but multiple gradient steps.  Uses all the
        data at each gradient update.

        The way the critic update is currently defined, the target network seems to be unnecessary.
        """

        states = []
        estimates = []

        for trajectory in self._trajectories:
            critic = self._session.run(self._critic, feed_dict={self._state_input: trajectory.states})
            values = np.empty([len(trajectory)])

            value = trajectory.rewards[-1]
            values[-1] = value

            for t in reversed(range(len(trajectory) - 1)):
                value *= self._discount * self._mixing
                value += trajectory.rewards[t] + ((1.0 - self._mixing) * self._discount * critic[t + 1, 0])
                values[t] = value

            states.extend(trajectory.states)
            estimates.extend(values)

        for _ in range(self._num_batches):
            self._session.run(self._critic_update, feed_dict={
                self._state_input: states,
                self._value_input: values
            })

        self._session.run(self._transfer_critic)

    def _update_critic_batch(self):
        """
        Updates the critic to estimate the value of the current policy.  Only updates the target network
        once, that is, it only does one round of value iteration, but multiple gradient steps.  Uses a random
        sample of states for each critic update

        The way the critic update is currently defined, the target network seems to be unnecessary.
        """

        samples = []

        for trajectory in self._trajectories:
            critic = self._session.run(self._critic, feed_dict={self._state_input: trajectory.states})

            value = trajectory.rewards[-1]
            samples.append(Sample(trajectory.states[-1], trajectory.actions[-1], value))

            for t in reversed(range(len(trajectory) - 1)):
                value *= self._discount * self._mixing
                value += trajectory.rewards[t] + ((1.0 - self._mixing) * self._discount * critic[t + 1, 0])
                samples.append(Sample(trajectory.states[t], trajectory.actions[t], value))

        for _ in range(self._num_batches):
            batch = np.random.choice(samples, self._batch_size, replace=False)
            states = []
            values = []

            for sample in batch:
                states.append(sample.state)
                values.append(sample.value)

                self._session.run(self._critic_update, feed_dict={
                    self._state_input: states,
                    self._value_input: values
                })

        self._session.run(self._transfer_critic)

    def _update(self):
        """
        Updates the agent's policy based on recent experience.

        Right now we pass every state through the critic, but this may
        end up being too computationally expensive.
        """

        self._update_critic_full()

        # Compute advantages
        samples = []

        for trajectory in self._trajectories:
            critic = self._session.run(self._critic, feed_dict={self._state_input: trajectory.states})

            value = trajectory.rewards[-1]
            samples.append(Sample(trajectory.states[-1], trajectory.actions[-1], value))

            for t in reversed(range(len(trajectory) - 1)):
                value *= self._discount * self._mixing
                value += trajectory.rewards[t] + (self._discount * critic[t + 1, 0]) - critic[t, 0]
                samples.append(Sample(trajectory.states[t], trajectory.actions[t], value))

        # Update actor
        for _ in range(self._num_batches):

            # Construct batch
            batch = np.random.choice(samples, self._batch_size, replace=False)
            states = []
            actions = []
            advantages = []

            for sample in batch:
                states.append(sample.state)
                actions.append(sample.action)
                advantages.append(sample.value)

            # Run update
            self._session.run(self._actor_update, feed_dict={
                self._state_input: states,
                self._action_input: actions,
                self._advantage_input: advantages
            })

        # Transfer parameters
        self._session.run(self._transfer_actor)

        # Clear experience data
        self._trajectories = []

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

    def reward(self, reward):
        """
        Gives an immediate reward to the agent for the most recent step.

        :param reward: the reward value
        """

        self._trajectory.reward(reward)

    def close(self):
        """
        Closes the Tensorflow session associated with this agent.
        """

        self._session.close()


def build(actor_fn, critic_fn, state_space, action_space,
          discount=0.99,
          mixing=0.9,
          learning_rate=0.0005,
          clip_epsilon=0.05,
          batch_size=50,
          num_batches=20,
          num_episodes=10):
    """
    Builds a new PPO reinforcement learning agent.  We may want to get
    rid of the keyword arguments dictionary altogether.

    :param actor_fn: the function used to build the actor graphs
    :param critic_fn: the function used to build the critic graphs
    :param state_space: the state space
    :param action_space: the action space
    :param discount: the discount factor of the MDP
    :param mixing: the mixing factor for the advantage estimators
    :param learning_rate: the learning rate used for training the policies
    :param clip_epsilon: the clipping radius for the policy ratio
    :param batch_size: the batch size used for training the policies
    :param num_batches: the number of gradient steps to do per update
    :param num_episodes: the number of episodes performed between updates
    :return: a new PPO reinforcement learning agent
    """

    return Agent(actor_fn=actor_fn,
                 critic_fn=critic_fn,
                 state_space=state_space,
                 action_space=action_space,
                 discount=discount,
                 mixing=mixing,
                 learning_rate=learning_rate,
                 clip_epsilon=clip_epsilon,
                 batch_size=batch_size,
                 num_batches=num_batches,
                 num_episodes=num_episodes)
