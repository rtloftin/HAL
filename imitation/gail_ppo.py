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
        self._discrete_action = action_space.discrete
        self._data = data

        # Build the policy network and learning update graph
        with graph.as_default(), tf.variable_scope(None, default_name='task'):

            # Define common state input
            self._state_input = tf.placeholder(dtype=tf.float32, shape=[None] + list(state_space.shape))

            # Define the discriminator, and the target and hypothesis models for the actor and critic
            with tf.variable_scope("target_actor"):
                target_actor = kwargs['actor_fn'](self._state_input)

            with tf.variable_scope("hypothesis_actor"):
                hypothesis_actor = kwargs['actor_fn'](self._state_input)

            with tf.variable_scope("target_critic"):
                target_critic = kwargs['critic_fn'](self._state_input)

            with tf.variable_scope("hypothesis_critic"):
                hypothesis_critic = kwargs['critic_fn'](self._state_input)

            with tf.variable_scope("discriminator"):
                discriminator = kwargs['cost_fn'](self._state_input)

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

            # Discriminator update
            self._expert = tf.placeholder(dtype=tf.float32, shape=[None])
            self.cost = tf.log(discriminator)

            loss = tf.reduce_sum((1.0 - self._expert) * self._cost + self._expert * tf.log(1.0 - discriminator))
            self._discriminator_update = tf.train.AdamOptimizer(learning_rate=kwargs['learning_rate']).minimize(loss)

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

    def _update_critic(self):
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

    def _update_discriminator(self):
        """
        Updates the discriminator based on recent experience
        """

        samples = []

        for trajectory in self._trajectories:
            samples.extend(trajectory.states)

        for _ in range(self._num_batches):
            states = []
            expert = []

            # Agent batch
            batch = np.random.choice(samples, self._batch_size, replace=False)

            for state in batch:
                states.append(state)
                expert.append(0.0)

            # Expert batch
            batch = np.random.choice(self._data, self._batch_size, replace=False)

            for sample in batch:
                states.append(sample.state)
                expert.append(1.0)

            self._session.run(self._discriminator_update, feed_dict={
                self._state_input: states,
                self._expert: expert
            })

    def _update(self):
        """
        Updates the agent's policy based on recent experience.

        Right now we pass every state through the critic, but this may
        end up being too computationally expensive.

        WE NEED TO ORGANIZE THIS BETTER, RIGHT NOW WE ARE DOING A LOT OF STUPID THINGS
        """

        # Update discriminator
        self._update_discriminator()

        # Update critic
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

        # Update actor

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


class Agent:
    """
    A multi-task GAIL agent.
    """

    def __init__(self, graph, session, kwargs):
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

    def set_task(self, name):
        """
        Sets the task that the agent is currently learning to perform.

        :param name: the name of the task
        """

        if name not in self._tasks:
            self._tasks[name] = TaskAgent(self._graph, self._session, self._kwargs)

        self._current = self._tasks[name]

    def reset(self, task=None):
        """
        Indicates to the agent that a new episode has started, that is
        the current state was sampled independently of the previous
        state.  Also allows for the current task to be set.

        :param task: the name of the current task, if set, will change the task the agent is executing
        """

        if task is not None:
            self.set_task(task)

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
