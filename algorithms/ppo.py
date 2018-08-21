"""
Defines a reinforcement learning agent which uses
proximal policy optimization.  Also defines simple
testing experiments for the PPO algorithm to verify
the implementation
"""

import tensorflow as tf
import numpy as np


class Factory:
    """
    A factory class for constructing identical
    PPO agents.
    """

    def __init__(self,
                 model_source,
                 is_discrete=True,
                 discount=1.0,
                 learning_rate=0.01,
                 clip_epsilon=0.2,
                 batch_size=1,
                 num_batches=50):
        """
        Initializes the factory.

        :param model_source:
        :param is_discrete:
        :param discount:
        :param learning_rate:
        :param clip_epsilon:
        :param batch_size:
        :param num_batches:
        """

        self.model_source = model_source
        self.is_discrete = is_discrete
        self.discount = discount
        self.learning_rate = learning_rate
        self.clip_epsilon = clip_epsilon
        self.batch_size = batch_size
        self.num_batches = num_batches

    def build(self):
        """
        Returns a new PPO agent configured according
        to the factory.  The TensorFlow graph and
        session to be used must be provided.

        :return: an initialized PPO agent
        """


class Step:
    """
    Represents a single step in a state-action trajectory.
    """

    def __init__(self, state, action, reward):
        """
        Creates the step object.

        :param state: the initial state
        :param action: the action taken
        :param reward: the immediate reward received
        """

        self.state = state
        self.action = action
        self.reward = 0.0
        self.advantage = 0.0


class Trajectory:
    """
    Represents a state action trajectory
    """

    def __init__(self):
        self.steps = []

    def append(self, state, action, reward):
        """
        Adds a new state action pair to the trajectory

        :param state: the state
        :param action: the action
        :param reward: the immediate reward
        """

        self._steps.append(Step(state, action, reward))

    def accumulate(self, gamma):
        """
        Calculates the advantage value for each time step.

        :param gamma: the discount factor
        """

        advantage = 0

        for step in reversed(self.steps):
            advantage = step.reward + (discount * advantage)
            step.advantage = advantage

    def random(self):
        """
        Samples a random time-step from this trajectory.

        :return: an object representing the time step
        """

        return self.steps[np.random.randint(0, len(self.steps))]


class Agent:
    """
    An online RL agent which updates its policy
    using a version of the PPO algorithm.
    """

    def __init__(self, config):
        """
        Initializes a new PPO agent.

        :param config: an object containing the settings for this agent.
        """

        # Define policy network
        policy_inputs, policy_output, policy_variables = config.model_source()

        # Define hypothesis network
        hypothesis_inputs, hypothesis_output, hypothesis_variables = config.model_source()

        # Define hypothesis to policy transfer op
        transfer_hypothesis = []

        for key in hypothesis_variables.keys():
            self._transfer_hypothesis.append(tf.assign(policy_variables[key], tf.assign(hypothesis_variables[key])))

        # Define likelihood ratio and action output
        if isinstance(policy_output, list):

            # Action space is continuous, so we have a mean and a deviation
            policy_mean = policy_output[0]
            policy_deviation = policy_output[1]

            hypothesis_mean = hypotheis_output[0]
            hypothesis_deviation = hypothesis_ouptut[1]

            # Define action input
            action_input = tf.placeholder(dtype=tf.float32, shape=policy_mean.shape)

            # Define likelihood ratio
            policy = tf.square(action_input - policy_mean) / tf.mul(tf.square(policy_deviation), 2.0)
            policy = tf.reduce_sum(policy + tf.log(policy_deviation), 1)

            hypothesis = tf.square(action_input - hypothesis_mean) / tf.mul(tf.square(hypothesis_deviation), 2.0)
            hypothesis = tf.reduce_sum(hypothesis + tf.log(hypothesis_deviation), 1)

            ratio = tf.exp(tf.stop_gradient(policy) - hypothesis)

            # Define action output
            action_output = policy_mean + (policy_deviation * tf.random_normal(shape=policy_mean.shape))
            self._is_discrete = False

        else:

            # Action space is discrete, so we only have a single output vector for each action
            action_input = tf.placeholder(dtype=tf.int32, shape=policy_output.shape)

            # Define likelihood ration
            policy = tf.exp(policy_output)
            policy = tf.reduce_sum(action_input * policy, 1) / tf.reduce_sum(policy, 1)

            hypothesis = tf.exp(hypothesis_output)
            hypothesis = tf.reduce_sum(action_input * hypothesis, 1) / tf.reduce_sum(hypothesis, 1)

            ratio = hypothesis / tf.stop_gradient(policy)

            # Define action output
            exponents = tf.exp(policy_output)
            action_output = exponents / tf.reduce_sum(exponents, 1)
            self._is_discrete = True

        # Define action advantage input
        advantage_input = tf.plaeceholder(dtype=tf.float32, shape=[None])

        # Define clipped loss function
        clipped_ratio = tf.clip_by_value(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
        loss = -tf.reduce_mean(tf.min(ratio * advantage_input, clipped_ratio * advantage_input))

        # Define policy update
        update_hypothesis = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

        # Start session
        self._sess = tf.Session()

        # Initialize agent
        sess.run(tf.variables_initializer(hypothesis_variables))
        sess.run(transfer_hypothesis)

        # Capture required instance variables
        self._policy_inputs = policy_inputs
        self._hypothesis_inputs = hypothesis_inputs
        self._action_input = action_input
        self._action_output = action_output
        self._advantage_input = advantage_input
        self._transfer_hypothesis = transfer_hypothesis
        self._update_hypothesis = update_hypothesis

        # Initialize experience buffers
        self._trajectories = []
        self._current_trajectory = None

        # Reset the agent so it treats input as an initial state
        self.reset()

    def update(self):
        """
        Updates the agent's policy based on recent experience.
        """

    def clear(self):
        """
        Clears the agent's experience data so that recent
        experiences won't affect its policy.
        """

        self._trajectories = []
        self.reset()

    def reset(self):
        """
        Tells the agent that the environment has been reset.
        """

        self._current_trajectory = Trajectory()

    def act(self, state):
        """
        Records the current state, and selects the agent's action

        :param state: a representation of the current state
        :return: a representation of the next action
        """

        # Convert the state into a feed dictionary
        input = {}

        for key in self._policy_inputs.keys():
            input[self._policy_inputs[key]] = state[key]

        # Get action output
        action_output = self._sess.run(self._action_output, feed_dict=input)[0]

        if self._is_discrete:
            action_output = np.random.multinomial(1, action_output)
            action = np.eye(self._action_output.shape[1], self._action_output.shape[1])[action_output]
        else:
            action = action_output

        return action_output



    def reward(self, reward):
        """
        Adds a reward signal to the current time step

        :param reward: the reward signal for the current time step
        """
