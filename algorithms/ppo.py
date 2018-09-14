"""
Defines a reinforcement learning agent which uses
proximal policy optimization.  Also defines simple
testing experiments for the PPO algorithm to verify
the implementation
"""

import tensorflow as tf
import numpy as np
import gym
import roboschool


class Step:
    """
    Represents a single step in a state-action trajectory.
    """

    def __init__(self, state, action):
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

    def append(self, state, action):
        """
        Adds a new state action pair to the trajectory

        :param state: the state
        :param action: the action
        """

        self.steps.append(Step(state, action))

    def reward(self, reward):
        """
        Sets the immediate reward for the most recent time step.

        :param reward: the immediate reward
        """

        self.steps[-1].reward = reward

    def accumulate(self, gamma):
        """
        Calculates the advantage value for each time step.

        :param gamma: the discount factor
        """

        advantage = 0

        for step in reversed(self.steps):
            advantage = step.reward + (gamma * advantage)
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

        # Determine if output is discrete our continuous
        self._discrete_action = not isinstance(policy_output, list)

        # Configure for discrete or continuous actions
        if self._discrete_action:

            # Define action input
            action_input = tf.placeholder(dtype=tf.int32, shape=[None], name="action_input")

            # Define action output
            action_output = tf.multinomial(policy_output, 1)

            # Define likelihood ratio
            one_hot = tf.one_hot(action_input, policy_output.shape[1])

            policy = tf.exp(policy_output)
            policy = tf.reduce_sum(one_hot * policy, 1) / tf.reduce_sum(policy, 1)

            hypothesis = tf.exp(hypothesis_output)
            hypothesis = tf.reduce_sum(one_hot * hypothesis, 1) / tf.reduce_sum(hypothesis, 1)

            ratio = hypothesis / tf.stop_gradient(policy)

        else:

            # Action space is continuous, so we have a mean and a deviation
            policy_mean = policy_output[0]
            policy_deviation = policy_output[1]

            hypothesis_mean = hypothesis_output[0]
            hypothesis_deviation = tf.exp(hypothesis_output[1])

            # Define action input
            action_input = tf.placeholder(dtype=tf.float32, shape=policy_mean.shape, name="action_input")

            # Define likelihood ratio
            policy = tf.square(action_input - policy_mean) / tf.multiply(tf.square(policy_deviation), 2.0)
            policy = tf.reduce_sum(policy + tf.log(policy_deviation), 1)

            hypothesis = tf.square(action_input - hypothesis_mean) / tf.multiply(tf.square(hypothesis_deviation), 2.0)
            hypothesis = tf.reduce_sum(hypothesis + tf.log(hypothesis_deviation), 1)

            ratio = tf.exp(tf.stop_gradient(policy) - hypothesis)

            # Define action output
            action_output = policy_mean + (policy_deviation * tf.random_normal(tf.shape(policy_deviation)))

        # Define action advantage input
        advantage_input = tf.placeholder(dtype=tf.float32, shape=[None])

        # Define clipped loss function
        clipped_ratio = tf.clip_by_value(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
        loss = -tf.reduce_mean(tf.minimum(ratio * advantage_input, clipped_ratio * advantage_input))

        # Define policy update
        update_hypothesis = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(loss)

        # Define hypothesis to policy transfer op
        transfer_hypothesis = []

        for key in hypothesis_variables.keys():
            transfer_hypothesis.append(tf.assign(policy_variables[key], hypothesis_variables[key]))

        # Start session
        self._sess = tf.Session()

        # Initialize agent
        self._sess.run(tf.global_variables_initializer())
        self._sess.run(transfer_hypothesis)

        # Capture required tensor references
        self._policy_inputs = policy_inputs
        self._hypothesis_inputs = hypothesis_inputs
        self._action_input = action_input
        self._action_output = action_output
        self._advantage_input = advantage_input
        self._transfer_hypothesis = transfer_hypothesis
        self._update_hypothesis = update_hypothesis

        # Capture configuration settings
        self._batch_size = config.batch_size
        self._num_batches = config.num_batches
        self._discount = config.discount

        # Initialize experience buffers
        self._trajectories = []
        self._current_trajectory = None

        # Reset the agent so it treats input as an initial state
        self.reset()

    def update(self):
        """
        Updates the agent's policy based on recent experience.

        TODO
        """

        # Compute advantages
        accumulated = []

        for trajectory in self._trajectories:
            if len(trajectory.steps) > 0:
                trajectory.accumulate(self._discount)
                accumulated.append(trajectory)

        # Perform updates
        for _ in range(self._num_batches):

            # Construct batch
            trajectories = np.random.choice(accumulated, self._batch_size, replace=False)
            batch = []

            for trajectory in trajectories:
                batch += trajectory.steps

            # Construct feed dictionary
            feed_dict = {
                self._advantage_input: [],
                self._action_input: []
            }

            for key in self._policy_inputs.keys():
                feed_dict[self._policy_inputs[key]] = []
                feed_dict[self._hypothesis_inputs[key]] = []

            for sample in batch:
                feed_dict[self._advantage_input].append(sample.advantage)
                feed_dict[self._action_input].append(sample.action)

                for key in self._policy_inputs.keys():
                    feed_dict[self._policy_inputs[key]].append(sample.state[key])
                    feed_dict[self._hypothesis_inputs[key]].append(sample.state[key])

            # Run update
            self._sess.run(self._update_hypothesis, feed_dict=feed_dict)

        # Transfer parameters
        self._sess.run(self._transfer_hypothesis)

        # Resent for next update
        self.clear()

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
        self._trajectories.append(self._current_trajectory)

    def act(self, state):
        """
        Records the current state, and selects the agent's action

        :param state: a representation of the current state
        :return: a representation of the next action
        """

        # Convert the state into a feed dictionary
        feed = {}

        for key in self._policy_inputs.keys():
            feed[self._policy_inputs[key]] = [state[key]]

        # Sample action from policy
        action = self._sess.run(self._action_output, feed_dict=feed)[0]

        # Extend trajectory
        self._current_trajectory.append(state, action)

        return action

    def reward(self, reward):
        """
        Adds a reward signal to the current time step

        :param reward: the reward signal for the current time step
        """

        self._current_trajectory.reward(reward)


class Factory:
    """
    A factory class for constructing identical
    PPO agents.
    """

    def __init__(self,
                 model_source,
                 discount=0.99,
                 learning_rate=0.0005,
                 clip_epsilon=0.05,
                 batch_size=5,
                 num_batches=10):
        """
        Initializes the factory.

        :param model_source:
        :param discount:
        :param learning_rate:
        :param clip_epsilon:
        :param batch_size:
        :param num_batches:
        """

        self.model_source = model_source
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

        return Agent(self)


def roboschool_test():
    """
    Tests this PPO implementation in the Roboschool ant domain
    """

    # Initialize environment
    # env = gym.make("RoboschoolAnt-v1")
    env = gym.make("RoboschoolHopper-v1")

    # Define model structure
    def network():

        hidden_nodes = 70
        hidden_layers = 2

        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]

        # Define state inputs
        state_inputs = {"state": tf.placeholder(dtype=tf.float32, shape=[None, state_size])}

        # Define variables
        input_weights = tf.Variable(tf.random_normal([state_size, hidden_nodes], mean=0.0, stddev=0.5))
        input_bias = tf.Variable(tf.zeros([hidden_nodes]))

        hidden_weights = []
        hidden_biases = []

        for _ in range(hidden_layers - 1):
            hidden_weights.append(tf.Variable(tf.random_normal([hidden_nodes, hidden_nodes], mean=0.0, stddev=0.5)))
            hidden_biases.append(tf.Variable(tf.zeros([hidden_nodes])))

        output_mean_weights = tf.Variable(tf.random_normal([hidden_nodes, action_size], mean=0.0, stddev=0.01))
        output_mean_bias = tf.Variable(tf.zeros([action_size]))

        output_deviation_weights = tf.Variable(tf.random_normal([hidden_nodes, action_size], mean=0.0, stddev=0.001))
        output_deviation_bias = tf.Variable(tf.constant(0.5, dtype=tf.float32, shape=[action_size]))

        variables = {
            "input_weights": input_weights,
            "input_bias": input_bias,
            "output_mean_weights": output_mean_weights,
            "output_mean_bias": output_mean_bias,
            "output_deviation_weights": output_deviation_weights,
            "output_deviation_bias": output_deviation_bias
        }

        for index in range(hidden_layers - 1):
            variables["hidden_weights_" + str(index)] = hidden_weights[index]
            variables["hidden_biases_" + str(index)] = hidden_biases[index]

        # Define network structure
        layer = tf.nn.tanh(tf.add(tf.matmul(state_inputs["state"], input_weights), input_bias))

        for index in range(hidden_layers - 1):
            layer = tf.nn.tanh(tf.add(tf.matmul(layer, hidden_weights[index]), hidden_biases[index]))

        output_mean = tf.add(tf.matmul(layer, output_mean_weights), output_mean_bias)
        # output_deviation = tf.exp(tf.add(tf.matmul(layer, output_deviation_weights), output_deviation_bias))
        output_deviation = tf.square(output_deviation_bias) + tf.constant(0.01, dtype=tf.float32, shape=[action_size])

        action_output = [output_mean, output_deviation]

        return state_inputs, action_output, variables

    # Initialize agent
    agent = Factory(network).build()

    # Run agent
    num_sessions = 1000
    episodes = 30
    max_steps = 500

    for session in range(num_sessions):
        total_return = 0.0

        for _ in range(episodes):
            agent.reset()

            state = env.reset()
            done = False
            step = 0

            while (not done) and (step < max_steps):
                action = agent.act({"state": state})

                # print("Action: " + str(action))

                state, reward, done, _ = env.step(action)
                agent.reward(reward)

                total_return += reward
                ++step

        print("Session " + str(session + 1) + ", total return: " + str(total_return / episodes))
        agent.update()
