"""
Defines a reinforcement learning agent which uses
proximal policy optimization.  Also defines simple
testing experiments for the PPO algorithm to verify
the implementation.
"""

import tensorflow as tf
import numpy as np
import gym
import roboschool


class Trajectory:
    """
    Represents a state action trajectory
    """

    def __init__(self):
        self.steps = []

    def append(self, state, action, reward):
        """
        Adds a new step to the trajectory.

        :param state: the state
        :param action: the action
        :param reward: the immediate reward received
        """

        self.steps.append((state, action, reward))

    def accumulate(self, discount):
        """
        Calculates the advantage value for each time step.

        :param discount: the discount factor
        :return: a list of tuples of (state, action, advantage)
        """

        samples = []
        advantage = 0

        for step in reversed(self.steps):
            advantage = step.reward + (discount * advantage)
            samples.append((step.state, step.action, advantage))

        return samples


class Agent:
    """
    An online RL agent which updates its policy
    using a version of the PPO algorithm.
    """

    def __init__(self, kwargs):
        """
        Initializes a new PPO agent.

        :param kwargs: the configuration options for the agent
        """

        # Capture the configuration parameters needed
        self._discrete_action = kwargs['discrete_action']
        self._discount = kwargs['discount']
        self._batch_size = kwargs['batch_size']
        self._num_batches = kwargs['num_batches']
        self._num_episodes = kwargs['num_episodes']

        # Create graph and session
        self._graph = tf.Graph()
        self._session = tf.Session(graph=self._graph)

        # Build the policy network and learning update graph
        with self._graph.as_default():
            self._state_input = tf.placeholder(dtype=tf.float32, shape=[None, kwargs['state_size']])

            with tf.variable_scope("policy"):
                policy_output = kwargs['model_fn'](self._state_input)

            with tf.variable_scope("hypothesis"):
                hypothesis_output = kwargs['model_fn'](self._state_input)

            if self._discrete_action:
                self._action_input = tf.placeholder(dtype=tf.int32, shape=[None])

                one_hot = tf.one_hot(action_input, kwargs['action_size'])

                policy = tf.exp(policy_output)
                policy = tf.reduce_sum(one_hot * policy, 1) / tf.reduce_sum(policy, 1)

                hypothesis = tf.exp(hypothesis_output)
                hypothesis = tf.reduce_sum(one_hot * hypothesis, 1) / tf.reduce_sum(hypothesis, 1)

                ratio = hypothesis / tf.stop_gradient(policy)

                self._action_output = tf.multinomial(policy_output, 1)
            else:
                self._action_input = tf.placeholder(dtype=tf.float32, shape=kwargs['action_size'])

                policy_mean, policy_deviation = tf.split(policy_output, 2, axis=1)
                hypothesis_mean, hypothesis_deviation = tf.split(hypothesis_output, 2, axis=1)

                policy = tf.square(action_input - policy_mean) / tf.exp(policy_deviation)
                policy = tf.reduce_sum(policy + policy_deviation, axis=1)

                hypothesis = tf.square(action_input - hypothesis_mean) / tf.exp(hypothesis_deviation)
                hypothesis = tf.reduce_sum(hypothesis + hypothesis_deviation, axis=1)

                ratio = tf.exp(tf.multiply(tf.stop_gradient(policy) - hypothesis, 0.5))

                noise = tf.random_normal(tf.shape(action_mean))
                self._action = action_mean + (noise * tf.exp(tf.multiply(action_deviation, 0.5)))

            self._advantage_input = tf.placeholder(dtype=tf.float32, shape=[None])

            clipped_ratio = tf.clip_by_value(ratio, 1.0 - kwargs['clip_epsilon'], 1.0 + kwargs['clip_epsilon'])
            loss = -tf.reduce_mean(tf.minimum(ratio * advantage_input, clipped_ratio * advantage_input))

            self._update_hypothesis = tf.train.AdamOptimizer(learning_rate=kwargs['learning_rate']).minimize(loss)

            policy_variables = dict(map(lambda x: (x.name, x), tf.trainable_variables(scope='policy')))
            hypothesis_variables = tf.trainable_variables(scope='hypothesis')

            self._transfer_hypothesis = []

            for var in hypothesis_variables:
                self._transfer_hypothesis.append(tf.assign(policy_variables[var.name], var))

            self._sess.run(tf.global_variables_initializer())
            self._sess.run(self._transfer_hypothesis)

        # Internal state
        self._data = []
        self._trajectory = None
        self._episode_count = 0

        # Reset the agent so it treats the next step as an initial state
        self.reset()

    def update(self):
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
                states.append(sample[0])
                actions.append(sample[1])
                advantages.append(sample[2])

            # Run update
            self._sess.run(self._update_hypothesis, feed_dict={
                self._state_input: states,
                self._action_input: actions,
                self._advantage_input: advantages
            })

        # Transfer parameters
        self._sess.run(self._transfer_hypothesis)

    def new_episode(self):
        """
        Tells the agent that a new episode has been started, the agent may
        choose to run an update at this time.
        """

        if self._trajectory is not None:
            self._data.extend(self._trajectory.accumulate(self._discount))

        self._trajectory = Trajectory()
        self._episode_count += 1

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

    def get_action(self, state):
        """
        Records the current state, and selects the agent's action

        :param state: a representation of the current state
        :return: a representation of the next action
        """

        action = self._sess.run(self._action_output, feed_dict={self._state_input: state})

        if self._discrete_action:
            return action[0, 0]
        else:
            return action[0]


def factory(model_fn, state_size, action_size,
            discrete_action=False,
            discount=0.99,
            learning_rate=0.0005,
            clip_epsilon=0.05,
            batch_size=50,
            num_batches=20,
            num_episodes=10):
    """
    Gets a method which constructs new PPO agents.

    :param model_fn: the function used to build the model graph
    :param state_size: the number of state features
    :param action_size: the number of actions or action features
    :param discrete_action: whether or not the actions are discrete
    :param discount: the discount factor used to estimate advantages
    :param learning_rate: the learning rate used for training the policies
    :param clip_epsilon: the clipping radius for the policy ratio
    :param batch_size: the batch size used for training the policies
    :param num_batches: the number of gradient steps to do per update
    :param num_episodes: the number of episodes performed between updates
    :return: a new PPO reinforcement learning agent
    """

    kwargs = {
        'model_fn': model_fn,
        'state_size': state_size,
        'action_size': action_size,
        'discrete_action': discrete_action,
        'discount': discount,
        'learning_rate': learning_rate,
        'clip_epsilon': clip_epsilon,
        'batch_size': batch_size,
        'num_batches': num_batches,
        'num_episodes': num_episodes
    }

    return lambda: Agent(kwargs)


def roboschool_test():
    """
    Tests this PPO implementation in the Roboschool ant domain

    WE MAY MOVE THIS TEST ELSEWHERE
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
