# This script is just a test of a simplified DQN algorithm to get started with RL in TensorFlow

import gym
import tensorflow as tf
import numpy as np

from collections import deque
from matplotlib import pyplot

def td_learn(env, policy, current_state_input, current_value_output, next_state_input, next_value_output):

    # Parameters
    num_episodes = 1000  # number of episodes to run
    step_limit = 200  # max episodes per step
    buffer_size = 1000  # max replay buffer size
    batch_size = 10   # mini-batch size
    alpha = 0.001  # learning rate
    gamma = 0.9  # discount factor

    # Build loss functions
    reward_input = tf.placeholder(tf.float32, [None, 1], "reward")
    terminal_input = tf.placeholder(tf.float32, [None, 1], "terminal")

    td = current_value_output - reward_input - gamma * terminal_input * tf.stop_gradient(next_value_output)
    td_error = tf.reduce_sum(tf.square(td))

    update = tf.train.AdamOptimizer(learning_rate=alpha).minimize(td_error)

    # Start the session
    sess = tf.Session()

    # initialize model parameters
    sess.run(tf.global_variables_initializer())

    # Initialize replay buffer
    buffer = deque()

    # Run learning algorithm
    for episode in range(num_episodes):
        err = 0.0
        steps = 0
        terminal = False

        state = env.reset()

        while (not terminal) and (steps < step_limit):

            # Do state update
            next_state, reward, terminal, _ = env.step(policy(state))
            steps += 1

            # Compute TD error
            err += sess.run(td_error, feed_dict={
                current_state_input: [state],
                reward_input: [[reward]],
                terminal_input: [[0.0] if terminal else [1.0]],
                next_state_input: [next_state]
            })

            # Save sample
            buffer.append({
                "state": state,
                "reward": reward,
                "terminal": terminal,
                "next_state": next_state
            })

            if len(buffer) > buffer_size:
                buffer.popleft()

            # Do batch update
            if len(buffer) >= batch_size:

                # Build mini_batch
                states = []
                rewards = []
                terminals = []
                next_states = []

                batch = np.random.choice(buffer, batch_size)

                for sample in batch:
                    states.append(sample["state"])
                    rewards.append([sample["reward"]])
                    terminals.append([0.0] if sample["terminal"] else [1.0])
                    next_states.append(sample["next_state"])

                # Do update
                sess.run(update, feed_dict={
                    current_state_input: states,
                    reward_input: rewards,
                    terminal_input: terminals,
                    next_state_input: next_states
                })

        print("episode: ", (episode + 1), ", td-error: ", (err / steps))


def q_learn(env, current_state_input, current_value_output, next_state_input, next_value_output):

    # Parameters
    num_epochs = 100  # number of epochs
    num_episodes = 50  # number of episodes per epoch
    step_limit = 200  # max episodes per step
    buffer_size = 10000  # max replay buffer size
    batch_size = 50   # mini-batch size
    alpha = 0.001  # learning rate
    gamma = 0.95  # discount factor

    # Random exploration
    initial_epsilon = 0.1  # initial exploration rate
    final_epsilon = 0.1  # final exploration rate
    exploration_epochs = 10  # number of epochs to explore for

    # Optimistic bias
    initial_penalty = 0.0  # initial regularization weight
    final_penalty = 0.0  # final regularization weight
    penalty_epochs = 30  # number of epochs to apply the penalty for

    penalty_input = tf.placeholder(tf.float32, [1], "penalty")

    # Get the number of actions
    num_actions = env.action_space.n

    # Identity matrix - for action indexes
    index = np.eye(num_actions)

    # build the action selection function
    policy = tf.argmax(current_value_output, axis=1)

    # Build loss functions
    reward_input = tf.placeholder(tf.float32, [None, 1], "reward")
    terminal_input = tf.placeholder(tf.float32, [None, 1], "terminal")
    action_input = tf.placeholder(tf.float32, [None, num_actions], "action")

    q_value = tf.reduce_sum(current_value_output * action_input, axis=1)
    regularizer = penalty_input * tf.reduce_sum(tf.square(q_value - 0.0))
    prediction = reward_input + gamma * terminal_input * tf.reduce_max(next_value_output, axis=1)
    q_loss = tf.reduce_sum(tf.square(q_value - tf.stop_gradient(prediction))) + regularizer

    update = tf.train.AdamOptimizer(learning_rate=alpha).minimize(q_loss)

    # Start the session
    sess = tf.Session()

    # initialize model parameters
    sess.run(tf.global_variables_initializer())

    # Initialize replay buffer
    buffer = deque()

    # Initialize performance store
    performance = []

    # Initialize exploration rate
    epsilon = initial_epsilon

    # Initialize penalty
    penalty = initial_penalty

    # Run learning algorithm
    for epoch in range(num_epochs):
        value = 0.0

        for episode in range(num_episodes):
            steps = 0
            terminal = False

            state = env.reset()

            while (not terminal) and (steps < step_limit):

                # Do state update
                if epsilon <= np.random.ranf():
                    action = env.action_space.sample()
                else:
                    action = sess.run(policy, feed_dict={
                        current_state_input: [state]
                    })[0]

                next_state, reward, terminal, _ = env.step(action)
                value += reward
                steps += 1

                # save sample
                buffer.append({
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "terminal": terminal,
                    "next_state": next_state
                })

                if len(buffer) > buffer_size:
                    buffer.popleft()

                # Do batch update
                if len(buffer) >= batch_size:
                    # Build mini_batch
                    states = []
                    actions = []
                    rewards = []
                    terminals = []
                    next_states = []

                    batch = np.random.choice(buffer, batch_size)

                    for sample in batch:
                        states.append(sample["state"])
                        actions.append(index[sample["action"]])
                        rewards.append([sample["reward"]])
                        terminals.append([0.0] if sample["terminal"] else [1.0])
                        next_states.append(sample["next_state"])

                    # Do update
                    sess.run(update, feed_dict={
                        current_state_input: states,
                        action_input: actions,
                        reward_input: rewards,
                        terminal_input: terminals,
                        next_state_input: next_states,
                        penalty_input: [penalty]
                    })

        value = value / num_episodes
        print("epoch: ", (epoch + 1), ", epsilon: ", epsilon, ", penalty: ", penalty, ", average: ", value)
        performance.append(value)

        if epoch < exploration_epochs:
            epsilon -= (initial_epsilon - final_epsilon) / exploration_epochs

        if epoch < penalty_epochs:
            penalty -= (initial_penalty - final_penalty) / penalty_epochs

    return performance

# def quantized(input_ranges, input_quantities, output_size):

def mlp(input_size, output_size, hidden_layers=1, hidden_nodes=20):
    input_weights = tf.Variable(tf.random_normal([input_size, hidden_nodes], stddev=0.35))
    input_biases = tf.Variable(tf.ones([hidden_nodes]))
    output_weights = tf.Variable(tf.random_normal([hidden_nodes, output_size], stddev=0.35))
    output_bias = tf.Variable(tf.ones([output_size]))

    weights = []
    biases = []

    for _ in range(hidden_layers - 1):
        weights.append(tf.Variable(tf.random_normal([hidden_nodes, hidden_nodes], stddev=0.35)))
        biases.append(tf.Variable(tf.ones([hidden_nodes])))

    def build(input):
        value = tf.nn.relu(tf.add(tf.matmul(input, input_weights), input_biases))

        for l in range(hidden_layers - 1):
            value = tf.nn.relu(tf.add(tf.matmul(value, weights[l]), biases[l]))

        return tf.add(tf.matmul(value, output_weights), output_bias)

    return build


# Build environment
# env = gym.make("LunarLander-v2")
# env = gym.make("Acrobot-v1")
# env = gym.make("CartPole-v1")
env = gym.make("MountainCar-v0")

state_dim = env.observation_space.shape[0]
num_actions = env.action_space.n

# Build q network
build_network = mlp(state_dim, num_actions, hidden_layers=1, hidden_nodes=10)
# build_network = mlp(state_dim, 1, hidden_layers=1, hidden_nodes=50)

state = tf.placeholder(tf.float32, [None, state_dim], "state")
next_state = tf.placeholder(tf.float32, [None, state_dim], "next_state")

value = build_network(state)
next_value = build_network(next_state)


def mcar_policy(state):
    if state[1] > 0:
        return 2
    else:
        return 0


def cpole_policy(state):
    return env.action_space.sample()


# td_learn(env, cpole_policy, state, value, next_state, next_value)

performance = q_learn(env, current_state_input=state, current_value_output=value,
        next_state_input=next_state, next_value_output=next_value)

pyplot.plot(performance)
pyplot.show()
