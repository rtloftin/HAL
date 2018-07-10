# Tests simple behavioral cloning in the roboschool environments

import gym
import numpy as np
import tensorflow as tf
from RoboschoolReacher_v0_2017may import SmallReactivePolicy as ReacherAgent

# Reacher
env = gym.make("RoboschoolReacher-v1")
teacher = ReacherAgent(env.observation_space, env.action_space)

# Hopper
# env = gym.make("RoboschoolHopper-v1")
# teacher = HopperAgent(env.observation_space, env.action_space)

# Ant
# env = gym.make("RoboschoolAnt-v1")
# teacher = AntAgent(env.observation_space, env.action_space)

# Pendulum
# env = gym.make("RoboschoolInvertedPendulumSwingup-v1")
# teacher = PendulumAgent(env.observation_space, env.action_space)

# Generate random baseline
random_episodes = 100
random_return = 0.0

for _ in range(random_episodes):
    env.reset()
    done = False
    step = 0

    while (not done) and (step < 500):
        _, reward, done, _ = env.step(env.action_space.sample())
        random_return += reward
        step += 1

    random_return = random_return / random_episodes

# Generate training data and performance baseline
training_episodes = 1000
training_examples = []

teacher_return = 0.0

for _ in range(training_episodes):
    state = env.reset()
    done = False
    step = 0

    while (not done) and (step < 500):
        action = teacher.act(state)

        training_examples.append({
            "state": state,
            "action": action
        })

        state, reward, done, _ = env.step(action)
        teacher_return += reward
        step += 1

teacher_return = teacher_return / training_episodes

print("Generated Training Data")

# Configure learning agent


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


sess = tf.Session()

state_input = tf.placeholder(tf.float32, [None, env.observation_space.shape[0]], "state")
target_action = tf.placeholder(tf.float32, [None, env.action_space.shape[0]], "action")
agent_action = mlp(env.observation_space.shape[0], env.action_space.shape[0],
                   hidden_layers=2, hidden_nodes=100)(state_input)

cloning_loss = tf.reduce_sum(tf.reduce_sum(tf.square(agent_action - target_action)))
cloning_update = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cloning_loss)

# Train learning agent

num_batches = 5000
batch_size = 50

sess.run(tf.global_variables_initializer())

for _ in range(num_batches):
    batch = np.random.choice(training_examples, batch_size)

    states = []
    actions = []

    for sample in batch:
        states.append(sample["state"])
        actions.append(sample["action"])

    sess.run(cloning_update, feed_dict={
        state_input: states,
        target_action: actions
    })

print("Trained agent")

# Test learning agent
test_episodes = 100
agent_return = 0.0

for _ in range(test_episodes):
    state = env.reset()
    done = False
    step = 0

    while (not done) and (step < 500):
        action = sess.run(agent_action, feed_dict={
            state_input: [state]
        })[0]

        action = np.clip(action, env.action_space.low, env.action_space.high)

        state, reward, done, _ = env.step(action)
        agent_return += reward
        step += 1

agent_return = agent_return / test_episodes

print("Tested agent")
print("Random baseline return: ", random_return)
print("Teacher's average return: ", teacher_return)
print("Agent's average return: ", agent_return)
