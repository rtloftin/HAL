# Tests simple behavioral cloning in the roboschool environments

from OpenGL import GLU
import time
import gym
import tensorflow as tf
import numpy as np
from RoboschoolReacher_v0_2017may import SmallReactivePolicy as ReacherAgent
from RoboschoolHopper_v0_2017may import SmallReactivePolicy as HopperAgent
from RoboschoolAnt_v0_2017may import SmallReactivePolicy as AntAgent
from RoboschoolInvertedPendulumSwingup_v0_2017may import SmallReactivePolicy as PendulumAgent

# Reacher
# env = gym.make("RoboschoolReacher-v1")
# teacher = ReacherAgent(env.observation_space, env.action_space)

# Hopper
# env = gym.make("RoboschoolHopper-v1")
# teacher = HopperAgent(env.observation_space, env.action_space)

# Ant
# env = gym.make("RoboschoolAnt-v1")
# teacher = AntAgent(env.observation_space, env.action_space)

# Pendulum
env = gym.make("RoboschoolInvertedPendulumSwingup-v1")
teacher = PendulumAgent(env.observation_space, env.action_space)

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

# Generate training data - we will use the cloning loss on the policy

training_episodes = 10
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

# Initialize policy, value, and reward networks


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



# Do training - now it involves interacting with the environment
