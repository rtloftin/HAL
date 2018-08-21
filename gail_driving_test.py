"""
This scripts runs the GAIL algorithm on
the driving domain, and evaluates the results.
"""

import domains.driving as dr
import tensorflow as tf
import numpy as np
from types import SimpleNamespace

# Initialize environment
env = dr.highway(npc=False)
env.set_task("exit")

# Generate data set
training_episodes = 10
time_step = 0.05

teacher_sensors = []
teacher_speeds = []
teacher_accelerations = []
teacher_steerings = []

for _ in range(training_episodes):
    env.reset()

    while not env.complete:

        # Get expert behavior
        acceleration, steering = env.expert()

        # Add training data
        teacher_sensors.append(np.copy(env.sensor))
        teacher_speeds.append([env.speed])
        teacher_accelerations.append([acceleration])
        teacher_steerings.append([steering])

        # Update environment
        env.update(acceleration, steering, time_step)

print("training samples: " + str(len(teacher_sensors)))

# Build actor and discriminator models
sess = tf.Session()

# Discriminator
discriminator_hidden_nodes = 100
discriminator_hidden_layers = 2

discriminator_input_weights = tf.Variable(
    tf.random_normal([env.sensor.size + 3, discriminator_hidden_nodes], stddev=0.5))
discriminator_input_biases = tf.Variable(tf.zeros([discriminator_hidden_nodes]))
discriminator_output_weights = tf.Variable(tf.random_normal([discriminator_hidden_nodes, 1], stddev=0.5))
discriminator_output_bias = tf.Variable(tf.zeros([1]))

discriminator_weights = []
discriminator_biases = []

for _ in range(discriminator_hidden_layers - 1):
    discriminator_weights.append(
        tf.Variable(tf.random_normal([discriminator_hidden_nodes, discriminator_hidden_nodes], stddev=0.5)))
    discriminator_biases.append(tf.Variable(tf.zeros([discriminator_hidden_nodes])))

# Agent path
agent_sensor = tf.placeholder(tf.float32, [None, env.sensor.size])
agent_speed = tf.placeholder(tf.float32, [None, 1])
agent_acceleration = tf.placeholder(tf.float32, [None, 1])
agent_steering = tf.placeholder(tf.float32, [None, 1])
agent_input = tf.concat([agent_speed, agent_acceleration, agent_steering, agent_sensor], 1)

layer = tf.nn.relu(tf.add(tf.matmul(agent_input, discriminator_input_weights), discriminator_input_biases))

for index in range(discriminator_hidden_layers - 1):
    layer = tf.nn.relu(tf.add(tf.matmul(layer, discriminator_weights[index]), discriminator_biases[index]))

agent_output = tf.add(tf.matmul(layer, discriminator_output_weights), discriminator_output_bias)
agent_loss = tf.log_sigmoid(agent_output)

# Teacher path
teacher_sensor = tf.placeholder(tf.float32, [None, env.sensor.size])
teacher_speed = tf.placeholder(tf.float32, [None, 1])
teacher_acceleration = tf.placeholder(tf.float32, [None, 1])
teacher_steering = tf.placeholder(tf.float32, [None, 1])
teacher_input = tf.concat([teacher_speed, teacher_acceleration, teacher_steering, teacher_sensor], 1)

layer = tf.nn.relu(tf.add(tf.matmul(teacher_input, discriminator_input_weights), discriminator_input_biases))

for index in range(discriminator_hidden_layers - 1):
    layer = tf.nn.relu(tf.add(tf.matmul(layer, discriminator_weights[index]), discriminator_biases[index]))

teacher_output = tf.add(tf.matmul(layer, discriminator_output_weights), discriminator_output_bias)

# Loss and update
discriminator_loss = tf.reduce_mean(tf.log_sigmoid(teacher_output)) + tf.reduce_mean(tf.log_sigmoid(-agent_output))
discriminator_update = tf.train.AdamOptimizer(learning_rate=0.01).minimize(discriminator_loss)

# Actor
actor_hidden_nodes = 100
actor_hidden_layers = 2

actor_input_weights = tf.Variable(
    tf.random_normal([env.sensor.size + 1, actor_hidden_nodes], stddev=0.5))
actor_input_biases = tf.Variable(tf.zeros([actor_hidden_nodes]))
actor_output_weights = tf.Variable(tf.random_normal([actor_hidden_nodes, 1], stddev=0.5))
actor_output_bias = tf.Variable(tf.zeros([4]))

actor_weights = []
actor_biases = []

for _ in range(actor_hidden_layers - 1):
    actor_weights.append(tf.Variable(tf.random_normal([actor_hidden_nodes, actor_hidden_nodes], stddev=0.5)))
    actor_biases.append(tf.Variable(tf.zeros([actor_hidden_nodes])))

sensor_input = tf.placeholder(tf.float32, [None, env.sensor.size])
speed_input = tf.placeholder(tf.float32, [None, 1])
state_input = tf.concat([speed_input, sensor_input], 1)

acceleration = tf.placeholder(tf.float32, [None, 1])
steering = tf.placeholder(tf.float32, [None, 1])
advantage = tf.placeholder(tf.float32, [None, 1])

layer = tf.nn.relu(tf.add(tf.matmul(state_input, actor_input_weights), actor_input_biases))

for index in range(actor_hidden_layers - 1):
    layer = tf.nn.relu(tf.add(tf.matmul(layer, actor_weights[index]), actor_biases[index]))

actor_output = tf.add(tf.matmul(layer, actor_output_weights), actor_output_bias)
acceleration_mean = tf.slice(actor_output, [0, 0], [-1, 1])
acceleration_variance = tf.slice(actor_output, [0, 0], [-1, 1])
steering_mean = tf.slice(actor_output, [0, 0], [-1, 1])
steering_variance = tf.slice(actor_output, [0, 0], [-1, 1])

acceleration_output = (acceleration_mean +
                       tf.sqrt(acceleration_variance) * tf.random_normal(tf.shape(acceleration_mean), mean=0, stddev=0.5))
steering_output = (steering_mean + tf.sqrt(steering_variance) * tf.random_normal(tf.shape(steering_mean), mean=0, stddev=0.5))

actor_probability = ((tf.square(acceleration - acceleration_mean) / acceleration_variance)
                     + (tf.square(steering - steering_mean) / steering_variance))
actor_ratio = tf.exp(tf.stop_gradient(actor_probability) - actor_probability)
actor_loss = tf.reduce_mean(tf.minimum(actor_ratio * advantage, tf.clip_by_value(actor_ratio, 0.8, 1.2) * advantage))
actor_update = tf.train.AdamOptimizer(learning_rate=0.01).minimize(actor_loss)

# Run GAIL algorithm

num_episodes = 5000
max_steps = 500
total_successes = 0

sess.run(tf.global_variables_initializer())

for episode in range(num_episodes):
    actor_sensors = []
    actor_speeds = []
    actor_accelerations = []
    actor_steerings = []

    env.reset()
    steps = 0

    # run episode
    while steps < max_steps and not env.complete:
        steps += 1

        acc, steer = sess.run([acceleration_output, steering_output], feed_dict={
            sensor_input: [env.sensor],
            speed_input: [[env.speed]],
        })

        actor_sensors.append(np.copy(env.sensor))
        actor_speeds.append([env.speed])
        actor_accelerations.append([acc[0, 0]])
        actor_steerings.append([steer[0, 0]])

        env.update(acc[0, 0], steer[0, 0], time_step)

    # print success rate
    if 0.0 < env.reward:
        total_successes += 1.0

    print("episode " + str(episode) + ", success rate: " + str(100 * total_successes / (episode + 1)) + "%")

    # compute advantages
    value = 0
    advantages = [0] * len(actor_sensors)

    for step in range(len(actor_sensors) - 1, -1, -1):
        advantages[step] = [value]

        adv = sess.run(agent_loss, feed_dict={
            agent_sensor: [actor_sensors[step]],
            agent_speed: [actor_speeds[step]],
            agent_acceleration: [actor_accelerations[step]],
            agent_steering: [actor_steerings[step]]
        })

        value += adv[0, 0]

    # perform updates
    sess.run(discriminator_update, feed_dict={
        teacher_sensor: teacher_sensors,
        teacher_speed: teacher_speeds,
        teacher_acceleration: teacher_accelerations,
        teacher_steering: teacher_steerings,
        agent_sensor: actor_sensors,
        agent_speed: actor_speeds,
        agent_acceleration: actor_accelerations,
        agent_steering: actor_steerings
    })

    sess.run(actor_update, feed_dict={
        sensor_input: actor_sensors,
        speed_input: actor_speeds,
        acceleration: actor_accelerations,
        steering: actor_steerings,
        advantage: advantages
    })
