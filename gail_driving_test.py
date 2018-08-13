"""
This scripts runs the GAIL algorithm on
the driving domain, and evaluates the results.
"""

import domains.driving as dr
import tensorflow as tf
import numpy as np

# Initialize environment
env = dr.highway(npc=False)
env.set_task("exit")

# Generate data set
training_episodes = 10
time_step = 0.05
data = []

for _ in range(training_episodes):
    env.reset()

    while not env.complete:

        # Get expert behavior
        acceleration, steering = env.expert()

        # Add training data
        data.append({
            'sensor': np.copy(env.sensor),
            'speed': env.speed,
            'x': env.x,
            'y': env.y,
            'direction': env.direction,
            'acceleration': acceleration,
            'steering': steering
        })

        # print(str(env.sensor))

        # Update environment
        env.update(acceleration, steering, time_step)

print("training samples: " + str(len(data)))

# Initialize behavior model - basic mlp regression for now
sess = tf.Session()

input_size = env.sensor.size + 1
# input_size = env.sensor.size + 4
# input_size = 4

print("input size: " + str(input_size))

output_size = 2
hidden_layers = 2
hidden_nodes = 200

input_weights = tf.Variable(tf.random_normal([input_size, hidden_nodes], stddev=0.5))
input_biases = tf.Variable(tf.zeros([hidden_nodes]))
output_weights = tf.Variable(tf.random_normal([hidden_nodes, output_size], stddev=0.05))
output_bias = tf.Variable(tf.zeros([output_size]))

weights = []
biases = []

for _ in range(hidden_layers - 1):
    weights.append(tf.Variable(tf.random_normal([hidden_nodes, hidden_nodes], stddev=0.5)))
    biases.append(tf.Variable(tf.zeros([hidden_nodes])))

sensor_input = tf.placeholder(tf.float32, [None, env.sensor.size], "sensor")
speed_input = tf.placeholder(tf.float32, [None, 1], "speed")
x_input = tf.placeholder(tf.float32, [None, 1], "x")
y_input = tf.placeholder(tf.float32, [None, 1], "y")
direction_input = tf.placeholder(tf.float32, [None, 1], "direction")

target_acceleration = tf.placeholder(tf.float32, [None, 1], "acceleration")
target_steering = tf.placeholder(tf.float32, [None, 1], "steering")

inpt = tf.concat([speed_input, sensor_input], 1)
# inpt = tf.concat([speed_input, x_input, y_input, direction_input, sensor_input], 1)
# inpt = tf.concat([speed_input, x_input, y_input, direction_input], 1)
layer = tf.nn.relu(tf.add(tf.matmul(inpt, input_weights), input_biases))

for index in range(hidden_layers - 1):
    layer = tf.nn.relu(tf.add(tf.matmul(layer, weights[index]), biases[index]))

output = tf.add(tf.matmul(layer, output_weights), output_bias)
agent_acceleration = tf.slice(output, [0, 0], [-1, 1])
agent_steering = tf.slice(output, [0, 1], [-1, 1])

# Debug variables
accel = tf.reduce_mean(agent_acceleration)
steer = tf.reduce_mean(agent_steering)

taccel = tf.reduce_mean(target_acceleration)
tsteer = tf.reduce_mean(target_steering)

maccel = tf.reduce_max(tf.abs(target_acceleration))
msteer = tf.reduce_max(tf.abs(target_steering))

errors = tf.square(agent_acceleration - target_acceleration) + tf.square(agent_steering - target_steering)
max_loss = tf.reduce_max(errors)
loss = tf.reduce_mean(errors)

update = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

# Train behavior model

num_batches = 5000
batch_size = 5000

sess.run(tf.global_variables_initializer())

# print("initial parameters")
# print("input weights: " + str(sess.run(input_weights)))
# print("output weights: " + str(sess.run(output_weights)))

for index in range(num_batches):
    batch = np.random.choice(data, batch_size)

    sensor_batch = []
    speed_batch = []
    acceleration_batch = []
    steering_batch = []
    x_batch = []
    y_batch = []
    direction_batch = []

    for sample in batch:
        sensor_batch.append(sample['sensor'])
        speed_batch.append([sample['speed']])
        acceleration_batch.append([sample['acceleration']])
        steering_batch.append([sample['steering']])
        x_batch.append([sample['x']])
        y_batch.append([sample['y']])
        direction_batch.append([sample['direction']])

    status = sess.run([loss, accel, steer, taccel, tsteer, max_loss, maccel, msteer], feed_dict={
        sensor_input: sensor_batch,
        speed_input: speed_batch,
        x_input: x_batch,
        y_input: y_batch,
        direction_input: direction_batch,
        target_acceleration: acceleration_batch,
        target_steering: steering_batch
    })

    print('Batch ' + str(index) + ", loss: " + str(status[0]))
    # print("Maximum error: " + str(status[5]))
    # print('Mean acceleration: ' + str(status[1]) + ', Mean steering: ' + str(status[2]))
    # print('Target acceleration: ' + str(status[3]) + ', Target steering: ' + str(status[4]))
    # print('Max acceleration: ' + str(status[6]) + ', Max steering: ' + str(status[7]))

    sess.run(update, feed_dict={
        sensor_input: sensor_batch,
        speed_input: speed_batch,
        x_input: x_batch,
        y_input: y_batch,
        direction_input: direction_batch,
        target_acceleration: acceleration_batch,
        target_steering: steering_batch
    })

# print("final parameters")
# print("input weights: " + str(sess.run(input_weights)))
# print("output weights: " + str(sess.run(output_weights)))

# Test behavior -- need to impose a
max_steps = 500
evaluation_episodes = 50

total_successes = 0.0

for episode in range(evaluation_episodes):
    env.reset()
    steps = 0

    print("Episode: " + str(episode))

    while steps < max_steps and not env.complete:
        steps += 1

        acceleration, steering = sess.run([agent_acceleration, agent_steering], feed_dict={
            sensor_input: [env.sensor],
            speed_input: [[env.speed]],
            x_input: [[env.x]],
            y_input: [[env.y]],
            direction_input: [[env.direction]],
        })

        # print('Acceleration: ' + str(acceleration))
        # print('Steering: ' + str(steering))

        env.update(acceleration[0, 0], steering[0, 0], time_step)

    if 0.0 < env.reward:
        total_successes += 1.0

print('Success rate: ' + str(100 * (total_successes / evaluation_episodes)))
