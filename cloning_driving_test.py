"""
This scripts runs behavioral cloning on
the driving domain, and evaluates the results.

Here we use regression rather than classification
for action selection, since the driving controllers
are not themselves discretized.
"""

import domains.driving as dr
import tensorflow as tf
import numpy as np

# Initialize environment
env = dr.intersection()
env.set_task("left")

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
            'sensor': env.sensor,
            'speed': env.speed,
            'acceleration': acceleration,
            'steering': steering
        })

        print(str(env.sensor))

        # Update environment
        env.update(acceleration, steering, time_step)

# Initialize behavior model - basic mlp regression for now
sess = tf.Session()

input_size = env.sensor.size + 1
output_size = 2
hidden_layers = 2
hidden_nodes = 500

input_weights = tf.Variable(tf.random_normal([input_size, hidden_nodes], stddev=0.05))
input_biases = tf.Variable(tf.zeros([hidden_nodes]))
output_weights = tf.Variable(tf.random_normal([hidden_nodes, output_size], stddev=0.05))
output_bias = tf.Variable(tf.zeros([output_size]))

weights = []
biases = []

for _ in range(hidden_layers - 1):
    weights.append(tf.Variable(tf.random_normal([hidden_nodes, hidden_nodes], stddev=0.05)))
    biases.append(tf.Variable(tf.zeros([hidden_nodes])))

sensor_input = tf.placeholder(tf.float32, [None, env.sensor.size], "sensor")
speed_input = tf.placeholder(tf.float32, [None, 1], "speed")

target_acceleration = tf.placeholder(tf.float32, [None, 1], "acceleration")
target_steering = tf.placeholder(tf.float32, [None, 1], "steering")

input = tf.concat([speed_input, sensor_input], 1)
layer = tf.nn.relu(tf.add(tf.matmul(input, input_weights), input_biases))

for index in range(hidden_layers - 1):
    layer = tf.nn.relu(tf.add(tf.matmul(layer, weights[index]), biases[index]))

output = tf.add(tf.matmul(layer, output_weights), output_bias)
agent_acceleration = tf.slice(output, [0, 0], [-1, 1])
agent_steering = tf.slice(output, [0, 1], [-1, 1])


errors = tf.square(agent_acceleration - target_acceleration) + tf.square(agent_steering - target_steering)
loss = tf.reduce_sum(errors)

# update = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.7).minimize(loss)
update = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
# update = tf.train.AdagradOptimizer(learning_rate=0.0001).minimize(loss)

# Train behavior model

num_batches = 1000
batch_size = 2000

sess.run(tf.global_variables_initializer())

for index in range(num_batches):
    batch = np.random.choice(data, batch_size)

    sensor_batch = []
    speed_batch = []
    acceleration_batch = []
    steering_batch = []

    for sample in batch:
        sensor_batch.append(sample['sensor'])
        speed_batch.append([sample['speed']])
        acceleration_batch.append([sample['acceleration']])
        steering_batch.append([sample['steering']])

    batch_loss = sess.run(loss, feed_dict={
        sensor_input: sensor_batch,
        speed_input: speed_batch,
        target_acceleration: acceleration_batch,
        target_steering: steering_batch
    })

    print('Batch ' + str(index) + ", loss: " + str(batch_loss))

    sess.run(update, feed_dict={
        sensor_input: sensor_batch,
        speed_input: speed_batch,
        target_acceleration: acceleration_batch,
        target_steering: steering_batch
    })

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
            speed_input: [[env.speed]]
        })

        env.update(acceleration[0, 0], steering[0, 0], time_step)

    if 0.0 < env.reward:
        total_successes += 1.0

print('Success rate: ' + str(100 * (total_successes / evaluation_episodes)))
