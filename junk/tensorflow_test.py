import tensorflow as tf
import numpy as np
import matplotlib.pyplot as pl

from enum import IntEnum
import collections


sess = tf.Session()

values = tf.placeholder(tf.float32, shape=[None, 2])
indices = tf.placeholder(tf.int32, shape=[None, 4])



print(sess.run(results, feed_dict={
    values: [[1, 2], [3, 4], [5, 6], [7, 8]],
    indices: [0, 1, 1, 0]
}))

"""
rewards = tf.placeholder(dtype=tf.float32, shape=[None])
transition_indices = tf.placeholder(dtype=tf.int32, shape=[None, None, None])
transition_probabilities = tf.placeholder(dtype=tf.float32, shape=[None, None, None])

discount = 0.9
steps = 20


def update(values, step):
    q = tf.gather(values, transition_indices)
    q = tf.reduce_sum(transition_probabilities * q, axis=2)
    values = rewards + (discount * tf.reduce_max(q, axis=1))

    return values, 1 + step


def limit(values, step):
    return step < steps


output, _ = tf.while_loop(limit, update, [tf.zeros_like(rewards, dtype=tf.float32), 0])

print(str(sess.run(output, feed_dict={
    rewards: [0.0, 0.0, 1.0, 0.0, 0.0],
    transition_indices: [
        [[4], [1]],
        [[0], [2]],
        [[1], [3]],
        [[2], [4]],
        [[3], [0]]],
    transition_probabilities: [
        [[1.], [1.]],
        [[1.], [1.]],
        [[1.], [1.]],
        [[1.], [1.]],
        [[1.], [1.]]]
})))
"""

"""
examples = tf.placeholder(dtype=tf.float32, shape=[None])
indices = tf.random_uniform([3], maxval=tf.shape(examples)[0], dtype=tf.int32)
batch = tf.gather(examples, indices)

result = sess.run(batch, feed_dict={
    examples: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
})

print("Random batch: " + str(result))
"""

"""
    alpha = tf.placeholder(dtype=tf.float32, shape=[None])
    beta = tf.fill(tf.shape(alpha), 0.9)
    output = tf.scan(lambda a, x: [(x[0] + (x[1] * a[0])), (x[1] * a[1])], [alpha, beta], initializer=[0.0, 1.0])[0]
    
    rewards = [1, 2, 1, -2, 5, 3, 0, -1]
    
    tf_result = sess.run(output, feed_dict={
        alpha: rewards
    })

    python_result = [0] * len(rewards)
    sum = 0.0
    
    for i in range(len(rewards)):
        sum = rewards[i] + (0.9 * sum)
        python_result[i] = sum
    
    python_result = ["{:5.3f}".format(x) for x in python_result]
    tf_result = ["{:5.3f}".format(x) for x in tf_result]
    
    print("Python     : " + str(python_result))
    print("Tensorflow : " + str(tf_result))
"""
