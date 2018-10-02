import tensorflow as tf
import numpy as np

import collections

# sess_1 = tf.Session()
# sess_2 = tf.Session()

x = 1


def update():
    x = 0

    for _ in range(10):
        x += 1

    print(x)


update()
print(x)

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
