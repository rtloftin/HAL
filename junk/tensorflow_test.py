import tensorflow as tf
import numpy as np

import collections

sess = tf.Session()

input = tf.placeholder(dtype=tf.float32, shape=[None])
var = tf.get_variable("var", dtype=tf.float32, shape=[None], use_resource=True)
assign = tf.assign(var, input)

sess.run(assign, feed_dict={input: [1., 2., 3.]})

# input_sequence = tf.placeholder(dtype=tf.float32, shape=[None])
# sum_sequence = tf.scan(lambda s, x: s + x, input_sequence, 0.0)

# items = tf.Variable([], dtype=tf.float32, use_resource=True)
# sums = tf.Variable([], dtype=tf.float32, use_resource=True)

# initialize_a = [tf.assign(items, input_sequence, validate_shape=False, use_locking=True)]
# initialize_b = [tf.assign(sums, sum_sequence, validate_shape=False, use_locking=True)]
# append = [tf.assign(items, tf.concat([items, input_sequence], 0), validate_shape=False, use_locking=True),
#          tf.assign(sums, tf.concat([sums, sum_sequence], 0), validate_shape=False, use_locking=True)]

# batch_indices = tf.random_uniform([4], maxval=tf.shape(items)[0], dtype=tf.int32)
# batch_items = tf.gather(items, batch_indices)
# batch_sums = tf.gather(sums, batch_indices)

# sess.run([initialize_b, initialize_a], feed_dict={input_sequence: [1., 2., 3.]})
# sess.run(initialize_a, feed_dict={input_sequence: [1., 2., 3.]})


# for _ in range(3):
#     sess.run(append, feed_dict={input_sequence: [1., 2., 3.]})

# [itms, sms] = sess.run([batch_items, batch_sums])

# print("Items: " + str(itms))
# print("Sums:  " + str(sms))

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
