import tensorflow as tf
import numpy as np

import collections

print('STARTING SESSION')
sess = tf.Session()
print('SESSION STARTED')

inputs = tf.placeholder(dtype=tf.int32, shape=[None, 2, 2])
outputs = tf.reshape(inputs, [-1, 4])

print('output: ' + str(sess.run(outputs, feed_dict={
    inputs: [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
})))
