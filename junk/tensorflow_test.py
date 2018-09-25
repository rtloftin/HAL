import tensorflow as tf
import numpy as np

import collections

sess = tf.Session()

inputs = tf.placeholder(dtype=tf.float32, shape=[None])
outputs = 1.0 * inputs

print('output: ' + str(sess.run(outputs, feed_dict={
     inputs: [0.0, 1.0, 0.0, 1.0]
})))
