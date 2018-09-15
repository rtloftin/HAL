import tensorflow as tf
import numpy as np

inputs = tf.placeholder(dtype=tf.float32, shape=[None, 4])
output = tf.multinomial(inputs, 1)

sess = tf.Session()
choice = sess.run(output, feed_dict={inputs: [[1.0, 0.5, 2.0, 1.0]]})

print("choice: " + str(choice))
