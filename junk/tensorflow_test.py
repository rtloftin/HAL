import tensorflow as tf
import numpy as np

sess = tf.Session()

inputs = tf.placeholder(dtype=tf.float32, shape=[None])
x = tf.Variable(tf.zeros(shape=[None], dtype=tf.float32))
y = tf.square(x)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.assign(x, inputs), feed_dict={inputs: [1, 2, 3, 4, 5]})

output = sess.run(y)

print("Total: " + str(output))
