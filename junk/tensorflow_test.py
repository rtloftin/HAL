import tensorflow as tf
import numpy as np

x = tf.constant([[1, 1, 1]], dtype=tf.float32)
y = tf.constant([[1, 1, 1]], dtype=tf.float32)

concat = tf.concat([x, y], 1)
sum = tf.reduce_sum(concat)

sess = tf.Session()

total = sess.run(sum)

print("Sum: " + str(total))
