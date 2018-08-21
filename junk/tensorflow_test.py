import tensorflow as tf
import numpy as np

sess = tf.Session()

values = tf.placeholder(dtype=tf.float32, shape=[None])
total = tf.reduce_sum(values)

output = sess.run(total, feed_dict={
    values: [1, 2, 3, 4]
})

print("Total: " + str(output))
