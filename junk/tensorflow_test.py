import tensorflow as tf
import numpy as np

# sess = tf.Session()

inputs = tf.placeholder(dtype=tf.int32, shape=[None, 5])

print(inputs.shape.dims)
