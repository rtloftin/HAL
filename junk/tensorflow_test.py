import tensorflow as tf
import numpy as np

import collections

# sess = tf.Session()
# inputs = tf.placeholder(dtype=tf.int32, shape=[None, 5])
# print(inputs.shape.dims)

queue = collections.deque()

queue.append(1)
queue.append(2)
queue.append(3)

print('queue length: ' + str(len(queue)))
