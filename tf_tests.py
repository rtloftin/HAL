
import tensorflow as tf

input = tf.placeholder(tf.float32, [None, 1], "input")
quantized = tf.quantize()
