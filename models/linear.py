"""
Defines a constructor for linear models.
"""

import tensorflow as tf
import numpy as np


def build_linear(inputs, input_shape, output_shape):
    """
    Constructs a linear network, which is the degenerate
    case of the MLP with zero hidden layers.

    :param inputs: the input tensor
    :param input_shape: the shape of the input
    :param output_shape: the shape of the output
    :return: the output tensor of the network
    """

    input_size = np.prod(input_shape)
    output_size = np.prod(output_shape)

    weights = tf.Variable(tf.random_normal([input_size, output_size],
                                           mean=0.0, stddev=(1.0 / input_size)), name="weights")
    biases = tf.Variable(tf.zeros([output_size], dtype=tf.float32), name="biases")

    inputs = tf.reshape(inputs, [-1, input_size])
    outputs = tf.matmul(inputs, weights) + biases

    return tf.reshape(outputs, [-1] + list(output_shape))


def linear(input_shape, output_shape):
    """
    Returns a function which builds a linear network using the given input layer.

    :param input_shape: the shape of the input
    :param output_shape: the shape of the output
    :return: the function which builds the network
    """

    return lambda x: build_linear(x, input_shape, output_shape)
