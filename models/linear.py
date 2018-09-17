"""
Defines a constructor for linear models.
"""

import tensorflow as tf


def build_linear(inputs, input_size, output_size):
    """
    Constructs a linear network, which is the degenerate
    case of the MLP with zero hidden layers.

    :param inputs: the input tensor
    :param input_size: the number of input values
    :param output_size: the number of output values
    :return: the output tensor of the network
    """

    weights = tf.Variable(tf.random_normal([input_size, output_size],
                                           mean=0.0, stddev=(1.0 / input_size)), name="weights")
    biases = tf.Variable(tf.zeros([output_size], dtype=tf.float32), name="biases")

    return tf.matmul(inputs, weights) + biases

def linear(input_size, output_size):
    """
    Returns a function which builds a linear network using the given input layer.

    :param input_size: the number of input values
    :param output_size: the number of output values
    :return: the function which builds the network
    """

    return lambda x: build_linear(x, input_size, output_size)