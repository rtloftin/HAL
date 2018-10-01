"""
Defines a constructor for linear models.
"""

import tensorflow as tf
import numpy as np


def build_linear(inputs, output_shape):
    """
    Constructs a linear network, which is the degenerate
    case of the MLP with zero hidden layers.

    :param inputs: the input tensors
    :param output_shape: the shape of the output
    :return: the output tensor of the network
    """

    # Construct input
    input_size = 0
    input_vectors = []

    for tensor in inputs:
        size = np.prod(tensor.shape[1:])
        input_vectors.append(tf.reshape(tensor, [-1, size]))
        input_size += size

    # Construct network
    output_size = np.prod(output_shape)

    weights = tf.Variable(tf.random_normal([input_size, output_size],
                                           mean=0.0, stddev=(1.0 / input_size)), name="weights")
    biases = tf.Variable(tf.zeros([output_size], dtype=tf.float32), name="biases")

    outputs = tf.matmul(tf.concat(inputt_vectors, 1), weights) + biases

    return tf.reshape(outputs, [-1] + list(output_shape))


def linear(output_shape):
    """
    Returns a function which builds a linear network using the given input tensors.

    :param output_shape: the shape of the desired output
    :return: the function which builds the network
    """

    return lambda *inputs: build_linear(output_shape, inputs)
