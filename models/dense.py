"""
Defines constructors for dense multilayer perceptrons.
"""

import tensorflow as tf
import numpy as np


def build_dense(inputs, input_shape, output_shape, hidden_layers, hidden_nodes, activation):
    """
    Constructs a dense multilayer perceptron.

    :param inputs: the input tensor
    :param input_shape: the shape of the input
    :param output_shape: the shape of the output
    :param hidden_nodes: the number of hidden nodes per layer
    :param hidden_layers: the number of hidden layers
    :param activation: the activation function to use
    :return: the output tensor of the network
    """

    input_size = np.prod(input_shape)
    output_size = np.prod(output_shape)

    input_weights = tf.Variable(tf.random_normal([input_size, hidden_nodes],
                                           mean=0.0, stddev=(1.0 / input_size)), name="input_weights")
    input_biases = tf.Variable(tf.zeros([hidden_nodes], dtype=tf.float32), name="input_biases")

    output_weights = tf.Variable(tf.random_normal([hidden_nodes, output_size],
                                           mean=0.0, stddev=(1.0 / hidden_nodes)), name="output_weights")
    output_biases = tf.Variable(tf.zeros([output_size], dtype=tf.float32), name="output_biases")

    hidden_weights = []
    hidden_biases = []

    for l in range(hidden_layers - 1):
        hidden_weights.append(tf.Variable(tf.random_normal([hidden_nodes, hidden_nodes], mean=0.0,
                                                           stddev=(1.0 / hidden_nodes)), name=("hidden_weights_" + str(l))))
        hidden_biases.append(tf.Variable(tf.zeros([hidden_nodes], dtype=tf.float32), name=("hidden_biases_" + str(l))))

    inputs = tf.reshape(inputs, [-1, input_size])
    layer = activation(tf.matmul(inputs, input_weights) + input_biases)

    for l in range(hidden_layers - 1):
        layer = activation(tf.matmul(layer, hidden_weights[l]) + hidden_biases[l])

    outputs = tf.matmul(layer, output_weights) + output_biases

    return tf.reshape(outputs, [-1] + list(output_shape))


def sigmoid(input_shape, output_shape, hidden_layers=1, hidden_nodes=50):
    """
    Returns a function which builds a network of the specified
    structure taking a given tensor as the input layer.  The network
    will use sigmoid activation functions.

    :param input_shape: the shape of the input
    :param output_shape: the shape of the output
    :param hidden_layers: the number of hidden layers
    :param hidden_nodes: the number of nodes per hidden layer
    :return: the function which builds the network
    """

    def activation(x):
        return tf.sigmoid(x)

    return lambda x: build_dense(x, input_shape, output_shape, hidden_layers, hidden_nodes, activation)


def tanh(input_shape, output_shape, hidden_layers=1, hidden_nodes=50):
    """
    Returns a function which builds a network of the specified
    structure taking a given tensor as the input layer.  The network
    will use hyperbolic tangent activation functions.

    :param input_shape: the shape of the input
    :param output_shape: the shape of the output
    :param hidden_layers: the number of hidden layers
    :param hidden_nodes: the number of nodes per hidden layer
    :return: the function which builds the network
    """

    def activation(x):
        return tf.tanh(x)

    return lambda x: build_dense(x, input_shape, output_shape, hidden_layers, hidden_nodes, activation)


def relu(input_shape, output_shape, hidden_layers=1, hidden_nodes=50):
    """
    Returns a function which builds a network of the specified
    structure taking a given tensor as the input layer.  The network
    will use rectified linear activation functions.

    :param input_shape: the shape of the input
    :param output_shape: the shape of the output
    :param hidden_layers: the number of hidden layers
    :param hidden_nodes: the number of nodes per hidden layer
    :return: the function which builds the network
    """

    def activation(x):
        return tf.relu(x)

    return lambda x: build_dense(x, input_shape, output_shape, hidden_layers, hidden_nodes, activation)
