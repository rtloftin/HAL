"""
Defines classes used to represent the state
and action spaces of our domains.
"""

import numpy as np


class Discrete:
    """
    A discrete space of a given size.
    """

    def __init__(self, size):
        """
        Initializes the space by defining its size.

        :param size: the number of items in the space
        """

        self._size = size

    @property
    def size(self):
        return self._size

    @property
    def discrete(self):
        return True


class Continuous:
    """
    A continuous space with arbitrary shape.
    """

    def __init__(self, shape, low=0, high=1):
        """
        Initializes the space with the given shape and
        upper and lower bounds.

        :param shape: the shape of elements of the space
        :param low: the lower bounds on the space - if this is a single number, all dimensions are the same
        :param high: the upper bounds on the space - if this is a single number, all dimensions are the same
        """

        self._shape = shape

        if isinstance(low, (int, float)):
            self._low = np.full(shape, low)
        else:
            self._low = low

        if isinstance(high, (int, float)):
            self._high = np.full(shape, high)
        else:
            self._high = high

    @property
    def shape(self):
        return self._shape

    @property
    def low(self):
        return self._low

    @property
    def high(self):
        return self._high

    @property
    def discrete(self):
        return False
