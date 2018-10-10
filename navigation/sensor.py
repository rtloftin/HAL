import numpy as np

from enum import IntEnum


class SensorState(IntEnum):
    """
    An enum describing possible states of the sensor map.
    """

    UNKNOWN = 0
    CLEAR = 1
    OCCUPIED = 2


class SquareSensor:
    """
    A sensor which can detect any obstacle within a square region around the agent.
    """

    def __init__(self, env, radius, map=None):
        """
        Initializes the sensor.

        :param env: the navigation environment this sensor operates in
        :param radius: the maximum distance (in cells) to which the sensor can detect
        :param map: optional initial values for the occupancy map
        """

        self._env = env
        self._radius = radius

        if map is None:
            self._map = np.full_like(env.occupied, SensorState.UNKNOWN)
        else:
            self._map = np.copy(map)

    def update(self):
        """
        Updates a sensor map based on the agent's current position
        """

        x = self._env.x
        y = self._env.y
        occupied = self._env.occupied

        x_start = max(x - self._radius, 0)
        y_start = max(y - self._radius, 0)
        x_end = min(x + self._radius + 1, self._env.width)
        y_end = min(y + self._radius + 1, self._env.height)

        for x_pos in range(x_start, x_end):
            for y_pos in range(y_start, y_end):
                if occupied[x_pos, y_pos]:
                    self._map[x_pos, y_pos] = SensorState.OCCUPIED
                else:
                    self._map[x_pos, y_pos] = SensorState.CLEAR

    def clone(self):
        """
        Creates a new sensor with the same map as this sensor, but
        which is not updated when this sensor is updated.

        :return: a copy of this sensor
        """

        return SquareSensor(self._env, self._radius, map=self._map)

    @property
    def map(self):
        return self._map

    @property
    def width(self):
        return self._env.width

    @property
    def height(self):
        return self._env.height


class RoundSensor:
    """
    A sensor which can detect any obstacle within a circular region around the agent.
    """

    def __init__(self, env, radius, map=None):
        """
        Initializes the sensor.

        :param env: the navigation environment this sensor operates in
        :param radius: the maximum distance (in cells) to which the sensor can detect
        :param map: optional initial values for the occupancy map
        """

        self._env = env
        self._radius = radius
        self._rsquared = radius * radius

        if map is None:
            self._map = np.full_like(env.occupied, SensorState.UNKNOWN)
        else:
            self._map = np.copy(map)

    def update(self):
        """
        Updates a sensor map based on the agent's current position
        """

        x = self._env.x
        y = self._env.y
        occupied = self._env.occupied

        x_start = max(x - self._radius, 0)
        y_start = max(y - self._radius, 0)
        x_end = min(x + self._radius + 1, self._env.width)
        y_end = min(y + self._radius + 1, self._env.height)

        for x_pos in range(x_start, x_end):
            for y_pos in range(y_start, y_end):
                dx = x_pos - x
                dy = y_pos - y

                if (dx * dx) + (dy * dy) <= self._rsquared:
                    if occupied[x_pos, y_pos]:
                        self._map[x_pos, y_pos] = SensorState.OCCUPIED
                    else:
                        self._map[x_pos, y_pos] = SensorState.CLEAR

    def clone(self):
        """
        Creates a new sensor with the same map as this sensor, but
        which is not updated when this sensor is updated.

        :return: a copy of this sensor
        """

        return RoundSensor(self._env, self._radius, map=self._map)

    @property
    def map(self):
        return self._map

    @property
    def width(self):
        return self._env.width

    @property
    def height(self):
        return self._env.height
