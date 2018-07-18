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

    def __init__(self, radius):
        """
        Initializes the sensor.

        :param radius: the maximum distance (in cells) to which the sensor can detect
        """

        self._radius = radius

    def update(self, x, y, occupancy_map, sensor_map):
        """
        Updates a sensor map from an occupancy map.

        :param x: the x position of the agent
        :param y: the y position of the agent
        :param occupancy_map: the underlying occupancy map
        :param sensor_map: the sensor map to update
        """

        x_start = max(x - self._radius, 0)
        y_start = max(y - self._radius, 0)
        x_end = min(x + self._radius + 1, occupancy_map.shape[1])
        y_end = min(y + self._radius + 1, occupancy_map.shape[0])

        for x_pos in range(x_start, x_end):
            for y_pos in range(y_start, y_end):
                if occupancy_map[x_pos, y_pos]:
                    sensor_map[x_pos, y_pos] = SensorState.OCCUPIED
                else:
                    sensor_map[x_pos, y_pos] = SensorState.CLEAR


class RoundSensor:
    """
    A sensor which can detect any obstacle within a circular region around the agent.
    """

    def __init__(self, radius):
        """
        Initializes the sensor.

        :param radius: the maximum distance (in cells) to which the sensor can detect
        """

        self._radius = radius
        self._rsquared = radius * radius

    def update(self, x, y, occupancy_map, sensor_map):
        """
        Updates a sensor map from an occupancy map.

        :param x: the x position of the agent
        :param y: the y position of the agent
        :param occupancy_map: the underlying occupancy map
        :param sensor_map: the sensor map to update
        """

        x_start = max(x - self._radius, 0)
        y_start = max(y - self._radius, 0)
        x_end = min(x + self._radius + 1, occupancy_map.shape[1])
        y_end = min(y + self._radius + 1, occupancy_map.shape[0])

        for x_pos in range(x_start, x_end):
            for y_pos in range(y_start, y_end):
                dx = x_pos - x
                dy = y_pos - y

                if dx * dx + dy * dy <= self._rsquared:
                    if occupancy_map[x_pos, y_pos]:
                        sensor_map[x_pos, y_pos] = SensorState.OCCUPIED
                    else:
                        sensor_map[x_pos, y_pos] = SensorState.CLEAR
