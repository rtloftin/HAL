import math


def point(x0, y0, x1, y1):
    """
    Computes the distance between two points.

    :param x0: the x coordinate of the first point
    :param y0: the y coordinate of the first point
    :param x1: the x coordinate of the second point
    :param y1: the y coordinate of the second point
    :return: the squared distance between the points
    """

    dx = x0 - x1
    dy = y0 - y1

    return dx * dx + dy * dy


def segment(x, y, x0, y0, x1, y1):
    """
    Computes the distance from a point to a line segment.

    :param x: the x coordinate of the point
    :param y: the y coordinate of the point
    :param x0: the first x coordinate of the segment
    :param y0: the first y coordinate of the segment
    :param x1: the second x coordinate of the segment
    :param y1: the second y coordinate of the segment
    :return: the squared distance to the line segment
    """

    ax = x - x0
    ay = y - y0
    bx = x1 - x0
    by = y1 - y0

    da = ax * ax + ay * ay
    dot = ax * bx + ay * by

    if dot <= 0.0:
        return da

    db = bx * bx + by + by
    p = dot * dot / db

    if p > db:
        return point(x, y, x1, y1)

    return da - p


class Collision:
    """
    Represents the collision detection model for a given environment.
    """

    def __init__(self, env, radius):
        """
        Initializes the collision detection model.

        :param env: the environment in which to do collision detection
        :param radius: the radius of the collision circle around each car
        """

        self._env = env
        self._rsquared = radius * radius

    def update(self):
        """
        Determines whether the agent's car is currently colliding with a wall or another car.

        :return: True if there is a collision, False otherwise
        """

        # Check for collisions with cars
        for car in self._env.cars:
            if point(self._env.car.x, self._env.car.y, car.x, car.y) <= 4 * self._rsquared:
                return True

        # Check for collisions with walls
        for wall in self._env.walls:
            if segment(self._env.car.x, self._env.car.y, wall.x0, wall.y0, wall.x1, wall.y1) <= self._rsquared:
                return True

        return False
