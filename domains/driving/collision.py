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

    def __init__(self, car, cars, walls, width, height, radius):
        """
        Initializes the collision detection model.

        :param car: the agent's car
        :param cars: the NPC cars
        :param walls: the walls
        :param width: the width of the environment
        :param height: the height of the environment
        :param radius: the radius of the collision circle around each car
        """

        self._car = car
        self._cars = cars
        self._walls = walls
        self._width = width
        self._height = height
        self._rsquared = radius * radius

        self.is_collision = False

    def update(self):
        """
        Determines whether the agent's car is currently colliding with a wall or another car.
        """

        # Check if the car is within the environment boundaries
        if self._car.x < 0.0 or self._car.x > self._width:
            self.is_collision = True
            return self.is_collision

        if self._car.y < 0.0 or self._car.y > self._height:
            self.is_collision = True
            return self.is_collision

        # Check for collisions with cars
        for car in self._cars:
            if point(self._car.x, self._car.y, car.x, car.y) <= 4 * self._rsquared:
                self.is_collision = True
                return self.is_collision

        # Check for collisions with walls
        for wall in self._walls:
            if segment(self._car.x, self._car.y, wall.x0, wall.y0, wall.x1, wall.y1) <= self._rsquared:
                self.is_collision = True
                return self.is_collision

        self.is_collision = False
        return self.is_collision
