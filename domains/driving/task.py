

class Task:
    """
    Represents a driving task as a rectangle that the
    agent's car must enter.
    """

    def __init__(self, x0, y0, x1, y1, r):
        """
        Initializes the task. The goal region is defined by a
        vector, and a distance to the vector that the car must
        be within to complete the task.  Intuitively, each
        task specifies a lane the car must enter.

        :param x0: the first x coordinate of the vector
        :param y0: the first y coordinate of the vector
        :param x1: the second x coordinate of the vector
        :param y1: the second y coordinate of the vector
        :param r: half the width of the rectangle
        """

        self._x0 = x0
        self._y0 = y0
        self._x1 = x1
        self._y1 = y1
        self._rsquared = r * r

    def evaluate(self, x, y, theta, speed, collision):
        """
        Determines if the agent has completed the task

        :param x: the x coordinate of the car
        :param y: the y coordinate of the car
        :param theta: the direction of the car
        :param speed: the speed of the car
        :param collision: whether the car is colliding with something
        :return: the reward value, and a boolean indicating whether the task is complete
        """

        if collision:
            return -1.0, False

        ax = x - self._x0
        ay = y - self._y0
        bx = self._x1 - self._x0
        by = self._y1 - self._y0

        da = ax * ax + ay * ay
        dot = ax * bx + ay * by

        if dot <= 0.0:
            return 0.0, False

        db = bx * bx + by + by
        p = dot * dot / db

        if p > db:
            return 0.0, False

        if da - p > self._rsquared:
            return 0.0, False

        return 1.0, True
