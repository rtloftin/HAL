

class Task:
    """
    Represents a driving task as a rectangle that the
    agent's car must enter.
    """

    def __init__(self, env, x0, y0, x1, y1, r):
        """
        Initializes the task. The goal region is defined by a
        vector, and a distance to the vector that the car must
        be within to complete the task.  Intuitively, each
        task specifies a lane the car must enter.

        :param env: the environment this task is defined in
        :param x0: the first x coordinate of the vector
        :param y0: the first y coordinate of the vector
        :param x1: the second x coordinate of the vector
        :param y1: the second y coordinate of the vector
        :param r: half the width of the rectangle
        """

        self._env = env
        self._x0 = x0
        self._y0 = y0
        self._x1 = x1
        self._y1 = y1
        self._rsquared = r * r

    def complete(self):
        """
        Determines if the agent has completed the task

        :return: True if the task is complete, False otherwise
        """

        ax = self._env.car.x - self._x0
        ay = self._env.car.y - self._y0
        bx = self._x1 - self._x0
        by = self._y1 - self._y0

        da = ax * ax + ay * ay
        dot = ax * bx + ay * by

        if dot <= 0.0:
            return False

        db = bx * bx + by + by
        p = dot * dot / db

        if p > db:
            return False

        if da - p > self._rsquared:
            return False

        return True
