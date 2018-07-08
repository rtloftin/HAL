import pyglet as pg
import math


class Environment:
    """
    Represents a simple simulated driving domain

    Right now focus on the driving dynamics, just how the car moves under human control
    """

    def __init__(self):
        """
        Initializes the car's position and orientation.
        """

        self.width = 10
        self.height = 10

        self.x = 5.0
        self.y = 5.0
        self.theta = 0.0
        self.phi = 0.0
        self.speed = 0.0

        self._min_steering = -math.pi / 10
        self._max_steering = math.pi / 10

        self._max_acceleration = 0.1
        self._min_acceleration = -0.1

        self._max_speed = 1.0
        self._min_speed = 0.0

        self._threshold = 0.05

        self._wheelbase = 0.5

    def update(self, acceleration, steering, delta):
        """
        Updates the state of the environment according to the
        dynamics equations.

        :param acceleration: acceleration of the vehicle (negative for braking)
        :param steering: steering rate (positive for left negative for right)
        :param delta: the size of the time step for this update
        """

        # Clip acceleration and steering rate
        acceleration = max(self._min_acceleration, min(self._max_acceleration, acceleration))
        steering = max(self._min_steering, min(self._max_steering, steering))

        # Update speed, steering angle
        self.speed = max(self._min_speed, min(self._max_speed, self.speed + acceleration * delta))
        self.phi = max(self._min_steering, min(self._max_steering, self.phi + steering * delta))

        # Update orientation
        if not (-self._threshold < self.phi < self._threshold):
            self.theta = self.theta - delta * self.speed * math.sin(self.phi) / self._wheelbase

        # Update position
        self.x = self.x - delta * (self.speed * math.sin(self.theta))
        self.y = self.y + delta * (self.speed * math.cos(self.theta))

        self.x = max(0.0, min(self.width, self.x))
        self.y = max(0.0, min(self.height, self.y))
