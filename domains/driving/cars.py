import math


class DriverCar:
    """
    Represents a car that can be controlled by the agent.
    """

    def __init__(self, x, y, theta, speed):
        """
        Initializes the position, direction, and speed of the car

        :param x: the initial x coordinate of the car
        :param y: the initial y coordinate of the car
        :param theta: the initial direction of the car
        :param speed: the initial speed of the car
        """

        # Define vehicle parameters
        self._min_steering = -math.pi / 10
        self._max_steering = math.pi / 10
        self._max_acceleration = 0.1
        self._min_acceleration = -0.1
        self._max_speed = 1.0
        self._min_speed = 0.0
        self._threshold = 0.05
        self._wheelbase = 0.5

        # Initialize state
        self.x = x
        self.y = y
        self.theta = theta
        self.speed = speed
        self.phi = 0.0

    def update(self, acceleration, steering, delta):
        """
        Updates the state of the car.

        To make things easier, replaced the steering rate (the rate at which
        the steering wheel turns, with the steering angle).  This means that
        if we set the steering angle to zero, the steering angle will be zero
        right away.  This makes it much easier to go in a straight line.

        :param acceleration: acceleration of the vehicle (negative for braking)
        :param steering: steering rate (positive for left negative for right)
        :param delta: the size of the time step for this update
        """

        # Clip acceleration and steering rate
        acceleration = max(self._min_acceleration, min(self._max_acceleration, acceleration))
        steering = max(self._min_steering, min(self._max_steering, steering))

        # Update speed, steering angle
        self.speed = max(self._min_speed, min(self._max_speed, self.speed + acceleration * delta))
        self.phi = max(self._min_steering, min(self._max_steering, steering))

        # Update orientation
        if not (-self._threshold < self.phi < self._threshold):
            self.theta = self.theta - delta * self.speed * math.sin(self.phi) / self._wheelbase

        # Update position
        self.x -= delta * (self.speed * math.sin(self.theta))
        self.y += delta * (self.speed * math.cos(self.theta))


class NPCCar:
    """
    Represents an NPC car, which just moves in a straight line.
    """

    def __init__(self, x, y, theta, speed):
        """
        Initializes the position, direction, and speed of an NPC car.

        :param x: the initial x coordinate of the car
        :param y: the initial y coordinate of the car
        :param theta: the direction of the car
        :param speed: the speed of the car
        """

        # Initialize state
        self.x = x
        self.y = y
        self.theta = theta
        self.speed = speed

    def update(self, delta):
        """
        Updates the position of the car.

        :param delta: the size of the time step for the update
        """

        self.x -= delta * math.sin(self.theta) * self.speed
        self.y += delta * math.cos(self.theta) * self.speed
